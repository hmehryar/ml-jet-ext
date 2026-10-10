#!/usr/bin/env python3
"""Create reproducible ML-JET-Ext splits from cleaned M1-M5 metadata.

Default design: 80% train, 10% validation, 10% held-out test per
physics configuration, plus five CV folds of the 90% development pool.

Source-jet overlap is checked within each configuration. For an exhaustive
cross-configuration check, pass --overlap-scope global (may be slower).
No files are written until validation has completed successfully.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("agg_file_path", "source_file_paths", "alpha", "q0", "config")


def parse_sources(value):
    if isinstance(value, list):
        paths = value
    elif pd.isna(value):
        paths = []
    else:
        s = str(value).strip()
        if s.startswith("[") and s.endswith("]"):
            paths = ast.literal_eval(s)
            if not isinstance(paths, (list, tuple)):
                raise ValueError("Source-path literal must be a list or tuple")
        elif "|" in s:
            paths = s.split("|")
        else:
            paths = [s]
    return [str(p).strip() for p in paths if str(p).strip()]


def checked_relative_path(value):
    p = Path(str(value))
    if not str(value).strip() or p.is_absolute() or ".." in p.parts:
        raise ValueError(f"Unsafe aggregate relative path: {value!r}")
    return p


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_source_membership(frames, scope, expected_sources, db_path=None):
    """Return overlap counts and examples; do not silently permit overlap.

    Per-config mode retains source-jet owners only for one config at a time.
    Global mode uses an on-disk SQLite uniqueness index (no huge RAM map).
    """
    overlaps = 0
    examples = []
    total_memberships = 0
    unique_sources = 0
    conn = None
    if scope == "global":
        if db_path is None:
            raise ValueError("Internal error: global mode needs a temporary DB")
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE seen (source_path TEXT PRIMARY KEY, aggregate_path TEXT NOT NULL)")

    try:
        for config, df in frames:
            seen = {} if scope == "per-config" else None
            batch = []
            for row in df.itertuples(index=False):
                # Access via explicit column names so CSV column order is irrelevant.
                aggregate = str(getattr(row, "agg_file_path"))
                sources = parse_sources(getattr(row, "source_file_paths"))
                if len(sources) != expected_sources or len(set(sources)) != expected_sources:
                    raise ValueError(
                        f"{aggregate}: expected {expected_sources} DISTINCT source jets; "
                        f"found {len(sources)} entries, {len(set(sources))} unique"
                    )
                total_memberships += len(sources)
                if scope == "per-config":
                    for source in sources:
                        previous = seen.setdefault(source, aggregate)
                        if previous != aggregate:
                            overlaps += 1
                            if len(examples) < 8:
                                examples.append({"source": source, "first": previous, "second": aggregate})
                else:
                    for source in sources:
                        batch.append((source, aggregate))
                    if len(batch) >= 4000:
                        for source, owner in batch:
                            try:
                                conn.execute("INSERT INTO seen VALUES (?, ?)", (source, owner))
                            except sqlite3.IntegrityError:
                                previous = conn.execute(
                                    "SELECT aggregate_path FROM seen WHERE source_path=?", (source,)
                                ).fetchone()[0]
                                if previous != owner:
                                    overlaps += 1
                                    if len(examples) < 8:
                                        examples.append({"source": source, "first": previous, "second": owner})
                        batch.clear()
            if scope == "per-config":
                unique_sources += len(seen)
            else:
                for source, owner in batch:
                    try:
                        conn.execute("INSERT INTO seen VALUES (?, ?)", (source, owner))
                    except sqlite3.IntegrityError:
                        previous = conn.execute(
                            "SELECT aggregate_path FROM seen WHERE source_path=?", (source,)
                        ).fetchone()[0]
                        if previous != owner:
                            overlaps += 1
                            if len(examples) < 8:
                                examples.append({"source": source, "first": previous, "second": owner})
                conn.commit()
        if conn is not None:
            unique_sources = conn.execute("SELECT COUNT(*) FROM seen").fetchone()[0]
    finally:
        if conn is not None:
            conn.close()
    return {
        "scope": scope,
        "memberships_examined": total_memberships,
        "unique_sources_in_scope": unique_sources,
        "overlapping_source_memberships": overlaps,
        "examples": examples,
        "cross_configuration_overlap_checked": scope == "global",
    }


def verify_partitions(partitions, expected_all_paths):
    named_sets = {}
    for name, frame in partitions.items():
        paths = frame["agg_file_path"].astype(str).tolist()
        if len(paths) != len(set(paths)):
            raise AssertionError(f"Duplicate aggregate paths in {name}")
        named_sets[name] = set(paths)
    names = list(named_sets)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            if named_sets[first] & named_sets[second]:
                raise AssertionError(f"Overlap between {first} and {second}")
    if set().union(*named_sets.values()) != expected_all_paths:
        raise AssertionError("Split union differs from input aggregates")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--metadata-dir", default="metadata")
    parser.add_argument("--output-dir", default="splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-configs", type=int, default=35)
    parser.add_argument("--expected-per-config", type=int, default=400)
    parser.add_argument("--expected-sources", type=int, default=500)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--overlap-scope", choices=("per-config", "global"), default="per-config")
    parser.add_argument("--allow-source-overlap", action="store_true",
                        help="Explicitly permit shared source jets, recording a warning in manifest")
    parser.add_argument("--force", action="store_true", help="Replace a pre-existing splits directory")
    args = parser.parse_args()

    root = args.dataset_root.expanduser().resolve()
    metadata = root / args.metadata_dir
    output = root / args.output_dir
    if not metadata.is_dir():
        parser.error(f"Metadata directory not found: {metadata}")
    if args.folds < 2:
        parser.error("--folds must be >= 2")
    if not (0 < args.train_fraction < 1 and 0 < args.val_fraction < 1):
        parser.error("Train and validation fractions must lie between 0 and 1")
    test_fraction = 1 - args.train_fraction - args.val_fraction
    if test_fraction <= 0:
        parser.error("train + validation fractions must be < 1")
    if output.exists() and not args.force:
        parser.error(f"Output already exists: {output} (use --force to replace)")

    files = sorted(metadata.glob("*_agg500_metadata.csv"))
    if len(files) != args.expected_configs:
        raise ValueError(f"Expected {args.expected_configs} CSVs; found {len(files)}")

    frames = []
    seen_paths = set()
    columns = None
    config_stats = {}
    for path in files:
        config = path.name.removesuffix("_agg500_metadata.csv")
        df = pd.read_csv(path)
        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
        if columns is None:
            columns = list(df.columns)
        elif set(df.columns) != set(columns):
            raise ValueError(f"{path.name}: inconsistent CSV columns")
        df = df[columns].copy()
        if len(df) != args.expected_per_config:
            raise ValueError(f"{config}: {len(df)} rows, expected {args.expected_per_config}")
        if df["agg_file_path"].duplicated().any():
            raise ValueError(f"{config}: duplicate aggregate paths")
        if df["config"].astype(str).ne(config).any():
            raise ValueError(f"{config}: configuration label mismatch")
        if df["alpha"].nunique() != 1 or df["q0"].nunique() != 1:
            raise ValueError(f"{config}: alpha/q0 vary within configuration")
        for rel in df["agg_file_path"]:
            p = checked_relative_path(rel)
            if p.parts[0] != config:
                raise ValueError(f"{config}: aggregate path in unexpected folder: {rel}")
            if not (root / p).is_file():
                raise FileNotFoundError(root / p)
            if str(p) in seen_paths:
                raise ValueError(f"Aggregate path reused across configurations: {p}")
            seen_paths.add(str(p))
        config_stats[config] = {"samples": len(df), "alpha": float(df["alpha"].iloc[0]),
                                "q0": float(df["q0"].iloc[0])}
        frames.append((config, df))
    label_pairs = {(v["alpha"], v["q0"]) for v in config_stats.values()}
    if len(label_pairs) != len(frames):
        raise ValueError("Multiple configurations share identical (alpha, q0) labels")

    print(f"Verified {len(frames)} configurations, {len(seen_paths)} unique aggregate files")
    print(f"Checking {args.expected_sources} source jets per aggregate ({args.overlap_scope})...")
    with tempfile.TemporaryDirectory(prefix="mljet_split_audit_") as tmp:
        overlap = verify_source_membership(
            frames, args.overlap_scope, args.expected_sources,
            Path(tmp) / "sources.sqlite" if args.overlap_scope == "global" else None,
        )
    print("Source-overlap occurrences:", overlap["overlapping_source_memberships"])
    if overlap["examples"]:
        for item in overlap["examples"][:3]:
            print("  example:", item)
    if overlap["overlapping_source_memberships"] and not args.allow_source_overlap:
        raise ValueError("Source-jet overlap detected: NO splits created. "
                         "Review overlapping groups or explicitly use --allow-source-overlap.")

    # We require exactly representable per-configuration sizes; no silent rounding.
    size = args.expected_per_config
    train_exact, val_exact = size * args.train_fraction, size * args.val_fraction
    if abs(train_exact - round(train_exact)) > 1e-8 or abs(val_exact - round(val_exact)) > 1e-8:
        raise ValueError("Fractions do not produce integer per-configuration counts")
    n_train, n_val = round(train_exact), round(val_exact)
    n_test = size - n_train - n_val
    n_dev = n_train + n_val
    if n_dev % args.folds:
        raise ValueError(f"Development samples per configuration ({n_dev}) not divisible by {args.folds}")

    rng = np.random.default_rng(args.seed)
    buckets = {"train": [], "val": [], "test": []}
    cv_parts = [{"train": [], "val": []} for _ in range(args.folds)]
    for config, df in frames:
        # Stable sorting before seeded shuffle makes results independent of CSV row order.
        ordered = df.sort_values("agg_file_path", kind="stable").reset_index(drop=True)
        shuffled = ordered.iloc[rng.permutation(len(ordered))].reset_index(drop=True)
        train, val, test = (shuffled.iloc[:n_train],
                            shuffled.iloc[n_train:n_dev], shuffled.iloc[n_dev:])
        buckets["train"].append(train)
        buckets["val"].append(val)
        buckets["test"].append(test)
        development = pd.concat([train, val], ignore_index=True)
        indices = np.array_split(np.arange(n_dev), args.folds)
        for i, val_indices in enumerate(indices):
            cv_parts[i]["val"].append(development.iloc[val_indices])
            train_indices = np.setdiff1d(np.arange(n_dev), val_indices, assume_unique=True)
            cv_parts[i]["train"].append(development.iloc[train_indices])

    splits = {key: pd.concat(parts, ignore_index=True) for key, parts in buckets.items()}
    verify_partitions(splits, seen_paths)
    test_paths = set(splits["test"]["agg_file_path"])
    dev_paths = set(splits["train"]["agg_file_path"]) | set(splits["val"]["agg_file_path"])
    folds = []
    all_fold_val_paths = []
    for i, parts in enumerate(cv_parts):
        fold = {key: pd.concat(frames_, ignore_index=True) for key, frames_ in parts.items()}
        verify_partitions(fold, dev_paths)
        for part in fold.values():
            if set(part["agg_file_path"]) & test_paths:
                raise AssertionError("Held-out test samples leaked into cross-validation")
        all_fold_val_paths.extend(fold["val"]["agg_file_path"].tolist())
        folds.append(fold)
    if len(all_fold_val_paths) != len(set(all_fold_val_paths)) or set(all_fold_val_paths) != dev_paths:
        raise AssertionError("Each development aggregate must appear in exactly one CV validation fold")

    # Stage outputs first, then replace destination once all writes succeed.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mljet_splits_", dir=output.parent) as tmp:
        staged = Path(tmp) / "splits"
        staged.mkdir()
        digest = {}
        for name, frame in splits.items():
            target = staged / f"{name}.csv"
            frame.to_csv(target, index=False)
            digest[target.name] = sha256(target)
        fold_dir = staged / "five_fold"
        fold_dir.mkdir()
        for i, fold in enumerate(folds):
            for part, frame in fold.items():
                target = fold_dir / f"fold_{i}_{part}.csv"
                frame.to_csv(target, index=False)
                digest[f"five_fold/{target.name}"] = sha256(target)
        manifest = {
            "schema_version": 1,
            "dataset_root": str(root),
            "metadata_dir": args.metadata_dir,
            "seed": args.seed,
            "fractions": {"train": args.train_fraction, "val": args.val_fraction,
                          "test": test_fraction},
            "counts": {key: len(frame) for key, frame in splits.items()},
            "per_configuration_counts": {"train": n_train, "val": n_val, "test": n_test},
            "cv_folds": args.folds,
            "cv_pool": "train+val only; test is never used by CV",
            "cv_validation_per_configuration": n_dev // args.folds,
            "source_overlap": overlap,
            "source_overlap_accepted": bool(overlap["overlapping_source_memberships"] and args.allow_source_overlap),
            "warning": ("Source overlap was allowed and can bias evaluation" if overlap["overlapping_source_memberships"]
                        else ("Cross-configuration source overlap was not checked" if args.overlap_scope == "per-config" else None)),
            "configurations": config_stats,
            "split_file_sha256": digest,
        }
        (staged / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if output.exists():
            # --force was explicitly requested, but don't overwrite arbitrary files.
            import shutil
            shutil.rmtree(output)
        os.rename(staged, output)
    print(f"\nSaved split files to: {output}")
    print(f"Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")
    print(f"CV: {args.folds} folds, {n_dev // args.folds} validation samples per config/fold")
    print("NOTE: CV folds use train+val pool; held-out test is untouched.")
    if args.overlap_scope == "per-config":
        print("NOTE: source overlap was checked within configs only; use --overlap-scope global for a full audit.")


if __name__ == "__main__":
    main()
