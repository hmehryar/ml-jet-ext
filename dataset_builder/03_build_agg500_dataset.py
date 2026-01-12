#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


# ----------------------------------------------------------
# Log normalization
# ----------------------------------------------------------
def log_normalize_image(img, epsilon=1e-8, img_max=1.0):
    """
    Apply log-based normalization:

        I_log = (log(I + eps) - log(eps)) / (log(img_max + eps) - log(eps))

    Then clip to [0,1].

    Assumes img is non-negative. If img_max <= 0, returns zeros.
    """
    I = np.asarray(img, dtype=np.float32)
    if img_max <= 0:
        return np.zeros_like(I, dtype=np.float32)

    num = np.log(I + epsilon) - math.log(epsilon)
    den = math.log(img_max + epsilon) - math.log(epsilon)
    if den <= 0:
        return np.zeros_like(I, dtype=np.float32)

    I_log = num / den
    return np.clip(I_log, 0.0, 1.0)


# ----------------------------------------------------------
# 500-jet aggregation for a single config
# ----------------------------------------------------------
def build_agg500_for_config(
    config_name: str,
    output_root: Path,
    metadata_dir: Path,
    n_agg: int = 400,
    group_size: int = 500,
    max_pt: float = 110.0,
    epsilon: float = 1e-8,
    rng_seed: int = 0,
):
    """
    For one config (e.g., c0005_alpha0p20_q_1p0_unfolded):

      - Read <config>_jets.csv from metadata_dir.
      - Shuffle all jets and create n_agg disjoint groups of size group_size.
      - For each group:
          * Load 500 .npy jets
          * Compute mean image
          * Normalize by max_pt
          * Log-normalize with epsilon, img_max=1.0
          * Save agg image in config directory
          * Record metadata row

      - Write per-config metadata file:
          <config>_agg500_metadata.csv in metadata_dir
    """
    jets_csv = metadata_dir / f"{config_name}_jets.csv"
    if not jets_csv.exists():
        raise FileNotFoundError(f"Jets CSV not found: {jets_csv}")

    df = pd.read_csv(jets_csv)
    if "file_paths" not in df.columns:
        raise ValueError(f"'file_paths' column not found in {jets_csv}")

    n_total = len(df)
    required = n_agg * group_size
    if n_total < required:
        raise ValueError(
            f"{config_name}: need {required} jets (for {n_agg} groups of {group_size}), "
            f"but only found {n_total}."
        )

    print(f"[INFO] Config {config_name}: {n_total} jets, building {n_agg} groups of {group_size}")

    # Random permutation of all jet indices
    rng = np.random.RandomState(rng_seed)
    perm = rng.permutation(n_total)
    used_indices = perm[:required]
    groups = used_indices.reshape(n_agg, group_size)

    config_dir = output_root / config_name
    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    agg_dir = config_dir  # same folder as original config, no seed subfolders
    agg_dir.mkdir(parents=True, exist_ok=True)

    agg_rows = []

    # label columns: keep useful labels, drop pure technical columns
    drop_cols = {
        "file_paths",
        "phi_min",
        "phi_max",
        "eta_min",
        "eta_max",
        "pT_min",
        "pT_max",
        "seed",
    }
    label_cols = [c for c in df.columns if c not in drop_cols]

    for agg_idx, idx_group in enumerate(groups):
        df_group = df.iloc[idx_group]

        # --- Load and aggregate 500 jets ---
        imgs = []
        src_paths = []

        for _, row in df_group.iterrows():
            rel_path = row["file_paths"]
            img_path = output_root / rel_path
            if not img_path.exists():
                raise FileNotFoundError(f"Missing jet .npy file: {img_path}")
            arr = np.load(img_path).astype(np.float32)
            imgs.append(arr)
            src_paths.append(rel_path)

        imgs = np.stack(imgs, axis=0)  # (500, H, W)
        mean_img = imgs.mean(axis=0)   # (H, W)

        # Normalize by MAX_PT
        mean_norm = mean_img / float(max_pt)

        # Log-normalize with img_max=1.0
        mean_log = log_normalize_image(mean_norm, epsilon=epsilon, img_max=1.0)

        # --- Save aggregated image ---
        agg_fname = f"agg500_{agg_idx:04d}.npy"
        agg_path = agg_dir / agg_fname
        np.save(agg_path, mean_log)

        # relative path to aggregated image (w.r.t. output_root)
        agg_rel = agg_path.relative_to(output_root)

        # labels: take from the first jet in the group (assuming same label per config)
        label_values = {}
        if label_cols:
            first_row = df_group.iloc[0]
            label_values = {col: first_row[col] for col in label_cols}

        # store metadata row
        agg_rows.append(
            {
                "config": config_name,
                "agg_file_path": str(agg_rel),
                "source_file_paths": "|".join(src_paths),
                **label_values,
            }
        )

        if (agg_idx + 1) % 50 == 0:
            print(f"[INFO]   {config_name}: built {agg_idx + 1}/{n_agg} aggregated jets")

    # --- Write per-config agg500 metadata CSV ---
    out_csv = metadata_dir / f"{config_name}_agg500_metadata.csv"
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(out_csv, index=False)

    print(f"[INFO]   {config_name}: wrote agg500 metadata to {out_csv}")
    return out_csv


# ----------------------------------------------------------
# Stratified 5-fold split on combined agg500 metadata
# ----------------------------------------------------------
def make_stratified_folds(
    metadata_dir: Path,
    out_prefix: str = "agg500",
    label_col: str = "energy_loss",
    n_folds: int = 5,
    rng_seed: int = 0,
):
    """
    - Read all *_agg500_metadata.csv in metadata_dir
    - Concatenate, shuffle
    - Make stratified n_folds by label_col
    - Write:
        <out_prefix>_all.csv
        <out_prefix>_stratified_fold0.csv ... fold{n_folds-1}.csv
    """
    agg_csv_files = sorted(metadata_dir.glob("*_agg500_metadata.csv"))
    if not agg_csv_files:
        raise FileNotFoundError(f"No *_agg500_metadata.csv found in {metadata_dir}")

    dfs = []
    for p in agg_csv_files:
        df = pd.read_csv(p)
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in {p}")
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Combined agg500 dataset: {len(all_df)} samples from {len(agg_csv_files)} configs")

    # Shuffle all rows once
    all_df = all_df.sample(frac=1.0, random_state=rng_seed).reset_index(drop=True)

    # Save combined CSV
    all_csv = metadata_dir / f"{out_prefix}_all.csv"
    all_df.to_csv(all_csv, index=False)
    print(f"[INFO] Wrote combined CSV: {all_csv}")

    # Stratified fold assignment (manual, no sklearn dependency)
    rng = np.random.RandomState(rng_seed)
    all_df["fold"] = -1

    for label_value, group_df in all_df.groupby(label_col):
        idxs = group_df.index.to_list()
        rng.shuffle(idxs)

        for i, idx in enumerate(idxs):
            fold_id = i % n_folds
            all_df.at[idx, "fold"] = fold_id

    # sanity check
    if (all_df["fold"] < 0).any():
        raise RuntimeError("Some rows did not get a fold assignment.")

    # Write per-fold CSVs
    for f in range(n_folds):
        fold_df = all_df[all_df["fold"] == f].reset_index(drop=True)
        fold_csv = metadata_dir / f"{out_prefix}_stratified_fold{f}.csv"
        fold_df.to_csv(fold_csv, index=False)
        print(f"[INFO] Wrote fold {f}: {len(fold_df)} samples -> {fold_csv}")


# ----------------------------------------------------------
# Main CLI
# ----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Build 500-jet aggregated dataset and stratified 5-fold splits."
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Root directory containing config folders and 'metadata/'",
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of config names (e.g. c0005_alpha0p20_q_1p0_unfolded). "
             "If omitted, infer from *_jets.csv in metadata/",
    )
    parser.add_argument(
        "--n-agg",
        type=int,
        default=400,
        help="Number of aggregated jets per config (default: 400)",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=500,
        help="Number of jets per aggregate (default: 500)",
    )
    parser.add_argument(
        "--max-pt",
        type=float,
        default=110.0,
        help="Fixed MAX_PT for per-aggregate normalization (default: 110.0)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
        help="Epsilon for log normalization (default: 1e-8)",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=0,
        help="Random seed for shuffling (default: 0)",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="energy_loss",
        help="Column name used for stratified folds (default: 'energy_loss')",
    )
    parser.add_argument(
        "--skip-folds",
        action="store_true",
        help="If set, do not build global combined CSV and stratified folds.",
    )

    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser().resolve()
    metadata_dir = output_root / "metadata"

    if not metadata_dir.exists():
        raise FileNotFoundError(f"Metadata directory not found: {metadata_dir}")

    # Determine configs
    if args.configs:
        configs = args.configs
    else:
        # infer from *_jets.csv
        jets_files = sorted(metadata_dir.glob("*_jets.csv"))
        if not jets_files:
            raise FileNotFoundError(f"No '*_jets.csv' found in {metadata_dir}")
        configs = [p.name.replace("_jets.csv", "") for p in jets_files]

    print(f"[INFO] Output root: {output_root}")
    print(f"[INFO] Metadata dir: {metadata_dir}")
    print(f"[INFO] Configs: {configs}")

    # Build agg500 per config
    for cfg in configs:
        build_agg500_for_config(
            config_name=cfg,
            output_root=output_root,
            metadata_dir=metadata_dir,
            n_agg=args.n_agg,
            group_size=args.group_size,
            max_pt=args.max_pt,
            epsilon=args.epsilon,
            rng_seed=args.rng_seed,
        )

    # Build combined CSV + stratified folds
    if not args.skip_folds:
        make_stratified_folds(
            metadata_dir=metadata_dir,
            out_prefix="agg500",
            label_col=args.label_col,
            n_folds=5,
            rng_seed=args.rng_seed,
        )


if __name__ == "__main__":
    main()
