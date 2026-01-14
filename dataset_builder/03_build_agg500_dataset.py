#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


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


def build_agg500_for_config(
    config_name: str,
    input_root: Path,
    agg_root: Path,
    n_agg: int = 400,
    group_size: int = 500,
    max_pt: float = 110.0,
    epsilon: float = 1e-8,
    rng_seed: int = 0,
    ):
    """
    Build 500-jet aggregates for a single config.

    - Read <config_name>_jets.csv from input_root/metadata.
    - Shuffle all jets, form n_agg disjoint groups of size group_size.
    - For each group:
        * Load 500 .npy jets from input_root
        * Compute mean image
        * Normalize by max_pt
        * Log-normalize with epsilon, img_max=1.0
        * Save agg500_event_XXXXXXX.npy in agg_root/config_name
        * Record metadata row
    - Write <config_name>_agg500_metadata.csv into agg_root/metadata.
    """
    metadata_dir = input_root / "metadata"
    jets_csv = metadata_dir / f"{config_name}_unfolded_jets.csv"
    if not jets_csv.exists():
        raise FileNotFoundError(f"Jets CSV not found: {jets_csv}")

    df = pd.read_csv(jets_csv)
    if "file_paths" not in df.columns:
        raise ValueError(f"'file_paths' column not found in {jets_csv}")

    # n_total = len(df)
    # required = n_agg * group_size
    # if n_total < required:
    #     raise ValueError(
    #         f"{config_name}: need {required} jets (for {n_agg} groups of {group_size}), "
    #         f"but only found {n_total}."
    #     )

    # print(f"[INFO] Config {config_name}: {n_total} jets, building {n_agg} groups of {group_size}")

    # # Random permutation of all jet indices
    # rng = np.random.RandomState(rng_seed)
    # perm = rng.permutation(n_total)
    # used_indices = perm[:required]
    # groups = used_indices.reshape(n_agg, group_size)

    n_total = len(df)

    # Need at least one full group
    if n_total < group_size:
        raise ValueError(
            f"{config_name}: need at least {group_size} jets to build one aggregate, "
            f"but only found {n_total}."
        )

    rng = np.random.RandomState(rng_seed)

    # First phase: use each jet at most once (disjoint groups)
    max_disjoint_groups = n_total // group_size          # how many non-overlapping groups we can form
    n_disjoint = min(max_disjoint_groups, n_agg)         # groups without reuse
    remaining_groups = n_agg - n_disjoint                # groups we still need (with reuse)

    print(
        f"[INFO] Config {config_name}: {n_total} jets, "
        f"building {n_disjoint} groups without reuse"
        + (f" and {remaining_groups} groups with reuse" if remaining_groups > 0 else "")
        + f", group_size={group_size}"
    )

    # Random permutation of all jet indices for the disjoint phase
    perm = rng.permutation(n_total)

    groups = []

    # --- Disjoint phase: each jet used at most once ---
    for i in range(n_disjoint):
        start = i * group_size
        end = start + group_size
        groups.append(perm[start:end])

    # --- Reuse phase: sample additional groups from all jets ---
    for _ in range(remaining_groups):
        # sample group_size *distinct* jets within this group,
        # but allow jets to be reused across different groups
        grp_idx = rng.choice(n_total, size=group_size, replace=False)
        groups.append(grp_idx)

    groups = np.array(groups)


    # Original config dir (where source jets live)
    config_input_dir = input_root / f"{config_name}_unfolded"
    if not config_input_dir.exists():
        raise FileNotFoundError(f"Config directory (input) not found: {config_input_dir}")

    # Aggregated config dir (where aggregated jets are saved)
    agg_config_dir = agg_root / config_name
    agg_config_dir.mkdir(parents=True, exist_ok=True)

    # Aggregated metadata dir
    agg_metadata_dir = agg_root / "metadata"
    agg_metadata_dir.mkdir(parents=True, exist_ok=True)

    agg_rows = []

    # label columns: keep interesting label info, drop technical columns
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

        imgs = []
        src_paths = []

        for _, row in df_group.iterrows():
            rel_path = row["file_paths"]
            img_path = input_root / rel_path
            if not img_path.exists():
                # Try previous event in the same directory: event_XXXXXXX -> event_(XXXXXXX-1)
                p = Path(rel_path)
                stem = p.stem  # e.g. "event_0000012"
                base, num_str = stem.rsplit("_", 1)

                try:
                    prev_idx = int(num_str) - 1
                except ValueError:
                    raise FileNotFoundError(f"Missing jet .npy file and cannot parse index: {img_path}")

                if prev_idx < 0:
                    raise FileNotFoundError(f"Missing jet .npy file and no previous index: {img_path}")

                prev_stem = f"{base}_{prev_idx:07d}"
                prev_rel_path = str(p.with_name(prev_stem + p.suffix))
                prev_img_path = input_root / prev_rel_path

                if not prev_img_path.exists():
                    raise FileNotFoundError(
                        f"Missing jet .npy file: {img_path} and fallback {prev_img_path} also missing"
                    )

                print(f"[WARN] {config_name}: missing {rel_path}, using {prev_rel_path} instead")
                rel_path = prev_rel_path
                img_path = prev_img_path

            arr = np.load(img_path).astype(np.float32)
            imgs.append(arr)
            src_paths.append(rel_path)

        imgs = np.stack(imgs, axis=0)  # (group_size, H, W)
        mean_img = imgs.mean(axis=0)   # (H, W)

        # Normalize by MAX_PT
        mean_norm = mean_img / float(max_pt)

        # Log-normalize with img_max=1.0
        mean_log = log_normalize_image(mean_norm, epsilon=epsilon, img_max=1.0)

        # Save aggregated image as agg500_event_XXXXXXX.npy
        agg_fname = f"agg500_event_{agg_idx:07d}.npy"
        agg_path = agg_config_dir / agg_fname
        np.save(agg_path, mean_log)

        # Path relative to agg_root
        agg_rel = agg_path.relative_to(agg_root)

        # Labels from first jet in group (config-constant in your setup)
        label_values = {}
        if label_cols:
            first_row = df_group.iloc[0]
            label_values = {col: first_row[col] for col in label_cols}

        agg_rows.append(
            {
                "config": config_name,
                "agg_file_path": str(agg_rel),
                # original 500 image paths (relative to input_root), joined
                "source_file_paths": "|".join(src_paths),
                **label_values,
            }
        )

        if (agg_idx + 1) % 25 == 0 or agg_idx == n_agg - 1:
            print(f"[INFO]   {config_name}: built {agg_idx + 1}/{n_agg} aggregated jets")

    # Write per-config agg500 metadata CSV
    out_csv = agg_metadata_dir / f"{config_name}_agg500_metadata.csv"
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(out_csv, index=False)

    print(f"[INFO]   {config_name}: wrote agg500 metadata to {out_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Build 500-jet aggregated dataset for a single config."
    )
    parser.add_argument(
        "--input-root",
        type=str,
        required=True,
        help="Root directory of the unfolded dataset (contains config dirs + metadata/).",
    )
    parser.add_argument(
        "--agg-root",
        type=str,
        required=True,
        help="Root directory where aggregated 500-jet dataset will be stored.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config name, e.g. c0005_alpha0p20_q_1p0_unfolded.",
    )
    parser.add_argument(
        "--n-agg",
        type=int,
        default=400,
        help="Number of 500-jet aggregates to build (default: 400).",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=500,
        help="Number of jets per aggregate (default: 500).",
    )
    parser.add_argument(
        "--max-pt",
        type=float,
        default=110.0,
        help="Fixed MAX_PT for normalization (default: 110.0).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
        help="Epsilon for log normalization (default: 1e-8).",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=0,
        help="Random seed for shuffling (default: 0).",
    )

    args = parser.parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    agg_root = Path(args.agg_root).expanduser().resolve()
    config_name = args.config

    if not (input_root / "metadata").exists():
        raise FileNotFoundError(f"Metadata directory not found in input-root: {input_root / 'metadata'}")

    agg_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] input_root: {input_root}")
    print(f"[INFO] agg_root:   {agg_root}")
    print(f"[INFO] config:     {config_name}")

    build_agg500_for_config(
        config_name=config_name,
        input_root=input_root,
        agg_root=agg_root,
        n_agg=args.n_agg,
        group_size=args.group_size,
        max_pt=args.max_pt,
        epsilon=args.epsilon,
        rng_seed=args.rng_seed,
    )


if __name__ == "__main__":
    main()
