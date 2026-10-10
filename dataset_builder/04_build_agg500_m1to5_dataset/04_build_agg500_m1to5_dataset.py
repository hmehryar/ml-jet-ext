#!/usr/bin/env python3
import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build 5-moment agg500 dataset for one configuration."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Configuration directory name, e.g. c0005_alpha0p20_q_1p0"
    )
    parser.add_argument(
        "--base_root",
        type=str,
        default="/rs/rs_grp_mljet/ml-jet-ext",
        help="Base root containing unfolded and aggregated dataset folders."
    )
    parser.add_argument(
        "--unfolded_dir_name",
        type=str,
        default="pTHat50-250_eCM5020_c0005_pT90-110_R0p4",
        help="Root folder of unfolded single-jet images."
    )
    parser.add_argument(
        "--old_agg_dir_name",
        type=str,
        default="pTHat50-250_eCM5020_c0005_pT90-110_R0p4_agg500",
        help="Root folder of existing agg500 mean dataset."
    )
    parser.add_argument(
        "--new_agg_dir_name",
        type=str,
        default="pTHat50-250_eCM5020_c0005_pT90-110_R0p4_agg500_m1-5",
        help="Root folder of new 5-moment agg500 dataset."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files."
    )
    parser.add_argument(
        "--check_mean_consistency",
        action="store_true",
        help="Check recomputed moment-1 against existing agg500 mean files."
    )
    parser.add_argument(
        "--normalize_moments",
        action="store_true",
        help="Apply fixed per-moment log/signed-log normalization before saving."
    )
    return parser.parse_args()


def parse_src_paths(value):
    if isinstance(value, list):
        return value

    if pd.isna(value):
        return []

    s = str(value).strip()

    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except Exception:
                pass

    if "|" in s:
        return [x.strip() for x in s.split("|") if x.strip()]

    return [s]


def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns found: {candidates}")


def infer_metadata_columns(df):
    file_col = find_column(
        df,
        ["source_file_paths", "file_paths", "src_file_paths", "member_file_paths"]
    )
    agg_col = find_column(
        df,
        ["agg_file_path", "file_path", "agg_path"]
    )

    alpha_col = find_column(df, ["alpha", "alpha_s", "alpha_value"])
    q0_col = find_column(df, ["q0", "Q0", "q_0"])
    # eloss_col = find_column(df, ["energy_loss", "eloss", "energy_loss_module"])

    config_col = None
    for c in ["config", "config_name"]:
        if c in df.columns:
            config_col = c
            break

    return {
        "file_col": file_col,
        "agg_col": agg_col,
        "alpha_col": alpha_col,
        "q0_col": q0_col,
        # "eloss_col": eloss_col,
        "config_col": config_col,
    }


def compute_moments_1_to_5(stack):
    stack = np.asarray(stack, dtype=np.float32)
    if stack.ndim != 3:
        raise ValueError(f"Expected stack shape [N,H,W], got {stack.shape}")

    m1 = stack.mean(axis=0)
    centered = stack - m1

    m2 = (centered ** 2).mean(axis=0)
    m3 = (centered ** 3).mean(axis=0)
    m4 = (centered ** 4).mean(axis=0)
    m5 = (centered ** 5).mean(axis=0)

    return np.stack([m1, m2, m3, m4, m5], axis=0).astype(np.float32)

import math

def log_normalize_image(img, epsilon=1e-8, img_max=1.0):
    """
    Log normalize nonnegative image to [0, 1].
    """
    I = np.asarray(img, dtype=np.float32)

    if img_max <= 0:
        return np.zeros_like(I, dtype=np.float32)

    num = np.log(I + epsilon) - math.log(epsilon)
    den = math.log(img_max + epsilon) - math.log(epsilon)

    if den <= 0:
        return np.zeros_like(I, dtype=np.float32)

    I_log = num / den
    return np.clip(I_log, 0.0, 1.0).astype(np.float32)


def signed_log_normalize_image(img, epsilon=1e-8, img_abs_max=1.0):
    """
    Signed log normalize image to [-1, 1].
    Suitable for odd central moments that may be negative.
    """
    I = np.asarray(img, dtype=np.float32)

    if img_abs_max <= 0:
        return np.zeros_like(I, dtype=np.float32)

    mag = np.abs(I)
    num = np.log(mag + epsilon) - math.log(epsilon)
    den = math.log(img_abs_max + epsilon) - math.log(epsilon)

    if den <= 0:
        return np.zeros_like(I, dtype=np.float32)

    I_log = num / den
    I_signed = np.sign(I) * np.clip(I_log, 0.0, 1.0)
    return I_signed.astype(np.float32)


def normalize_moments_1_to_5(moments, epsilon=1e-8):
    """
    moments: [5, H, W]
    Returns normalized moments:
      m1, m2, m4 -> [0, 1]
      m3, m5     -> [-1, 1]
    """
    moments = np.asarray(moments, dtype=np.float32)
    assert moments.shape[0] == 5, f"Expected 5 channels, got {moments.shape}"

    m1_max = 110.0
    m2_max = 110.0 ** 2
    m3_abs_max = 110.0 ** 3
    m4_max = 110.0 ** 4
    m5_abs_max = 110.0 ** 5

    out = np.stack([
        log_normalize_image(moments[0], epsilon=epsilon, img_max=m1_max),
        log_normalize_image(moments[1], epsilon=epsilon, img_max=m2_max),
        signed_log_normalize_image(moments[2], epsilon=epsilon, img_abs_max=m3_abs_max),
        log_normalize_image(moments[3], epsilon=epsilon, img_max=m4_max),
        signed_log_normalize_image(moments[4], epsilon=epsilon, img_abs_max=m5_abs_max),
    ], axis=0).astype(np.float32)

    return out

import torch


def log_normalize_image_torch(img: torch.Tensor, epsilon: float = 1e-8, img_max: float = 1.0):
    """
    Log normalize nonnegative image to [0, 1].
    img: [H, W]
    """
    if img_max <= 0:
        return torch.zeros_like(img)

    eps = torch.tensor(epsilon, dtype=img.dtype, device=img.device)
    img_max_t = torch.tensor(img_max, dtype=img.dtype, device=img.device)

    num = torch.log(img + eps) - torch.log(eps)
    den = torch.log(img_max_t + eps) - torch.log(eps)

    if den.item() <= 0:
        return torch.zeros_like(img)

    out = num / den
    return torch.clamp(out, 0.0, 1.0)


def signed_log_normalize_image_torch(img: torch.Tensor, epsilon: float = 1e-8, img_abs_max: float = 1.0):
    """
    Signed log normalize image to [-1, 1].
    Suitable for odd central moments that may be negative.
    img: [H, W]
    """
    if img_abs_max <= 0:
        return torch.zeros_like(img)

    eps = torch.tensor(epsilon, dtype=img.dtype, device=img.device)
    img_abs_max_t = torch.tensor(img_abs_max, dtype=img.dtype, device=img.device)

    mag = torch.abs(img)
    num = torch.log(mag + eps) - torch.log(eps)
    den = torch.log(img_abs_max_t + eps) - torch.log(eps)

    if den.item() <= 0:
        return torch.zeros_like(img)

    out = num / den
    out = torch.clamp(out, 0.0, 1.0)
    return torch.sign(img) * out

def normalize_moments_1_to_5_torch(moments: torch.Tensor, epsilon: float = 1e-8):
    """
    moments: [5, H, W] torch tensor on CPU or GPU

    Returns:
      m1, m2, m4 -> [0, 1]
      m3, m5     -> [-1, 1]
    """
    if moments.ndim != 3 or moments.shape[0] != 5:
        raise ValueError(f"Expected [5,H,W], got {tuple(moments.shape)}")

    moments = moments.to(torch.float32)

    m1_max = 110.0
    m2_max = 110.0 ** 2
    m3_abs_max = 110.0 ** 3
    m4_max = 110.0 ** 4
    m5_abs_max = 110.0 ** 5

    out = torch.stack([
        log_normalize_image_torch(moments[0], epsilon=epsilon, img_max=m1_max),
        log_normalize_image_torch(moments[1], epsilon=epsilon, img_max=m2_max),
        signed_log_normalize_image_torch(moments[2], epsilon=epsilon, img_abs_max=m3_abs_max),
        log_normalize_image_torch(moments[3], epsilon=epsilon, img_max=m4_max),
        signed_log_normalize_image_torch(moments[4], epsilon=epsilon, img_abs_max=m5_abs_max),
    ], dim=0)

    return out

def load_single_image(unfolded_root: Path, rel_path: str) -> np.ndarray:
    img_path = unfolded_root / rel_path
    if not img_path.exists():
        raise FileNotFoundError(f"Missing source image: {img_path}")
    return np.load(img_path).astype(np.float32)

def compute_moments_1_to_5_torch(stack: torch.Tensor) -> torch.Tensor:
    """
    stack: [N, H, W] on CPU or GPU
    returns: [5, H, W]
    """
    if stack.ndim != 3:
        raise ValueError(f"Expected [N,H,W], got {tuple(stack.shape)}")

    m1 = stack.mean(dim=0)
    centered = stack - m1

    m2 = (centered ** 2).mean(dim=0)
    m3 = (centered ** 3).mean(dim=0)
    m4 = (centered ** 4).mean(dim=0)
    m5 = (centered ** 5).mean(dim=0)

    return torch.stack([m1, m2, m3, m4, m5], dim=0)

def is_valid_moment_file(path, expected_shape=(5, 32, 32)):
    if not path.exists():
        return False
    try:
        arr = np.load(path, mmap_mode="r")
        return arr.shape == expected_shape
    except Exception:
        return False
    
def atomic_save_npy(path: Path, arr: np.ndarray):
    """Atomically replace a completed NPY file (np.save adds .npy to strings)."""
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + '.', suffix='.npy', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            np.save(f, arr)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def prepare_metadata(df, colmap, run_dir):
    """Validate and deduplicate source aggregate membership, preserving row order."""
    agg_col = colmap["agg_col"]
    file_col = colmap["file_col"]
    config_col = colmap["config_col"]
    alpha_col = colmap["alpha_col"]
    q0_col = colmap["q0_col"]

    required = [agg_col, file_col, alpha_col, q0_col]
    if df[required].isna().any().any():
        raise ValueError('Input metadata has missing paths or physics labels')
    if config_col is not None and (df[config_col].astype(str) != run_dir).any():
        raise ValueError(f'Input metadata contains a different configuration than {run_dir}')

    retained = []
    seen_paths = {}
    seen_groups = {}
    for idx, row in df.iterrows():
        path = str(row[agg_col]).strip()
        relative = Path(path)
        if not path or relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Row {idx}: unsafe aggregate path: {path}')
        if relative.parts[0] != run_dir:
            raise ValueError(f'Row {idx}: aggregate path does not match {run_dir}: {path}')

        members = parse_src_paths(row[file_col])
        if len(members) != 500 or len(set(members)) != 500:
            raise ValueError(f'Row {idx}: expected 500 distinct source files, got {len(members)}')
        signature = tuple(sorted(members))
        labels = (float(row[alpha_col]), float(row[q0_col]))

        if path in seen_paths:
            if seen_paths[path] != (signature, labels):
                raise ValueError(f'Conflicting metadata for aggregate path: {path}')
            continue
        if signature in seen_groups:
            raise ValueError(f'Same 500-source group mapped to two output paths: {seen_groups[signature]} and {path}')
        seen_paths[path] = (signature, labels)
        seen_groups[signature] = path
        retained.append(idx)

    result = df.loc[retained].copy().reset_index(drop=True)
    print(f'[INFO] Source metadata: {len(df)} rows -> {len(result)} unique aggregates')
    return result


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] Using device: {device}')

    base_root = Path(args.base_root)
    unfolded_root = base_root / args.unfolded_dir_name
    old_agg_root = base_root / args.old_agg_dir_name
    new_agg_root = base_root / args.new_agg_dir_name
    if not unfolded_root.is_dir():
        raise FileNotFoundError(f'Unfolded dataset root not found: {unfolded_root}')
    if not old_agg_root.is_dir():
        raise FileNotFoundError(f'Old agg500 dataset root not found: {old_agg_root}')
    if old_agg_root.resolve() == new_agg_root.resolve():
        raise ValueError('Old and new aggregate roots must differ')

    csv_name = f'{args.run_dir}_agg500_metadata.csv'
    # Always load the source aggregation manifest, never the output manifest.
    input_csv = old_agg_root / 'metadata' / csv_name
    output_csv = new_agg_root / 'metadata' / csv_name
    if not input_csv.is_file():
        raise FileNotFoundError(f'Source metadata CSV not found: {input_csv}')
    print(f'[INFO] Reading SOURCE metadata: {input_csv}')
    df = pd.read_csv(input_csv)
    colmap = infer_metadata_columns(df)
    df = prepare_metadata(df, colmap, args.run_dir)

    # A destination may already contain normalized tensors. Reruns must use
    # the same --normalize_moments setting that originally generated them.
    if (not args.overwrite) and output_csv.exists():
        previous = pd.read_csv(output_csv)
        if colmap['agg_col'] in previous.columns:
            existing = previous.drop_duplicates(colmap['agg_col'])
            if len(existing) and not set(existing[colmap['agg_col']]).issubset(set(df[colmap['agg_col']])):
                raise ValueError('Output metadata contains paths absent from source metadata')

    new_agg_root.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_rows = []
    computed = skipped = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=args.run_dir):
        relative = str(row[colmap['agg_col']]).strip()
        destination = new_agg_root / relative
        if not args.overwrite and is_valid_moment_file(destination):
            skipped += 1
        else:
            sources = parse_src_paths(row[colmap['file_col']])
            stack_np = np.stack(
                [load_single_image(unfolded_root, p) for p in sources], axis=0
            ).astype(np.float32)
            if stack_np.shape != (500, 32, 32):
                raise ValueError(f'Row {idx}: unexpected source stack shape {stack_np.shape}')

            stack = torch.from_numpy(stack_np).to(device)
            with torch.no_grad():
                moments = compute_moments_1_to_5_torch(stack)
                if args.normalize_moments:
                    moments = normalize_moments_1_to_5_torch(moments)
                values = moments.cpu().numpy().astype(np.float32)

            if not np.isfinite(values).all():
                raise ValueError(f'Row {idx}: nonfinite moments for {relative}')

            if args.check_mean_consistency and not args.normalize_moments:
                old_mean_path = old_agg_root / relative
                if not old_mean_path.is_file():
                    raise FileNotFoundError(f'Missing old mean: {old_mean_path}')
                old_mean = np.load(old_mean_path, allow_pickle=False).astype(np.float32)
                if not np.allclose(old_mean, values[0], atol=1e-6):
                    error = np.max(np.abs(old_mean - values[0]))
                    raise ValueError(f'Mean mismatch for {relative}: max error {error}')

            atomic_save_npy(destination, values)
            computed += 1

        # Exactly one metadata row per unique source aggregate, including skips.
        output_rows.append(row.to_dict())

    out_df = pd.DataFrame(output_rows, columns=df.columns)
    if out_df[colmap['agg_col']].duplicated().any():
        raise AssertionError('Internal error: duplicate output aggregate paths')
    # Atomic CSV write; a failed run leaves the previous metadata unchanged.
    import os
    import tempfile
    fd, temp_name = tempfile.mkstemp(prefix=csv_name + '.', suffix='.tmp', dir=output_csv.parent)
    try:
        with os.fdopen(fd, 'w', newline='') as f:
            out_df.to_csv(f, index=False)
        os.replace(temp_name, output_csv)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)

    print(f'[INFO] Saved metadata: {output_csv}')
    print(f'[INFO] Unique rows: {len(out_df)}; computed: {computed}; reused: {skipped}')


if __name__ == '__main__':
    main()
