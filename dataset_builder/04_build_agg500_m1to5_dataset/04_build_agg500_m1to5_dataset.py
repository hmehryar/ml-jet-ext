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
    tmp_path = Path(str(path) + ".tmp")
    np.save(tmp_path, arr)
    tmp_path.replace(path)

def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    args = parse_args()

    base_root = Path(args.base_root)
    unfolded_root = base_root / args.unfolded_dir_name
    if not unfolded_root.exists():
        raise FileNotFoundError(f"Unfolded dataset root not found: {unfolded_root}")
    
    old_agg_root = base_root / args.old_agg_dir_name
    if not old_agg_root.exists():
        raise FileNotFoundError(f"Old agg500 dataset root not found: {old_agg_root}")
    
    new_agg_root = base_root / args.new_agg_dir_name
    if not new_agg_root.exists():
        print(f"New agg500 dataset root does not exist, will create: {new_agg_root}")

    new_meta_dir = new_agg_root / "metadata"
    if not new_meta_dir.exists():
        print(f"New metadata directory does not exist, will create: {new_meta_dir}")

    new_agg_root.mkdir(parents=True, exist_ok=True)
    new_meta_dir.mkdir(parents=True, exist_ok=True)

    csv_name = f"{args.run_dir}_agg500_metadata.csv"
    cfg_csv = new_meta_dir / csv_name

    if not cfg_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {cfg_csv}")

    print(f"[INFO] Reading config metadata: {cfg_csv}")
    df = pd.read_csv(cfg_csv)
    colmap = infer_metadata_columns(df)

    print("[INFO] Column mapping:")
    for k, v in colmap.items():
        print(f"  {k}: {v}")

    out_rows = []
    n_rows = len(df)

    for idx, row in tqdm(df.iterrows(), total=n_rows, desc=args.run_dir):
        
        old_rel = str(row[colmap["agg_col"]]).strip()
        new_rel = old_rel  # preserve same relative layout as agg500 mean dataset
        new_abs = new_agg_root / new_rel
        new_abs.parent.mkdir(parents=True, exist_ok=True)

        if (not args.overwrite) and is_valid_moment_file(new_abs):
            out_row = row.to_dict()
            out_row[colmap["agg_col"]] = new_rel
            out_rows.append(out_row)
            continue

        src_paths = parse_src_paths(row[colmap["file_col"]])
        if len(src_paths) != 500:
            raise ValueError(
                f"Row {idx}: expected 500 source paths, got {len(src_paths)}"
            )

        # stack = np.stack(
        #     [load_single_image(unfolded_root, p) for p in src_paths],
        #     axis=0
        # )

        # moments = compute_moments_1_to_5(stack)

        imgs = [load_single_image(unfolded_root, p) for p in src_paths]   # still loaded from disk via numpy
        stack_np = np.stack(imgs, axis=0).astype(np.float32)              # [500, 32, 32]

        stack = torch.from_numpy(stack_np).to(device)                     # move to GPU
        moments = compute_moments_1_to_5_torch(stack)  
        if args.normalize_moments:
            moments = normalize_moments_1_to_5_torch(moments)                   # compute on GPU
        moments_np = moments.detach().cpu().numpy().astype(np.float32)                # move back to CPU for saving
        # atomic_save_npy(new_abs, moments_np)
        np.save(new_abs, moments_np)

        out_row = row.to_dict()
        out_row[colmap["agg_col"]] = new_rel
        out_rows.append(out_row)

        if args.check_mean_consistency and (not args.normalize_moments):
            old_mean_path = old_agg_root / old_rel
            if old_mean_path.exists():
                old_mean = np.load(old_mean_path).astype(np.float32)
                max_abs_diff = np.max(np.abs(old_mean - moments[0]))
                if not np.allclose(old_mean, moments[0], atol=1e-6):
                    raise ValueError(
                        f"Mean mismatch for {old_rel}; max abs diff={max_abs_diff}"
                    )

        out_row = row.to_dict()
        out_row[colmap["agg_col"]] = new_rel
        out_rows.append(out_row)

    out_df = pd.DataFrame(out_rows)
    out_csv = new_meta_dir / csv_name
    out_df.to_csv(out_csv, index=False)

    print(f"[INFO] Wrote updated metadata: {out_csv}")
    print(f"[INFO] Done building 5-moment aggregates for {args.run_dir}")
    print(f"[INFO] Total rows processed: {len(out_df)}")


if __name__ == "__main__":
    main()