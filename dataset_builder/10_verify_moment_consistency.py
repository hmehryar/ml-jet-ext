
#!/usr/bin/env python3
"""
T01 - Verify saved M1-M5 aggregates against their 500 source jets.

- Reads cleaned metadata CSVs.
- Recomputes central moments using the original dataset builder.
- Applies the original normalization.
- Compares saved and recomputed tensors.
- Reports raw M3/M5 sign statistics.
- Never modifies dataset files.
"""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
BUILDER_PATH = (
    SCRIPT_DIR
    / "04_build_agg500_m1to5_dataset"
    / "04_build_agg500_m1to5_dataset.py"
)

spec = importlib.util.spec_from_file_location(
    "moment_builder", BUILDER_PATH
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def validate_sample(
    row,
    dataset_root,
    unfolded_root,
    device,
    atol,
    rtol,
):
    # Read exact 500-jet membership from metadata.
    source_paths = builder.parse_src_paths(
        row["source_file_paths"]
    )

    if len(source_paths) != 500:
        raise ValueError(
            f"Expected 500 jets, got {len(source_paths)}"
        )

    if len(set(source_paths)) != 500:
        raise ValueError("Repeated source jets within aggregate")

    # Load the original jet images.
    images = []
    for rel_path in source_paths:
        image = builder.load_single_image(
            unfolded_root, rel_path
        )
        if image.shape != (32, 32):
            raise ValueError(
                f"Invalid source shape: {rel_path}: {image.shape}"
            )
        images.append(image)

    stack_np = np.stack(images, axis=0).astype(np.float32)
    stack = torch.from_numpy(stack_np).to(device)

    # Reproduce builder's numerical operations.
    with torch.no_grad():
        raw_moments = builder.compute_moments_1_to_5_torch(
            stack
        )

        normalized = builder.normalize_moments_1_to_5_torch(
            raw_moments
        )

        recomputed = normalized.cpu().numpy()

        # Inspect signs before normalization.
        raw_m3 = raw_moments[2]
        raw_m5 = raw_moments[4]

        raw_m3_negative = int((raw_m3 < 0).sum().item())
        raw_m5_negative = int((raw_m5 < 0).sum().item())

        raw_m3_min = float(raw_m3.min().item())
        raw_m5_min = float(raw_m5.min().item())

    # Read existing normalized M1-M5 tensor.
    saved_path = dataset_root / str(row["agg_file_path"])
    saved = np.load(saved_path, allow_pickle=False)

    if saved.shape != (5, 32, 32):
        raise ValueError(
            f"Invalid saved tensor shape: {saved.shape}"
        )

    if not np.isfinite(recomputed).all():
        raise ValueError("Recomputed tensor contains NaN/Inf")

    # Compare each moment independently.
    difference = np.abs(
        recomputed.astype(np.float64)
        - saved.astype(np.float64)
    )

    max_errors = difference.max(axis=(1, 2))
    mean_errors = difference.mean(axis=(1, 2))

    channel_pass = [
        bool(np.allclose(
            recomputed[i],
            saved[i],
            atol=atol,
            rtol=rtol,
        ))
        for i in range(5)
    ]

    return {
        "path": str(row["agg_file_path"]),
        "alpha": float(row["alpha"]),
        "q0": float(row["q0"]),
        "max_errors": max_errors.tolist(),
        "mean_errors": mean_errors.tolist(),
        "channel_pass": channel_pass,
        "passed": all(channel_pass),
        "raw_m3_negative": raw_m3_negative,
        "raw_m5_negative": raw_m5_negative,
        "raw_m3_min": raw_m3_min,
        "raw_m5_min": raw_m5_min,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root of normalized M1-M5 dataset",
    )
    parser.add_argument(
        "--unfolded-root",
        type=Path,
        required=True,
        help="Root containing original single-jet NPY images",
    )
    parser.add_argument(
        "--samples-per-config",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-4,
    )

    args = parser.parse_args()

    if args.samples_per_config < 1:
        parser.error("--samples-per-config must be >= 1")

    dataset_root = args.dataset_root.resolve()
    unfolded_root = args.unfolded_root.resolve()

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    metadata_dir = dataset_root / "metadata"
    csv_files = sorted(
        metadata_dir.glob("*_agg500_metadata.csv")
    )

    if len(csv_files) != 35:
        raise ValueError(
            f"Expected 35 metadata files, found {len(csv_files)}"
        )

    print(f"Device: {device}")
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("Metadata files:", len(csv_files))

    # Deterministic sampling within each configuration.
    rng = np.random.default_rng(args.seed)
    selected_rows = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        required = {
            "agg_file_path",
            "source_file_paths",
            "alpha",
            "q0",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"{csv_file.name}: missing {missing}"
            )

        if df["agg_file_path"].duplicated().any():
            raise ValueError(
                f"{csv_file.name}: duplicate aggregate paths"
            )

        if len(df) < args.samples_per_config:
            raise ValueError(
                f"Insufficient samples: {csv_file.name}"
            )

        indices = rng.choice(
            len(df),
            size=args.samples_per_config,
            replace=False,
        )

        for idx in indices:
            selected_rows.append(df.iloc[int(idx)])

    print("Selected aggregates:", len(selected_rows))
    print("Absolute tolerance:", args.atol)
    print("Relative tolerance:", args.rtol)

    results = []
    failures = []

    for row in tqdm(
        selected_rows,
        desc="Verifying moment consistency",
    ):
        try:
            result = validate_sample(
                row,
                dataset_root,
                unfolded_root,
                device,
                args.atol,
                args.rtol,
            )
            results.append(result)

            if not result["passed"]:
                failures.append(
                    (result["path"], "Numerical mismatch")
                )

        except Exception as exc:
            failures.append((
                str(row["agg_file_path"]),
                str(exc),
            ))

    print("\n========== MOMENT CONSISTENCY ==========")
    print("Selected aggregates:", len(selected_rows))
    print("Successfully compared:", len(results))
    print("Failures:", len(failures))

    if results:
        max_errors = np.array([
            result["max_errors"] for result in results
        ])

        mean_errors = np.array([
            result["mean_errors"] for result in results
        ])

        print("\nMoment    Max Abs Error    Mean Abs Error")

        for i in range(5):
            print(
                f"M{i+1:<8}"
                f"{max_errors[:, i].max():.10e}    "
                f"{mean_errors[:, i].mean():.10e}"
            )

        print("\nRaw odd-moment statistics:")
        print(
            "M3 negative pixels:",
            sum(r["raw_m3_negative"] for r in results),
        )
        print(
            "M5 negative pixels:",
            sum(r["raw_m5_negative"] for r in results),
        )
        print(
            "Minimum raw M3:",
            min(r["raw_m3_min"] for r in results),
        )
        print(
            "Minimum raw M5:",
            min(r["raw_m5_min"] for r in results),
        )

    if failures:
        print("\nFailure examples:")
        for path, message in failures[:10]:
            print(f"  {path}: {message}")

    passed = (
        len(results) == len(selected_rows)
        and not failures
    )

    print("\nRESULT:", "PASS" if passed else "FAIL")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
