
#!/usr/bin/env python3
"""
Parallel validation of normalized M1-M5 jet aggregates.
"""

import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm


EXPECTED_SHAPE = (5, 32, 32)
LOWER = np.array([0, 0, -1, 0, -1])
UPPER = np.array([1, 1, 1, 1, 1])


def check_file(path):
    try:
        arr = np.load(path, allow_pickle=False)

        if arr.shape != EXPECTED_SHAPE:
            return ("shape", str(path), None, None)

        if arr.dtype != np.float32:
            return ("dtype", str(path), None, None)

        if not np.isfinite(arr).all():
            return ("nonfinite", str(path), None, None)

        channel_min = arr.min(axis=(1, 2))
        channel_max = arr.max(axis=(1, 2))

        if (
            np.any(channel_min < LOWER - 1e-6)
            or np.any(channel_max > UPPER + 1e-6)
        ):
            return (
                "range",
                str(path),
                channel_min,
                channel_max,
            )

        return ("ok", str(path), channel_min, channel_max)

    except Exception as exc:
        return ("unreadable", f"{path}: {exc}", None, None)


def validate(root, workers):
    files = sorted(root.glob("c0005_alpha*_q_*/*.npy"))

    print(f"Found {len(files)} NPY files")
    print(f"Using {workers} CPU workers")

    global_min = np.full(5, np.inf)
    global_max = np.full(5, -np.inf)
    errors = {}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = executor.map(check_file, files)

        for status, path, mins, maxs in tqdm(
            results,
            total=len(files),
            desc="Validating M1-M5",
        ):
            if status != "ok":
                errors.setdefault(status, []).append(path)

            if mins is not None:
                global_min = np.minimum(global_min, mins)
                global_max = np.maximum(global_max, maxs)

    print("\n========== VALIDATION SUMMARY ==========")
    print(f"Total files: {len(files)}")

    for status in [
        "shape", "dtype", "nonfinite", "range", "unreadable"
    ]:
        paths = errors.get(status, [])
        print(f"{status:<12}: {len(paths)}")

        for path in paths[:3]:
            print(f"  Example: {path}")

    print("\nGlobal moment ranges:")
    for i in range(5):
        print(
            f"M{i+1}: "
            f"min={global_min[i]:.8f}, "
            f"max={global_max[i]:.8f}"
        )

    passed = len(files) == 14000 and not errors

    print("\nRESULT:", "PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, required=True
    )
    parser.add_argument(
        "--workers", type=int, default=8
    )

    args = parser.parse_args()

    if not validate(args.root, args.workers):
        raise SystemExit(1)
