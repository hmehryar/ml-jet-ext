
#!/usr/bin/env python3
"""Smoke test for the real ML-JET-Ext moment dataset."""

import argparse
from pathlib import Path

import torch

from data.moment_loader import get_moment_dataloaders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    root = Path(args.dataset_root).expanduser().resolve()

    for moments in ([1], [1, 2], [1, 2, 3, 4, 5]):
        print(f"\nTesting moments: {moments}", flush=True)

        train_loader, val_loader, test_loader = (
            get_moment_dataloaders(
                dataset_root=root,
                moments=moments,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                validate_files=False,
                check_normalization=True,
            )
        )

        expected_sizes = (11200, 1400, 1400)

        for loader, expected in zip(
            (train_loader, val_loader, test_loader),
            expected_sizes,
        ):
            assert len(loader.dataset) == expected

            images, labels = next(iter(loader))

            assert images.shape == (
                min(args.batch_size, expected),
                len(moments),
                32,
                32,
            )

            assert images.dtype == torch.float32
            assert torch.isfinite(images).all()

            alpha = labels["alpha_output"]
            q0 = labels["q0_output"]

            assert alpha.dtype == torch.long
            assert q0.dtype == torch.long

            assert ((alpha >= 0) & (alpha < 7)).all()
            assert ((q0 >= 0) & (q0 < 5)).all()

            print(
                f"  PASS: {len(loader.dataset)} samples, "
                f"batch={tuple(images.shape)}",
                flush=True,
            )

    print("\nPASS: Real-dataset DataLoader smoke test")


if __name__ == "__main__":
    main()
