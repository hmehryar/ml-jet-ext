#!/usr/bin/env python3
"""
Validate signed-log normalization for M3 and M5.

Tests that positive, negative, and zero moment values
retain their signs after normalization.
"""

import importlib.util
from pathlib import Path

import torch


# Locate the existing dataset builder.
BUILDER_PATH = (
    Path(__file__).resolve().parent
    / "04_build_agg500_m1to5_dataset"
    / "04_build_agg500_m1to5_dataset.py"
)

if not BUILDER_PATH.is_file():
    raise FileNotFoundError(
        f"Cannot find dataset builder: {BUILDER_PATH}"
    )

# Import the builder without executing its main() function.
spec = importlib.util.spec_from_file_location(
    "moment_builder", BUILDER_PATH
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

normalize_moments = module.normalize_moments_1_to_5_torch


def test_sign_preservation():
    moments = torch.zeros((5, 32, 32), dtype=torch.float32)

    # M3: third central moment
    moments[2, 0, 0] = -100.0
    moments[2, 0, 1] = 100.0

    # M5: fifth central moment
    moments[4, 0, 0] = -1000.0
    moments[4, 0, 1] = 1000.0

    normalized = normalize_moments(moments)

    # Negative values must stay negative.
    assert normalized[2, 0, 0].item() < 0
    assert normalized[4, 0, 0].item() < 0

    # Positive values must stay positive.
    assert normalized[2, 0, 1].item() > 0
    assert normalized[4, 0, 1].item() > 0

    # Zeros must stay zero.
    assert normalized[2, 1, 1].item() == 0
    assert normalized[4, 1, 1].item() == 0

    # All normalized values must be finite.
    assert torch.isfinite(normalized).all()

    # Expected normalization ranges.
    assert normalized[2].min().item() >= -1
    assert normalized[2].max().item() <= 1
    assert normalized[4].min().item() >= -1
    assert normalized[4].max().item() <= 1

    print("\nM3 signed normalization:")
    print("  Negative:", normalized[2, 0, 0].item())
    print("  Positive:", normalized[2, 0, 1].item())

    print("\nM5 signed normalization:")
    print("  Negative:", normalized[4, 0, 0].item())
    print("  Positive:", normalized[4, 0, 1].item())

    print("\nPASS: M3 and M5 preserve positive, negative, and zero signs.")


if __name__ == "__main__":
    test_sign_preservation()