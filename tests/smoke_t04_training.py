
"""T04 integration smoke test with the existing moment training utilities."""

import argparse

import torch
from torch.utils.data import DataLoader, Dataset

from models.model_clip import create_model as create_clip
from models.model_siglip import create_model as create_siglip
from train_utils.moment_training import (
    train_moment_epoch,
    evaluate_moment,
)
from train_utils.ordinal_loss import build_moment_criterion


class SyntheticMomentDataset(Dataset):
    def __init__(self, samples=2, channels=5):
        generator = torch.Generator().manual_seed(42)
        self.images = torch.rand(
            samples, channels, 32, 32, generator=generator
        )
        # M3 and M5 can have signed values.
        for index in (2, 4):
            if index < channels:
                self.images[:, index] = (
                    self.images[:, index] * 2 - 1
                )
        self.alpha = torch.randint(
            0, 7, (samples,), generator=generator
        )
        self.q0 = torch.randint(
            0, 5, (samples,), generator=generator
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], {
            "alpha_output": self.alpha[index],
            "q0_output": self.q0[index],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["clip", "siglip"], required=True
    )
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA GPU required"
    device = torch.device("cuda")
    torch.manual_seed(42)

    factories = {
        "clip": (create_clip, "clip_vit_b16"),
        "siglip": (create_siglip, "siglip_vit_b16"),
    }
    factory, backbone = factories[args.model]

    model, optimizer = factory(
        backbone=backbone,
        input_shape=(5, 32, 32),
        dataset_type="moment",
        freeze_backbone=True,
        pretrained=True,
        learning_rate=1e-4,
    )
    model.to(device)

    loader = DataLoader(
        SyntheticMomentDataset(), batch_size=2, shuffle=False
    )
    criterion = build_moment_criterion(
        {"type": "cross_entropy"}, device=device
    )
    weights = {
        "alpha_output": 1.0,
        "q0_output": 1.0,
    }

    torch.cuda.reset_peak_memory_stats()

    print(f"\nTesting {args.model.upper()}...")

    train_metrics = train_moment_epoch(
        loader, model, criterion, optimizer, device, weights
    )
    val_metrics = evaluate_moment(
        loader, model, criterion, device, weights
    )

    assert torch.isfinite(
        torch.tensor(train_metrics["loss"])
    )
    assert torch.isfinite(
        torch.tensor(val_metrics["loss"])
    )
    assert 0 <= val_metrics["accuracy"] <= 1

    torch.cuda.synchronize()

    print("\nTraining loss:", train_metrics["loss"])
    print("Validation loss:", val_metrics["loss"])
    print("Alpha accuracy:", val_metrics["accuracy_alpha"])
    print("Q0 accuracy:", val_metrics["accuracy_q0"])
    print("Joint accuracy:", val_metrics["accuracy"])
    print(
        "Peak GPU memory (GiB):",
        round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    )
    print(f"\n{args.model.upper()} T04 training integration: PASS")


if __name__ == "__main__":
    main()
