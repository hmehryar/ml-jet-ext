
"""T04 CLIP/SigLIP moment-classification training runner."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from data.moment_loader import get_moment_dataloaders
from models.model_clip import create_model as create_clip
from models.model_siglip import create_model as create_siglip
from train_utils.ordinal_loss import build_moment_criterion
from train_utils.moment_training import run_moment_training_loop
from train_utils.t03_reporting import generate_t03_report


MODEL_FACTORIES = {
    "clip": (create_clip, "clip_vit_b16"),
    "siglip": (create_siglip, "siglip_vit_b16"),
}


def load_config(path):
    with open(path, encoding="utf-8") as f:
        settings = yaml.safe_load(f)

    if not isinstance(settings, dict):
        raise ValueError("YAML configuration must be a mapping")

    model_name = settings.get("model")
    if model_name not in MODEL_FACTORIES:
        raise ValueError(f"Unsupported T04 model: {model_name}")

    moments = settings.get("moments")
    if (
        not isinstance(moments, list)
        or not moments
        or len(set(moments)) != len(moments)
        or any(type(m) is not int or m not in range(1, 6)
               for m in moments)
    ):
        raise ValueError("moments must contain distinct integers 1..5")

    for key in ("dataset_root", "output_dir"):
        if not settings.get(key):
            raise ValueError(f"Missing required setting: {key}")

    return settings


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    settings = load_config(args.config)

    root = Path(settings["dataset_root"]).expanduser().resolve()
    out = Path(settings["output_dir"]).expanduser().resolve()

    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root missing: {root}")

    # Check the published split files before accessing the GPU.
    split_dir = root / "splits"
    for split in ("train.csv", "val.csv", "test.csv"):
        if not (split_dir / split).is_file():
            raise FileNotFoundError(f"Missing {split_dir / split}")

    if out.exists():
        raise FileExistsError(
            f"Output directory already exists: {out}"
        )

    if args.check:
        print("T04 configuration: PASS")
        print("Model:", settings["model"])
        print("Moments:", settings["moments"])
        print("Dataset:", root)
        print("Output:", out)
        return

    if not torch.cuda.is_available():
        raise RuntimeError("T04 GPU training requires CUDA")

    device = torch.device("cuda")
    seed = int(settings.get("seed", 42))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    moments = settings["moments"]

    train_loader, val_loader, _ = get_moment_dataloaders(
        dataset_root=root,
        moments=moments,
        batch_size=int(settings.get("batch_size", 4)),
        num_workers=int(settings.get("num_workers", 0)),
        seed=seed,
        validate_files=True,
        check_normalization=True,
    )

    factory, backbone = MODEL_FACTORIES[settings["model"]]

    model, optimizer = factory(
        backbone=backbone,
        input_shape=(len(moments), 32, 32),
        dataset_type="moment",
        freeze_backbone=bool(
            settings.get("freeze_backbone", True)
        ),
        pretrained=True,
        learning_rate=float(settings.get("learning_rate", 1e-4)),
    )

    model = model.to(device)

    criterion = build_moment_criterion(
        settings.get("moment_loss", {"type": "cross_entropy"}),
        device=device,
    )

    cfg = SimpleNamespace(
        output_dir=str(out),
        epochs=int(settings.get("epochs", 2)),
        patience=int(settings.get("patience", 5)),
        moments=moments,
        loss_weights={
            "alpha_output": 1.0,
            "q0_output": 1.0,
        },
        moment_loss=settings.get(
            "moment_loss", {"type": "cross_entropy"}
        ),
    )

    out.mkdir(parents=True)
    (out / "config.resolved.json").write_text(
        json.dumps(settings, indent=2)
    )

    best_epoch, best_acc, _ = run_moment_training_loop(
        cfg,
        train_loader,
        val_loader,
        device,
        model,
        criterion,
        optimizer,
        scheduler=None,
    )

    print("Best epoch:", best_epoch)
    print("Best validation joint accuracy:", best_acc)

    generate_t03_report(
        model,
        val_loader,
        device,
        out,
        expected_samples=len(val_loader.dataset),
    )

    print("T04 training completed successfully.")


if __name__ == "__main__":
    main()
