"""T04-11 reproducible, sample-aligned prediction export."""

import argparse
import csv
import gc
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.moment_loader import MomentJetDataset
from tools.t04_11.smoke_inference import build_model


MANIFEST = Path("results/t04_11/audit/checkpoint_candidates.csv")
OUTPUT_ROOT = Path("results/t04_11/predictions")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def export_one(row, split, batch_size, workers, force):
    role = row["role"]
    seed = int(row["seed"])
    run_dir = Path(row["run_directory"])
    checkpoint_path = Path(row["checkpoint"])
    cfg_path = run_dir / "config.resolved.json"
    cfg = json.loads(cfg_path.read_text())

    output_dir = OUTPUT_ROOT / split / f"seed-{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"{role}.npz"
    json_path = output_dir / f"{role}.json"

    if npz_path.exists() or json_path.exists():
        if not force:
            raise FileExistsError(
                f"Output exists for {role}/seed-{seed}. "
                "Use --force to replace it."
            )

    dataset_root = Path(cfg["dataset_root"])
    split_path = dataset_root / "splits" / f"{split}.csv"

    dataset = MomentJetDataset(
        csv_path=split_path,
        dataset_root=dataset_root,
        moments=tuple(cfg["moments"]),
    )

    # Preserve the exact CSV row identity used by MomentJetDataset.
    sample_ids = dataset.df["agg_file_path"].to_numpy(dtype=str)
    alpha_true = np.asarray(dataset.alpha_labels, dtype=np.int64)
    q0_true = np.asarray(dataset.q0_labels, dtype=np.int64)

    if len(set(sample_ids.tolist())) != len(dataset):
        raise ValueError("Duplicate sample IDs")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,  # User-owned, trusted checkpoints.
    )

    expected_task = (
        "joint" if role.endswith("_joint")
        else "alpha_only" if role.endswith("_alpha")
        else "q0_only"
    )
    if expected_task != "joint":
        assert cfg["task"] == expected_task
        assert checkpoint["task"] == expected_task

    model = build_model(role, cfg)
    state = (
        checkpoint["model_state_dict"]
        if expected_task == "joint"
        else checkpoint["model"]
    )
    model.load_state_dict(state, strict=True)

    best_epoch = int(checkpoint["best_epoch"])
    assert int(checkpoint["epoch"]) == best_epoch
    stored_best_metric = (
        checkpoint.get("best_metric")
        if expected_task == "joint"
        else checkpoint.get("best_f1")
    )

    del checkpoint, state

    device = torch.device("cuda")
    model = model.to(device).eval()

    all_logits = {}
    observed_alpha = []
    observed_q0 = []

    precision = cfg.get("precision", "fp32").lower()
    if precision not in {"bf16", "fp32"}:
        raise ValueError(
            f"Unsupported inference precision: {precision}"
        )

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=(precision == "bf16"),
            ):
                outputs = model(images)

            for head, value in outputs.items():
                value = value.float().cpu().numpy()
                if not np.isfinite(value).all():
                    raise ValueError(
                        f"Nonfinite logits: {role}, {head}"
                    )
                all_logits.setdefault(head, []).append(value)

            observed_alpha.extend(
                labels["alpha_output"].tolist()
            )
            observed_q0.extend(
                labels["q0_output"].tolist()
            )

    assert np.array_equal(observed_alpha, alpha_true)
    assert np.array_equal(observed_q0, q0_true)

    arrays = {
        "sample_ids": sample_ids,
        "true_alpha": alpha_true,
        "true_q0": q0_true,
    }

    for head, chunks in all_logits.items():
        matrix = np.concatenate(chunks, axis=0)
        assert matrix.shape[0] == len(dataset)
        arrays[head.replace("_output", "_logits")] = matrix

    expected_shapes = {
        "alpha_logits": (len(dataset), 7),
        "q0_logits": (len(dataset), 5),
    }

    expected_heads = (
        set(expected_shapes)
        if expected_task == "joint"
        else {
            "alpha_logits"
            if expected_task == "alpha_only"
            else "q0_logits"
        }
    )
    assert set(arrays) - {
        "sample_ids", "true_alpha", "true_q0"
    } == expected_heads

    for head in expected_heads:
        assert arrays[head].shape == expected_shapes[head]

    metrics = {}
    if "alpha_logits" in arrays:
        metrics["alpha_accuracy"] = float(
            np.mean(
                arrays["alpha_logits"].argmax(axis=1)
                == alpha_true
            )
        )
    if "q0_logits" in arrays:
        metrics["q0_accuracy"] = float(
            np.mean(
                arrays["q0_logits"].argmax(axis=1)
                == q0_true
            )
        )
    if expected_task == "joint":
        metrics["joint_accuracy"] = float(
            np.mean(
                (arrays["alpha_logits"].argmax(axis=1) == alpha_true)
                &
                (arrays["q0_logits"].argmax(axis=1) == q0_true)
            )
        )

    metadata = {
        "schema": "t04_11_predictions_v1",
        "role": role,
        "seed": seed,
        "split": split,
        "sample_count": len(dataset),
        "run_id": run_dir.name,
        "training_strategy": cfg["strategy"],
        "inference_precision": precision,
        "moments": cfg["moments"],
        "model": cfg["model"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_best_epoch": best_epoch,
        "checkpoint_best_validation_metric": (
            float(stored_best_metric)
            if stored_best_metric is not None
            else None
        ),
        "config_path": str(cfg_path),
        "config_sha256": sha256(cfg_path),
        "split_csv": str(split_path),
        "split_sha256": sha256(split_path),
        "source_jet_overlap_disclosed": True,
        "git_commit": git_revision(),
        "metrics": metrics,
    }

    # Save metadata only after all predictions have passed validation.
    np.savez_compressed(npz_path, **arrays)
    metadata["prediction_npz_sha256"] = sha256(npz_path)
    json_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )

    print(
        f"PASS {role:14s} seed={seed:4d} "
        f"n={len(dataset)} metrics={metrics}",
        flush=True,
    )

    del model, loader, dataset
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"],
                        default="val")
    parser.add_argument("--seed", type=int, choices=[42, 123, 2026])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    assert torch.cuda.is_available()

    with MANIFEST.open() as f:
        rows = list(csv.DictReader(f))

    selected = [
        row for row in rows
        if args.seed is None or int(row["seed"]) == args.seed
    ]

    expected_count = 18 if args.seed is None else 6
    assert len(selected) == expected_count

    for row in selected:
        export_one(
            row, args.split, args.batch_size,
            args.workers, args.force
        )

    print(f"\nEXPORTED {len(selected)} PREDICTION CACHES")


if __name__ == "__main__":
    main()
