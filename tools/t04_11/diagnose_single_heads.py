"""Validate single-head checkpoints using their training precision."""

import csv
import gc
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from data.moment_loader import MomentJetDataset
from tools.t04_11.smoke_inference import build_model


MANIFEST = Path("results/t04_11/audit/checkpoint_candidates.csv")


def main():
    with MANIFEST.open() as f:
        rows = [
            row for row in csv.DictReader(f)
            if int(row["seed"]) == 42
            and not row["role"].endswith("_joint")
        ]

    assert len(rows) == 4
    results = []

    for row in rows:
        role = row["role"]
        run_dir = Path(row["run_directory"])
        cfg = json.loads(
            (run_dir / "config.resolved.json").read_text()
        )

        assert cfg["precision"].lower() == "bf16"

        checkpoint = torch.load(
            row["checkpoint"],
            map_location="cpu",
            weights_only=False,
        )

        model = build_model(role, cfg)
        model.load_state_dict(
            checkpoint["model"],
            strict=True,
        )
        model = model.cuda().eval()

        dataset_root = Path(cfg["dataset_root"])
        dataset = MomentJetDataset(
            csv_path=dataset_root / "splits/val.csv",
            dataset_root=dataset_root,
            moments=tuple(cfg["moments"]),
        )

        loader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=False,
            num_workers=2,
        )

        alpha_task = cfg["task"] == "alpha_only"
        output_key = (
            "alpha_output" if alpha_task else "q0_output"
        )
        class_count = 7 if alpha_task else 5

        predictions = []
        truths = []

        with torch.inference_mode():
            for images, labels in loader:
                images = images.cuda(non_blocking=True)

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):
                    outputs = model(images)

                predictions.extend(
                    outputs[output_key]
                    .argmax(dim=1)
                    .cpu()
                    .tolist()
                )
                truths.extend(
                    labels[output_key].tolist()
                )

        f1 = f1_score(
            truths,
            predictions,
            labels=list(range(class_count)),
            average="macro",
            zero_division=0,
        )
        accuracy = float(np.mean(
            np.asarray(predictions) == np.asarray(truths)
        ))
        stored = float(checkpoint["best_f1"])

        print(f"\nROLE: {role}")
        print(f"Stored macro-F1: {stored:.6f}")
        print(f"BF16 macro-F1:   {f1:.6f}")
        print(f"BF16 accuracy:   {accuracy:.6f}")
        print(f"Difference:      {f1 - stored:+.6f}")
        print(
            "Macro-F1 matches:",
            abs(f1 - stored) < 1e-6
        )

        results.append((role, stored, f1))

        del model, checkpoint, dataset, loader
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "=" * 65)
    print("SUMMARY")

    for role, stored, actual in results:
        print(
            f"{role:16s}: "
            f"{'PASS' if abs(actual-stored) < 1e-6 else 'INVESTIGATE'}"
        )


if __name__ == "__main__":
    main()
