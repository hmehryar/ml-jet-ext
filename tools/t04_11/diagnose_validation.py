"""Compare cached FP32 and fresh BF16 validation predictions."""

import csv
import json
import gc
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.moment_loader import MomentJetDataset
from tools.t04_11.smoke_inference import build_model


MANIFEST = Path("results/t04_11/audit/checkpoint_candidates.csv")
CACHE_ROOT = Path("results/t04_11/predictions/val/seed-42")


def accuracy(a_logits, q_logits, alpha, q0):
    alpha_correct = a_logits.argmax(1) == alpha
    q0_correct = q_logits.argmax(1) == q0
    return float(np.mean(alpha_correct & q0_correct))


def main():
    with MANIFEST.open() as f:
        rows = list(csv.DictReader(f))

    rows = [
        r for r in rows
        if int(r["seed"]) == 42
        and r["role"] in ("clip_joint", "siglip_joint")
    ]

    for row in rows:
        role = row["role"]
        cfg = json.loads(
            (Path(row["run_directory"]) /
             "config.resolved.json").read_text()
        )

        cached = np.load(CACHE_ROOT / f"{role}.npz")
        alpha = cached["true_alpha"]
        q0 = cached["true_q0"]

        fp32 = accuracy(
            cached["alpha_logits"],
            cached["q0_logits"],
            alpha, q0
        )

        checkpoint = torch.load(
            row["checkpoint"],
            map_location="cpu",
            weights_only=False,
        )

        model = build_model(role, cfg)
        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )
        model = model.cuda().eval()

        dataset = MomentJetDataset(
            csv_path=Path(cfg["dataset_root"]) / "splits/val.csv",
            dataset_root=cfg["dataset_root"],
            moments=tuple(cfg["moments"]),
        )
        loader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=False,
            num_workers=2,
        )

        preds_alpha = []
        preds_q0 = []

        with torch.inference_mode():
            for x, _ in loader:
                x = x.cuda()
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):
                    outputs = model(x)

                preds_alpha.extend(
                    outputs["alpha_output"].argmax(1).cpu().tolist()
                )
                preds_q0.extend(
                    outputs["q0_output"].argmax(1).cpu().tolist()
                )

        bf16 = float(np.mean(
            (np.asarray(preds_alpha) == alpha)
            &
            (np.asarray(preds_q0) == q0)
        ))

        stored = float(checkpoint["best_metric"])

        print(f"\n{'=' * 65}")
        print("ROLE:", role)
        print(f"Stored metric:     {stored:.6f}")
        print(f"Cached FP32:       {fp32:.6f}")
        print(f"Fresh BF16:        {bf16:.6f}")
        print(f"BF16 minus stored: {bf16 - stored:+.6f}")
        print(
            "BF16 matches stored:",
            abs(bf16 - stored) < 1e-12
        )

        del model, checkpoint, dataset, loader
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
