"""T04-11: six-role seed-42 inference smoke test."""

import csv
import gc
import json
from pathlib import Path

import torch

from data.moment_loader import MomentJetDataset
from models.model_clip import CLIPClassifier
from models.model_siglip import SigLIPClassifier
from models.t04_10_single_head import build_clip_siglip_single_head


MANIFEST = Path("results/t04_11/audit/checkpoint_candidates.csv")


def build_model(role, cfg):
    architecture = cfg["model"]
    strategy = cfg["strategy"]
    moments = cfg["moments"]
    pretrained = strategy != "S4"

    common = dict(
        checkpoint=cfg["checkpoint"],
        revision=cfg.get("checkpoint_revision"),
        strategy=strategy,
        pretrained=pretrained,
    )

    if role.endswith("_joint"):
        cls = {
            "clip": CLIPClassifier,
            "siglip": SigLIPClassifier,
        }[architecture]

        return cls(
            input_shape=(len(moments), 32, 32),
            **common,
        )

    return build_clip_siglip_single_head(
        architecture=architecture,
        task=cfg["task"],
        channels=len(moments),
        **common,
    )


def main():
    assert torch.cuda.is_available(), "GPU is required"
    device = torch.device("cuda:0")

    with MANIFEST.open() as f:
        rows = [
            row for row in csv.DictReader(f)
            if int(row["seed"]) == 42
        ]

    assert len(rows) == 6

    for row in rows:
        role = row["role"]
        run_dir = Path(row["run_directory"])
        config_path = run_dir / "config.resolved.json"

        cfg = json.loads(config_path.read_text())
        moments = tuple(cfg["moments"])

        print(f"\n{'=' * 60}")
        print("ROLE:", role)
        print("STRATEGY:", cfg["strategy"])
        print("MOMENTS:", moments)

        model = build_model(role, cfg)

        ckpt = torch.load(
            row["checkpoint"],
            map_location="cpu",
            weights_only=False,
        )

        state = (
            ckpt["model_state_dict"]
            if role.endswith("_joint")
            else ckpt["model"]
        )

        model.load_state_dict(state, strict=True)
        print("STRICT STATE-DICT LOAD: PASS")

        model = model.to(device).eval()

        dataset_root = Path(cfg["dataset_root"])
        dataset = MomentJetDataset(
            csv_path=dataset_root / "splits/val.csv",
            dataset_root=dataset_root,
            moments=moments,
        )

        examples = [dataset[i] for i in range(2)]
        images = torch.stack([x for x, _ in examples]).to(device)

        with torch.inference_mode():
            outputs = model(images)

        expected = (
            {"alpha_output": (2, 7), "q0_output": (2, 5)}
            if role.endswith("_joint")
            else {
                "alpha_output" if cfg["task"] == "alpha_only"
                else "q0_output":
                (2, 7) if cfg["task"] == "alpha_only" else (2, 5)
            }
        )

        assert set(outputs) == set(expected)

        for key, shape in expected.items():
            value = outputs[key]
            assert tuple(value.shape) == shape
            assert torch.isfinite(value).all()

            print(
                f"{key}: shape={tuple(value.shape)} "
                f"finite=PASS "
                f"predictions={value.argmax(dim=1).tolist()}"
            )

        print("FORWARD PASS: PASS")

        del model, ckpt, state, outputs, images, dataset
        gc.collect()
        torch.cuda.empty_cache()

    print("\nALL SIX MODEL ROLES PASSED")


if __name__ == "__main__":
    main()
