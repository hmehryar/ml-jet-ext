"""CPU gates and optional CUDA smoke for additive T04-10 implementation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models.t04_10_single_head import (
    SingleHeadMomentVisionClassifier, SingleHeadViTClassifier,
    build_trainable_optimizer, TASK_OUTPUT,
)
from train_utils.t04_10_training import run_single_head_epoch
from train_utils.t04_10_checkpoint import fit_fixed_budget


class FakeVisionBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.conv = nn.Conv2d(3, 8, 1)

    def forward(self, pixel_values):
        return SimpleNamespace(pooler_output=self.conv(pixel_values).mean((2, 3)))


class FakeViTBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_features = 8
        self.conv = nn.Conv2d(3, 8, 1)

    def forward(self, x):
        return self.conv(x).mean((2, 3))


class ToyData(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return torch.ones(5, 32, 32) * (idx + 1) / 5, {
            "alpha_output": torch.tensor(idx % 7),
            "q0_output": torch.tensor(idx % 5),
        }


def loader():
    return DataLoader(ToyData(), batch_size=2, shuffle=False)


@pytest.mark.parametrize("task,classes", [("alpha_only", 7), ("q0_only", 5)])
def test_clip_siglip_style_single_head_shape_optimizer_grad(task, classes):
    model = SingleHeadMomentVisionClassifier(FakeVisionBackbone(), task=task, strategy="S1")
    assert not hasattr(model, "alpha_head") and not hasattr(model, "q0_head")
    assert model.classification_head.out_features == classes
    assert set(model(torch.randn(2, 5, 32, 32))) == {TASK_OUTPUT[task]}
    optim = build_trainable_optimizer(model)
    assert all(not p.requires_grad for p in model.backbone.parameters())
    metrics = run_single_head_epoch(loader(), model, task=task, device="cpu", optimizer=optim)
    assert metrics["n"] == 4 and 0 <= metrics["macro_f1"] <= 1
    assert "joint_accuracy" not in metrics
    assert model.classification_head.weight.grad is not None
    assert torch.isfinite(model.classification_head.weight.grad).all()


@pytest.mark.parametrize("task,classes", [("alpha_only", 7), ("q0_only", 5)])
def test_vit_style_single_head(task, classes):
    model = SingleHeadViTClassifier(FakeViTBackbone(), task=task)
    assert model.classification_head.out_features == classes
    assert set(model(torch.randn(2, 5, 32, 32))) == {TASK_OUTPUT[task]}
    result = run_single_head_epoch(loader(), model, task=task, device="cpu")
    assert result["n"] == 4


def test_missing_other_target_is_fine():
    class AlphaOnly(Dataset):
        def __len__(self): return 2
        def __getitem__(self, idx):
            return torch.ones(5, 32, 32), {"alpha_output": torch.tensor(idx % 7)}
    model = SingleHeadMomentVisionClassifier(FakeVisionBackbone(), task="alpha_only")
    result = run_single_head_epoch(DataLoader(AlphaOnly(), batch_size=2),
                                   model, task="alpha_only", device="cpu")
    assert result["n"] == 2


def test_task_mismatch_and_unsupported_tasks():
    with pytest.raises(ValueError):
        SingleHeadMomentVisionClassifier(FakeVisionBackbone(), task="joint")
    model = SingleHeadMomentVisionClassifier(FakeVisionBackbone(), task="alpha_only")
    with pytest.raises(ValueError):
        run_single_head_epoch(loader(), model, task="q0_only", device="cpu")


def test_exact_resume_equivalent(tmp_path):
    def new_model():
        return SingleHeadMomentVisionClassifier(FakeVisionBackbone(), task="alpha_only", strategy="S1")
    torch.manual_seed(777)
    full_model = new_model()
    torch.manual_seed(777)
    resumed_model = new_model()
    o1 = build_trainable_optimizer(full_model)
    o2 = build_trainable_optimizer(resumed_model)
    config_hash = hashlib.sha256(b"T04-10-fixed-fixture").hexdigest()
    fit_fixed_budget(model=full_model, train_loader=loader(), val_loader=loader(),
                     optimizer=o1, task="alpha_only", output_dir=tmp_path / "full",
                     config_sha256=config_hash, epochs=3)
    fit_fixed_budget(model=resumed_model, train_loader=loader(), val_loader=loader(),
                     optimizer=o2, task="alpha_only", output_dir=tmp_path / "resumed",
                     config_sha256=config_hash, epochs=1)
    fit_fixed_budget(model=resumed_model, train_loader=loader(), val_loader=loader(),
                     optimizer=o2, task="alpha_only", output_dir=tmp_path / "resumed",
                     config_sha256=config_hash, epochs=3, resume=True)
    for k, v in full_model.state_dict().items():
        torch.testing.assert_close(v, resumed_model.state_dict()[k], rtol=0, atol=0)
    a = json.loads((tmp_path / "full/history.json").read_text())
    b = json.loads((tmp_path / "resumed/history.json").read_text())
    assert a == b


def test_resume_wrong_config_rejected(tmp_path):
    model = SingleHeadMomentVisionClassifier(FakeVisionBackbone(), task="q0_only")
    optim = build_trainable_optimizer(model)
    fit_fixed_budget(model=model, train_loader=loader(), val_loader=loader(),
                     optimizer=optim, task="q0_only", output_dir=tmp_path / "run",
                     config_sha256="0" * 64, epochs=1)
    with pytest.raises(ValueError, match="mismatch"):
        fit_fixed_budget(model=model, train_loader=loader(), val_loader=loader(),
                         optimizer=optim, task="q0_only", output_dir=tmp_path / "run",
                         config_sha256="1" * 64, epochs=2, resume=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable on this runner")
@pytest.mark.parametrize("task", ["alpha_only", "q0_only"])
def test_cuda_bf16_smoke(task):
    model = SingleHeadMomentVisionClassifier(FakeVisionBackbone(), task=task).cuda()
    optim = build_trainable_optimizer(model)
    result = run_single_head_epoch(loader(), model, task=task, device="cuda",
                                   precision="bf16", optimizer=optim)
    assert result["n"] == 4

@pytest.mark.parametrize("task", ["alpha_only", "q0_only"])
def test_cpu_bf16_autocast(task):
    model = SingleHeadMomentVisionClassifier(FakeVisionBackbone(), task=task, strategy="S3")
    optimizer = build_trainable_optimizer(model)
    result = run_single_head_epoch(loader(), model, task=task, device="cpu",
                                   precision="bf16", optimizer=optimizer)
    assert result["n"] == 4
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_resume_history_reconciliation_after_atomic_checkpoint(tmp_path):
    model = SingleHeadMomentVisionClassifier(FakeVisionBackbone(), task="q0_only")
    optim = build_trainable_optimizer(model)
    where = tmp_path / "run"
    fit_fixed_budget(model=model, train_loader=loader(), val_loader=loader(),
                     optimizer=optim, task="q0_only", output_dir=where,
                     config_sha256="a" * 64, epochs=2)
    (where / "history.json").write_text("[]")
    fit_fixed_budget(model=model, train_loader=loader(), val_loader=loader(),
                     optimizer=optim, task="q0_only", output_dir=where,
                     config_sha256="a" * 64, epochs=3, resume=True)
    assert len(json.loads((where / "history.json").read_text())) == 3
