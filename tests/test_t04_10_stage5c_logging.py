"""T04-10: T04-07-compatible stdout telemetry without changed checkpoint semantics."""
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models.t04_10_single_head import SingleHeadMomentVisionClassifier, build_trainable_optimizer
from train_utils.t04_10_checkpoint import fit_fixed_budget
from train_utils.t04_10_training import run_single_head_epoch


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.proj = nn.Conv2d(3, 4, 1)

    def forward(self, pixel_values):
        return SimpleNamespace(pooler_output=self.proj(pixel_values).mean((2, 3)))


class TinyData(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, i):
        return torch.full((2, 32, 32), 0.1 * (i + 1)), {
            'alpha_output': torch.tensor(i % 7),
            'q0_output': torch.tensor(i % 5),
        }


def build(task):
    model = SingleHeadMomentVisionClassifier(
        Backbone(), task=task, channels=2, strategy='S3')
    optimizer = build_trainable_optimizer(model)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
    return model, optimizer, scheduler


@pytest.mark.parametrize('task', ('alpha_only', 'q0_only'))
def test_single_head_t04_07_format_and_resume(tmp_path, capsys, task):
    loader = DataLoader(TinyData(), batch_size=2)
    config_sha = hashlib.sha256(b'fixture').hexdigest()
    out = tmp_path / 'training'
    model, optimizer, scheduler = build(task)
    result = fit_fixed_budget(
        model=model, train_loader=loader, val_loader=loader,
        optimizer=optimizer, task=task, output_dir=out,
        config_sha256=config_sha, epochs=1, scheduler=scheduler,
        microbatch_size=2)
    assert result['epoch'] == 1
    lines = capsys.readouterr().out.splitlines()
    telemetry_lines = [l for l in lines if l.startswith('[T04-04] epoch telemetry: ')]
    assert len(telemetry_lines) == 1
    record = json.loads(telemetry_lines[0].split(': ', 1)[1])
    assert record['samples'] == 4
    assert record['microbatch_size'] == 2
    assert record['gradient_accumulation'] == 1
    assert record['configured_effective_batch'] == 2
    assert record['optimizer_steps_attempted'] == 2
    assert record['optimizer_steps_completed'] == 2
    assert record['optimizer_steps_skipped'] == 0
    assert record['elapsed_seconds'] > 0
    assert record['samples_per_second'] > 0
    assert isinstance(record['learning_rates'], dict)
    assert 'host_maxrss_kib' in record
    name = 'alpha' if task == 'alpha_only' else 'q0'
    assert any(l.startswith('[T04-10] Epoch 1 / 1: val ' + name + ' macro-F1 ')
               for l in lines)

    state_1 = torch.load(out / 'last.pth', weights_only=False)
    history_1 = json.loads((out / 'history.json').read_text())
    assert len(history_1) == 1
    assert 'telemetry' not in state_1  # no checkpoint schema change
    model2, opt2, sch2 = build(task)
    fit_fixed_budget(
        model=model2, train_loader=loader, val_loader=loader,
        optimizer=opt2, task=task, output_dir=out,
        config_sha256=config_sha, epochs=2, scheduler=sch2, resume=True,
        microbatch_size=2)
    lines = capsys.readouterr().out.splitlines()
    assert sum(x.startswith('[T04-04] epoch telemetry: ') for x in lines) == 1
    assert any(x.startswith('[T04-10] Epoch 2 / 2:') for x in lines)
    assert len(json.loads((out / 'history.json').read_text())) == 2


def test_count_steps_does_not_change_returned_metrics():
    loader = DataLoader(TinyData(), batch_size=2)
    model, optimizer, _ = build('alpha_only')
    steps = {}
    metrics = run_single_head_epoch(loader, model, task='alpha_only',
                                    device='cpu', optimizer=optimizer,
                                    telemetry_counts=steps)
    assert metrics['n'] == 4
    assert steps == dict(optimizer_steps=2, optimizer_steps_completed=2,
                         optimizer_steps_skipped=0)
    assert 'optimizer_steps' not in metrics
