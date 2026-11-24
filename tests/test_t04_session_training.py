import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from train_utils import t04_sessions as sm
from train_utils import t04_session_training as tr


def fixture_run(tmp_path, monkeypatch):
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'run_manifest.json').write_text('{}')
    (run / 'config.resolved.json').write_text(json.dumps({'scheduler': {'monitor': 'val_loss'}}))
    monkeypatch.setattr(sm, '_validate_run', lambda r: Path(r).resolve())
    monkeypatch.setattr(tr, 'latest_checkpoint', sm.latest_checkpoint)
    return run


def metrics(value, loss):
    return {'loss': loss, 'loss_alpha': loss / 2, 'loss_q0': loss / 2,
            'accuracy': value, 'alpha': {'accuracy': value, 'f1': value},
            'q0': {'accuracy': value, 'f1': value}}


def execute(run, monkeypatch, epochs, steps, checkpoint_retention='last_best'):
    torch.manual_seed(17)
    model = torch.nn.Linear(2, 1)
    optim = torch.optim.SGD(model.parameters(), lr=0.03, momentum=.9)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=1)
    loader = SimpleNamespace(generator=torch.Generator())
    def fake_train(loader, model, criterion, optimizer, device, weights):
        optimizer.zero_grad()
        x = torch.randn(3, 2)
        loss = model(x).square().mean()
        loss.backward()
        optimizer.step()
        return metrics(0.5, float(loss.detach()))
    def fake_eval(loader, model, criterion, device, weights):
        return metrics(0.5, 1.0)
    monkeypatch.setattr(tr, 'train_moment_epoch', fake_train)
    monkeypatch.setattr(tr, 'evaluate_moment', fake_eval)
    with sm.experiment_lock(run):
        for mode, target in steps:
            session = sm.start_session(run, mode=mode, target_epochs=target)
            tr.train_session(run=run, session=session, mode=mode,
                target_epochs=target, seed=42, min_epochs=1, patience=100,
                train_loader=loader, val_loader=object(), model=model,
                optimizer=optim, scheduler=sched, criterion=None, device='cpu',
                checkpoint_retention=checkpoint_retention)
    with (run / 'moment_epoch_metrics.csv').open() as f:
        rows = list(csv.DictReader(f))
    return model.state_dict(), optim.state_dict(), sched.state_dict(), rows


def test_uninterrupted_equals_resumed(tmp_path, monkeypatch):
    (tmp_path / 'a').mkdir()
    (tmp_path / 'b').mkdir()
    a = fixture_run(tmp_path / 'a', monkeypatch)
    b = fixture_run(tmp_path / 'b', monkeypatch)
    wa, oa, sa, ra = execute(a, monkeypatch, 4, [('fresh',4)])
    wb, ob, sb, rb = execute(b, monkeypatch, 4, [('fresh',2),('resume',4)])
    assert len(ra) == len(rb) == 4
    assert ra == rb
    assert all(torch.equal(wa[k],wb[k]) for k in wa)
    assert sa == sb
    assert (b / 'sessions/session-001/checkpoints/epoch_000002.pth').exists()
    assert not (b / 'sessions/session-001/checkpoints/epoch_000001.pth').exists()
    assert (b / 'sessions/session-001/checkpoints/best_epoch_000001.pth').exists()
    assert (b / 'sessions/session-002/checkpoints/epoch_000004.pth').exists()
    assert sm.latest_checkpoint(b)['epoch'] == 4


def test_cannot_resume_past_early_stop(tmp_path, monkeypatch):
    run = fixture_run(tmp_path, monkeypatch)
    execute(run, monkeypatch, 2, [('fresh',2)])
    # Ordinary completed epochs are resumable; no duplicate session number.
    with sm.experiment_lock(run):
        with pytest.raises(sm.SessionError):
            sm.start_session(run, mode='fresh', target_epochs=4)


def test_last_best_default_does_not_accumulate_epoch_checkpoints(tmp_path, monkeypatch):
    run = fixture_run(tmp_path, monkeypatch)
    execute(run, monkeypatch, 4, [('fresh', 4)])
    files = sorted(x.name for x in (run / 'sessions/session-001/checkpoints').glob('*.pth'))
    assert files == ['best_epoch_000001.pth', 'epoch_000004.pth']
    pointer = sm.latest_checkpoint(run)
    assert pointer['epoch'] == 4
    assert Path(pointer['absolute_path']).is_file()


def test_all_retention_keeps_every_epoch(tmp_path, monkeypatch):
    run = fixture_run(tmp_path, monkeypatch)
    execute(run, monkeypatch, 4, [('fresh', 4)], checkpoint_retention='all')
    files = sorted(x.name for x in (run / 'sessions/session-001/checkpoints').glob('*.pth'))
    assert files == [f'epoch_{i:06d}.pth' for i in range(1, 5)]


def test_pruning_cannot_touch_previous_session(tmp_path, monkeypatch):
    run = fixture_run(tmp_path, monkeypatch)
    execute(run, monkeypatch, 4, [('fresh', 2), ('resume', 4)])
    first = sorted(x.name for x in (run / 'sessions/session-001/checkpoints').glob('*.pth'))
    second = sorted(x.name for x in (run / 'sessions/session-002/checkpoints').glob('*.pth'))
    assert first == ['best_epoch_000001.pth', 'epoch_000002.pth']
    assert second == ['epoch_000004.pth']
