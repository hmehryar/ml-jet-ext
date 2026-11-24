"""T04-02 Stage 4: fail-closed regression and recovery acceptance checks.

CPU fixture intentionally avoids downloads and GPU requirements. Run on Grid
alongside the complete production tests; real multi-process/GPU equivalence is
an additional acceptance gate, not implied by this fixture.
"""
import csv
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from train_utils import t04_sessions as sm
from train_utils import t04_session_training as tr
from train_utils.t04_resume_state import ResumeError, load_checkpoint, reconcile_epoch_csv, sha256


def setup_run(tmp_path, monkeypatch):
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'run_manifest.json').write_text('{}')
    (run / 'config.resolved.json').write_text(json.dumps({'scheduler': {'monitor': 'val_loss'}}))
    monkeypatch.setattr(sm, '_validate_run', lambda p: Path(p).resolve())
    return run


def report(score, loss=1.):
    return {'loss': loss, 'loss_alpha': loss/2, 'loss_q0': loss/2,
            'accuracy': score, 'alpha': {'accuracy': score, 'f1': score},
            'q0': {'accuracy': score, 'f1': score}}


def components(monkeypatch, scores):
    random.seed(9); np.random.seed(9); torch.manual_seed(9)
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.SGD(model.parameters(), lr=.02, momentum=.9)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=1, factor=.5)
    loader = SimpleNamespace(generator=torch.Generator())
    i = iter(scores)
    def train(*args):
        opt.zero_grad()
        x = torch.randn(4, 2)
        loss = model(x).square().mean()
        loss.backward()
        opt.step()
        return report(.4, float(loss.detach()))
    def evaluate(*args):
        return report(next(i))
    monkeypatch.setattr(tr, 'train_moment_epoch', train)
    monkeypatch.setattr(tr, 'evaluate_moment', evaluate)
    return model, opt, sched, loader


def session(run, mode, target, parts, scores, monkeypatch, *, min_epochs=1, patience=20, retention='last_best'):
    model, opt, sched, loader = components(monkeypatch, scores)
    with sm.experiment_lock(run):
        s = sm.start_session(run, mode=mode, target_epochs=target)
        tr.train_session(run=run, session=s, mode=mode, target_epochs=target,
            seed=42, min_epochs=min_epochs, patience=patience,
            train_loader=loader, val_loader=object(), model=model, optimizer=opt,
            scheduler=sched, criterion=None, device='cpu', checkpoint_retention=retention)
    return s


def checkpoint_state_for(run):
    pointer = sm.latest_checkpoint(run)
    return load_checkpoint(pointer['absolute_path'], map_location='cpu',
       manifest_sha256=sha256(run/'run_manifest.json'),
       config_sha256=sha256(run/'config.resolved.json'))


def test_resume_from_last_not_best_and_preserve_prior_hashes(tmp_path, monkeypatch):
    run = setup_run(tmp_path, monkeypatch)
    first = session(run, 'fresh', 3, None, [.9, .4, .3], monkeypatch)
    parent = sm.latest_checkpoint(run)
    assert parent['epoch'] == 3
    assert checkpoint_state_for(run)['best_epoch'] == 1
    best = first/'checkpoints/best_epoch_000001.pth'
    before = {p.name: sha256(p) for p in (first/'checkpoints').glob('*.pth')}
    assert best.exists()
    second = session(run, 'resume', 5, None, [.2, .1], monkeypatch)
    assert sm.latest_checkpoint(run)['epoch'] == 5
    state = checkpoint_state_for(run)
    assert state['best_epoch'] == 1
    assert state['stagnant'] == 4
    assert state['epoch'] == 5
    assert {p.name:sha256(p) for p in (first/'checkpoints').glob('*.pth')} == before
    assert sorted(p.name for p in (second/'checkpoints').glob('*.pth')) == ['epoch_000005.pth']
    with (run/'moment_epoch_metrics.csv').open() as f:
        assert [int(r['epoch']) for r in csv.DictReader(f)] == [1,2,3,4,5]


def test_reconcile_missing_csv_row_after_pointer_commit(tmp_path, monkeypatch):
    run = setup_run(tmp_path, monkeypatch)
    session(run, 'fresh', 2, None, [.8,.7], monkeypatch)
    state = checkpoint_state_for(run)
    path = run/'moment_epoch_metrics.csv'
    with path.open(newline='') as f:
        reader=csv.DictReader(f); rows=list(reader); columns=reader.fieldnames
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns); writer.writeheader(); writer.writerow(rows[0])
    reconcile_epoch_csv(path,state)
    reconcile_epoch_csv(path,state)
    with path.open(newline='') as f:
        assert [int(r['epoch']) for r in csv.DictReader(f)] == [1,2]


def test_corrupted_last_checkpoint_rejected_even_if_best_valid(tmp_path, monkeypatch):
    run=setup_run(tmp_path,monkeypatch)
    first=session(run,'fresh',3,None,[.9,.4,.3],monkeypatch)
    pointer=sm.latest_checkpoint(run)
    assert (first/'checkpoints/best_epoch_000001.pth').exists()
    with open(pointer['absolute_path'],'ab') as f: f.write(b'tamper')
    with pytest.raises(sm.SessionError,match='SHA-256'):
        sm.latest_checkpoint(run)
    with pytest.raises(sm.SessionError,match='SHA-256'):
        with sm.experiment_lock(run):
            sm.start_session(run,mode='resume',target_epochs=5)


def test_early_stopped_checkpoint_cannot_resume(tmp_path, monkeypatch):
    run=setup_run(tmp_path,monkeypatch)
    session(run,'fresh',5,None,[.9,.8],monkeypatch,patience=1)
    state=checkpoint_state_for(run)
    assert state['stopped'] and state['epoch']==2
    with pytest.raises(ResumeError,match='terminated by early stopping'):
        session(run,'resume',5,None,[.5,.4],monkeypatch,patience=1)


def test_default_retention_storage_bounded(tmp_path, monkeypatch):
    run=setup_run(tmp_path,monkeypatch)
    session(run,'fresh',8,None,[.9,.8,.7,.6,.5,.4,.3,.2],monkeypatch)
    checkpoints=list((run/'sessions/session-001/checkpoints').glob('*.pth'))
    assert len(checkpoints)==2
    assert {p.name for p in checkpoints} == {'best_epoch_000001.pth','epoch_000008.pth'}
    assert sm.latest_checkpoint(run)['epoch']==8


def _assert_nested_equal(a, b):
    """Exact recursive equality, including optimizer momentum and RNG tensors."""
    if torch.is_tensor(a):
        assert torch.is_tensor(b)
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray)
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            _assert_nested_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            _assert_nested_equal(x, y)
    else:
        assert a == b


def test_exact_uninterrupted_vs_resumed_trajectory(tmp_path, monkeypatch):
    """Identical initial conditions: 4 continuous epochs vs 2 + 2 resumed.

    This fixture exercises real torch forward/backward, optimizer momentum,
    ReduceLROnPlateau scheduler, stochastic training inputs, RNG restore,
    best/stagnant counters, and epoch CSV. It does not download a VLM.
    """
    baseline_root = tmp_path / 'baseline'
    resumed_root = tmp_path / 'resumed'
    baseline_root.mkdir()
    resumed_root.mkdir()
    baseline = setup_run(baseline_root, monkeypatch)
    resumed = setup_run(resumed_root, monkeypatch)
    scores = [.8, .7, .6, .9]  # best != LAST after epoch 3; new best at 4

    session(baseline, 'fresh', 4, None, scores, monkeypatch)
    baseline_state = checkpoint_state_for(baseline)

    first = session(resumed, 'fresh', 2, None, scores[:2], monkeypatch)
    previous_hashes = {p.name: sha256(p) for p in (first/'checkpoints').glob('*.pth')}
    session(resumed, 'resume', 4, None, scores[2:], monkeypatch)
    resumed_state = checkpoint_state_for(resumed)

    for field in ('model_state_dict', 'optimizer_state_dict',
                  'scheduler_state_dict', 'rng_state', 'best_metric',
                  'best_epoch', 'stagnant', 'stopped', 'metrics_row'):
        _assert_nested_equal(baseline_state[field], resumed_state[field])
    assert baseline_state['epoch'] == resumed_state['epoch'] == 4

    with (baseline/'moment_epoch_metrics.csv').open(newline='') as f:
        baseline_rows = list(csv.DictReader(f))
    with (resumed/'moment_epoch_metrics.csv').open(newline='') as f:
        resumed_rows = list(csv.DictReader(f))
    assert baseline_rows == resumed_rows
    assert [int(row['epoch']) for row in resumed_rows] == [1,2,3,4]
    assert {p.name: sha256(p) for p in (first/'checkpoints').glob('*.pth')} == previous_hashes
