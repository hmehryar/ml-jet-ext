"""T04-10 session production contract, CPU-only; no /rs writes or Slurm jobs."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_utils import t04_10_sessions as sessions
from train_utils.t04_10_session_training import run_session
from models.t04_10_single_head import SingleHeadMomentVisionClassifier, build_trainable_optimizer


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.proj = nn.Conv2d(3, 4, 1)

    def forward(self, pixel_values):
        return SimpleNamespace(pooler_output=self.proj(pixel_values).mean((2, 3)))


class Samples(Dataset):
    def __len__(self):
        return 6

    def __getitem__(self, index):
        return (torch.ones(2, 32, 32) * (index + 1) / 10,
                {'alpha_output': torch.tensor(index % 7),
                 'q0_output': torch.tensor(index % 5)})


@pytest.fixture
def run(tmp_path, monkeypatch):
    (tmp_path / 'config.resolved.json').write_text('{}')
    digest = hashlib.sha256((tmp_path / 'config.resolved.json').read_bytes()).hexdigest()
    (tmp_path / 'run_manifest.json').write_text(json.dumps({'config_sha256': digest}))
    monkeypatch.setattr(sessions, '_validate_run', lambda directory: Path(directory).resolve())
    return tmp_path, digest


def components(task):
    model = SingleHeadMomentVisionClassifier(Backbone(), task=task, channels=2, strategy='S3')
    optim = build_trainable_optimizer(model)
    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min')
    dl = DataLoader(Samples(), batch_size=2, shuffle=False)
    return dict(model=model, optimizer=optim, scheduler=schedule,
                train_loader=dl, val_loader=dl)


@pytest.mark.parametrize('task', ['alpha_only', 'q0_only'])
def test_two_sessions_checkpoint_history_telemetry(run, task):
    directory, digest = run
    torch.manual_seed(10)
    with sessions.experiment_lock(directory):
        first, outcome1 = run_session(run=directory, mode='fresh', task=task,
            config_sha256=digest, epochs=2, **components(task))
    assert first.name == 'session-001'
    assert outcome1['epoch'] == 2
    assert sessions.latest_checkpoint(directory)['epoch'] == 2
    manifest = json.loads((first / 'session_manifest.json').read_text())
    assert manifest['status'] == 'completed'
    assert len(list((first / 'telemetry').glob('epoch_*.json'))) == 2
    assert (first / 'single_head_epoch_metrics.csv').is_file()
    first_digest = sessions._digest(sessions.latest_checkpoint(directory)['absolute_path'])
    with sessions.experiment_lock(directory):
        second, outcome2 = run_session(run=directory, mode='resume', task=task,
            config_sha256=digest, epochs=4, **components(task))
    assert second.name == 'session-002'
    assert outcome2['epoch'] == 4
    assert sessions.latest_checkpoint(directory)['epoch'] == 4
    assert sessions._digest(directory/'sessions/session-001/checkpoints/epoch_000002.pth') == first_digest
    assert len(list((second/'telemetry').glob('epoch_*.json'))) == 2
    assert json.loads((second/'session_manifest.json').read_text())['parent_checkpoint']['epoch'] == 2
    global_history = json.loads((directory/'history.json').read_text())
    assert [row['epoch'] for row in global_history] == [1, 2, 3, 4]
    for name in ('alpha_only', 'q0_only'):
        assert all(row['val']['task'] == task for row in global_history)
    assert not (directory/'training').exists()


def test_refuse_corrupt_checkpoint(run):
    directory, digest = run
    with sessions.experiment_lock(directory):
        run_session(run=directory, mode='fresh', task='q0_only',
                    config_sha256=digest, epochs=1, **components('q0_only'))
    pointer = sessions.latest_checkpoint(directory)
    with open(pointer['absolute_path'], 'ab') as f:
        f.write(b'tampering')
    with pytest.raises(sessions.SessionError, match='SHA-256'):
        sessions.latest_checkpoint(directory)
    with sessions.experiment_lock(directory):
        with pytest.raises(sessions.SessionError, match='SHA-256'):
            run_session(run=directory, mode='resume', task='q0_only',
                        config_sha256=digest, epochs=2, **components('q0_only'))


def test_interrupted_session_is_reconciled(run):
    directory, digest = run
    with sessions.experiment_lock(directory):
        first, _ = run_session(run=directory, mode='fresh', task='alpha_only',
                config_sha256=digest, epochs=1, **components('alpha_only'))
        second = sessions.start_session(directory, mode='resume', target_epochs=3)
        assert json.loads((second/'session_manifest.json').read_text())['status'] == 'reserved'
    # Simulate process death before saving an epoch; resume creates session-003.
    with sessions.experiment_lock(directory):
        third, outcome = run_session(run=directory, mode='resume', task='alpha_only',
                config_sha256=digest, epochs=3, **components('alpha_only'))
    assert json.loads((second/'session_manifest.json').read_text())['status'] == 'interrupted'
    assert third.name == 'session-003'
    assert outcome['epoch'] == 3


def test_recovered_trajectory_matches_uninterrupted(run, tmp_path, monkeypatch):
    """Deterministic resume: same task weights and optimizer path after two sessions."""
    split_run, digest = run
    full = tmp_path/'continuous'
    full.mkdir()
    (full/'config.resolved.json').write_text('{}')
    (full/'run_manifest.json').write_text(json.dumps({'config_sha256': digest}))
    torch.manual_seed(321)
    with sessions.experiment_lock(split_run):
        run_session(run=split_run, mode='fresh', task='q0_only',
                    config_sha256=digest, epochs=2, **components('q0_only'))
    # New model initializations deliberately consume RNG; resumed checkpoint
    # restores all saved states and must ignore this new initialization.
    with sessions.experiment_lock(split_run):
        run_session(run=split_run, mode='resume', task='q0_only',
                    config_sha256=digest, epochs=4, **components('q0_only'))
    resumed = torch.load(sessions.latest_checkpoint(split_run)['absolute_path'],
                         map_location='cpu', weights_only=False)
    torch.manual_seed(321)
    with sessions.experiment_lock(full):
        run_session(run=full, mode='fresh', task='q0_only',
                    config_sha256=digest, epochs=4, **components('q0_only'))
    continuous = torch.load(sessions.latest_checkpoint(full)['absolute_path'],
                            map_location='cpu', weights_only=False)
    for key in resumed['model']:
        torch.testing.assert_close(resumed['model'][key], continuous['model'][key], rtol=0, atol=0)
    assert resumed['history'] == continuous['history']
