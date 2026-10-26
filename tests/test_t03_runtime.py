"""Synthetic safety and reproducibility regression tests (no GPU or real data)."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

from config import get_config
from train_utils import moment_training
from train_utils.t03_runtime import parser, prepare, reserve_run, resolve_scheduler, verify_source


@pytest.fixture(autouse=True)
def writable_temporary_snapshots(tmp_path, monkeypatch):
    monkeypatch.setattr('train_utils.t03_runtime._git',
                        lambda repo: {'commit':'synthetic', 'status':'', 'diff':''})
    yield
    # Production snapshots stay read-only. Allow pytest to clean temporary ones.
    for path in tmp_path.rglob('*'):
        if path.is_dir():
            path.chmod(0o755)
        elif path.is_file():
            path.chmod(0o644)


def fixture_config(tmp_path):
    root = tmp_path / 'dataset'
    (root / 'splits').mkdir(parents=True)
    for name in ('train', 'val', 'test'):
        (root / 'splits' / (name + '.csv')).write_text('metadata only\n')
    checkpoint = tmp_path / 'checkpoint.pth'
    checkpoint.write_bytes(b'not deserialized during preparation')
    raw = dict(dataset_root=str(root), dataset_type='moment', moments=[1, 2],
               model_tag='E2', backbone='vit_gaussian', transfer_mode='jetbench_full',
               checkpoint_path=str(checkpoint), epochs=7, num_workers=1, seed=42,
               output_dir=str(tmp_path / 'runs'), scheduler={'type':'ReduceLROnPlateau', 'mode':'max'})
    config = tmp_path / 'experiment.yml'
    config.write_text(yaml.safe_dump(raw))
    return config, raw


def test_override_precedence_and_defaults(tmp_path):
    config, raw = fixture_config(tmp_path)
    cfg = get_config(str(config))
    assert cfg.dataset_root_dir == raw['dataset_root']
    assert cfg.num_workers == 1 and cfg.epochs == 7
    other = dict(dataset_root=str(tmp_path / 'other'), num_workers=0, output_dir=str(tmp_path / 'new'))
    changed = get_config(str(config), overrides=other)
    assert changed.dataset_root_dir == other['dataset_root']
    assert changed.num_workers == 0
    assert changed.output_dir.startswith(other['output_dir'])
    assert yaml.safe_load(config.read_text()) == raw


@pytest.mark.parametrize('monitor,mode', [('val_loss','min'), ('val_joint_accuracy','max')])
def test_scheduler_resolution(monitor, mode):
    assert resolve_scheduler({'monitor':monitor, 'mode':mode})['monitor'] == monitor
    with pytest.raises(ValueError, match='mismatch'):
        resolve_scheduler({'monitor':monitor, 'mode':'max' if mode == 'min' else 'min'})


def test_atomic_reservation(tmp_path):
    target = tmp_path / 'same'
    def attempt(_):
        try:
            reserve_run(target)
            return True
        except FileExistsError:
            return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(attempt, range(2))) == [False, True]
    (target / 'evidence').write_text('preserve')
    with pytest.raises(FileExistsError):
        reserve_run(target)
    assert (target / 'evidence').read_text() == 'preserve'


def test_snapshot_and_provenance(tmp_path):
    config, raw = fixture_config(tmp_path)
    args = parser().parse_args(['--config', str(config), '--prepare-only', '--workers', '0',
                                '--device', 'cpu', '--run-id', 'synthetic'])
    out = prepare(args)
    verify_source(out)
    manifest = json.loads((out / 'run_manifest.json').read_text())
    json.dumps(manifest, allow_nan=False)
    assert out.parts[-4:] == ('local', 'E2', 'seed-42', 'synthetic')
    assert manifest['resolved']['num_workers'] == 0
    assert manifest['resolved']['device'] == 'cpu'
    assert manifest['scheduler']['monitor'] == 'val_joint_accuracy'
    assert set(manifest['inputs']) == {'checkpoint','train_split','validation_split','test_split'}
    assert (out / 'source/train_t03_run.py').is_file()
    assert not (out / 'source/AGENTS.md').exists()
    assert not (out / 'source/environment.local.yml').exists()
    assert not ((out / 'source/train_utils/t03_runner.py').stat().st_mode & 0o222)
    env = dict(os.environ, PYTHONPATH=str(out / 'source'), PYTHONDONTWRITEBYTECODE='1')
    # Use a fresh interpreter: imports must resolve to the snapshot, not checkout.
    result = subprocess.check_output([sys.executable, '-B', '-c',
        'import train_utils.t03_runner as r; print(r.__file__)'], cwd=out / 'source', env=env, text=True)
    assert str(out / 'source/train_utils/t03_runner.py') in result
    with pytest.raises(FileExistsError):
        prepare(args)
    # Restore permissions only inside the temporary fixture to test tamper detection.
    archived = out / 'source/config.py'
    archived.chmod(0o644)
    archived.write_text('# changed\n')
    with pytest.raises(RuntimeError, match='hash'):
        verify_source(out)


def test_profiles_separate_and_checkpoint_override(tmp_path):
    config, raw = fixture_config(tmp_path)
    replacement = tmp_path / 'replacement.pth'
    replacement.write_bytes(b'override')
    root = tmp_path / 'explicit'
    paths = []
    for profile in ('local', 'hpc'):
        args = parser().parse_args(['--config', str(config), '--prepare-only', '--profile', profile,
                                   '--output-root', str(root), '--checkpoint', str(replacement),
                                   '--epochs', '1', '--run-id', 'same'])
        out = prepare(args); paths.append(out)
        cfg = json.loads((out / 'config.resolved.json').read_text())
        assert cfg['checkpoint_path'] == str(replacement) and cfg['epochs'] == 1
    assert paths[0] != paths[1]


@pytest.mark.parametrize('monitor,mode,expected', [('val_loss','min',2.5), ('val_joint_accuracy','max',0.25)])
def test_loop_scheduler_uses_declared_metric(tmp_path, monkeypatch, monitor, mode, expected):
    metrics = dict(loss=2.5, loss_alpha=1.0, loss_q0=1.5, accuracy=0.25,
                   alpha={'accuracy':0.5,'f1':0.3}, q0={'accuracy':0.5,'f1':0.4})
    monkeypatch.setattr(moment_training, 'train_moment_epoch', lambda *a: metrics)
    monkeypatch.setattr(moment_training, 'evaluate_moment', lambda *a: metrics)
    model = torch.nn.Linear(1,1)
    opt = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode=mode)
    cfg = SimpleNamespace(output_dir=str(tmp_path), epochs=1, patience=3,
                          scheduler={'monitor':monitor,'mode':mode})
    moment_training.run_moment_training_loop(cfg, [], [], 'cpu', model, {}, opt, scheduler)
    assert scheduler.best == expected
    checkpoint = torch.load(tmp_path / 'last_model.pth', weights_only=True)
    assert checkpoint['scheduler_settings']['monitor'] == monitor
    assert checkpoint['scheduler_state_dict']['best'] == expected
    assert checkpoint['metrics']['accuracy'] == 0.25  # selection remains accuracy
    with pytest.raises(FileExistsError):
        moment_training.run_moment_training_loop(cfg, [], [], 'cpu', model, {}, opt, scheduler)


def test_hpc_execution_requires_allocation(monkeypatch):
    from train_utils.t03_runner import check_execution_host
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    with pytest.raises(RuntimeError, match='Slurm'):
        check_execution_host('hpc')
