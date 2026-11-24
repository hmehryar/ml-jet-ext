"""Open-ended continuation safety and CPU trajectory tests."""
import hashlib
import json
from pathlib import Path

import pytest
import torch

from train_utils import t04_10_extension as ext
from train_utils import t04_10_sessions as sessions
from train_utils.t04_10_session_training import run_session
from test_t04_10_stage6_sessions import components


def test_targets_are_unbounded():
    for value in (81, 200, 201, 1000, 100000):
        assert ext.validate_target(value) == value
    for value in (0, -1, 1.5, True, '200'):
        with pytest.raises(ValueError):
            ext.validate_target(value)


def test_protocol_snapshot_hashes(tmp_path):
    folder = tmp_path/'continuation-protocol'
    source = folder/'source/train_utils'
    source.mkdir(parents=True)
    (source/'t04_10_extension.py').write_text('print("example")\n')
    manifest = {'schema': 1, 'name': 't04_10_unbounded_continuation',
                'source_sha256': ext.source_hashes(folder/'source')}
    (folder/'protocol_manifest.json').write_text(json.dumps(manifest))
    assert ext.verify_protocol(folder)['schema'] == 1
    (source/'t04_10_extension.py').write_text('print("tampered")\n')
    with pytest.raises(ValueError, match='SHA-256'):
        ext.verify_protocol(folder)


def test_80_to_200_to_205_without_changing_config(tmp_path, monkeypatch):
    # Uses short CPU epochs to exercise arbitrary target values without 200 costly passes.
    # Baseline is 2 epochs, continuation is to 4, then to 6. No run config changes.
    directory = tmp_path/'run'
    directory.mkdir()
    (directory/'config.resolved.json').write_text('{"epochs": 80}')
    config_hash = hashlib.sha256((directory/'config.resolved.json').read_bytes()).hexdigest()
    (directory/'run_manifest.json').write_text(json.dumps({'config_sha256': config_hash}))
    monkeypatch.setattr(sessions, '_validate_run', lambda path: Path(path).resolve())
    torch.manual_seed(45)
    with sessions.experiment_lock(directory):
        first, _ = run_session(run=directory, mode='fresh', task='alpha_only',
                               config_sha256=config_hash, epochs=2, **components('alpha_only'))
    original_manifest_hash = hashlib.sha256((directory/'run_manifest.json').read_bytes()).hexdigest()
    with sessions.experiment_lock(directory):
        second, _ = run_session(run=directory, mode='resume', task='alpha_only',
                                config_sha256=config_hash, epochs=4, **components('alpha_only'))
    with sessions.experiment_lock(directory):
        third, _ = run_session(run=directory, mode='resume', task='alpha_only',
                               config_sha256=config_hash, epochs=6, **components('alpha_only'))
    assert [p.name for p in (first, second, third)] == ['session-001', 'session-002', 'session-003']
    assert sessions.latest_checkpoint(directory)['epoch'] == 6
    assert hashlib.sha256((directory/'run_manifest.json').read_bytes()).hexdigest() == original_manifest_hash
    assert json.loads((third/'session_manifest.json').read_text())['parent_checkpoint']['epoch'] == 4
    assert [h['epoch'] for h in json.loads((directory/'history.json').read_text())] == list(range(1,7))


def test_real_80_epoch_checkpoint_resumes_at_81(tmp_path, monkeypatch):
    """Synthetic epoch-80 state tests the 80->81 boundary without 80 CPU epochs."""
    directory = tmp_path/'run'
    directory.mkdir()
    (directory/'config.resolved.json').write_text('{"epochs": 80}')
    config_hash = hashlib.sha256((directory/'config.resolved.json').read_bytes()).hexdigest()
    (directory/'run_manifest.json').write_text(json.dumps({'config_sha256': config_hash}))
    monkeypatch.setattr(sessions, '_validate_run', lambda path: Path(path).resolve())
    with sessions.experiment_lock(directory):
        first, _ = run_session(run=directory, mode='fresh', task='q0_only',
                               config_sha256=config_hash, epochs=2, **components('q0_only'))
    old = sessions.latest_checkpoint(directory)
    state = torch.load(old['absolute_path'], map_location='cpu', weights_only=False)
    for epoch in range(3, 81):
        row = dict(state['history'][-1])
        row['epoch'] = epoch
        state['history'].append(row)
    state['epoch'] = 80
    state['best_epoch'] = max(range(1,81), key=lambda i: state['history'][i-1]['val']['macro_f1'])
    state['best_f1'] = state['history'][state['best_epoch']-1]['val']['macro_f1']
    # Fabricate an immutable latest checkpoint at the historical boundary.
    checkpoint = first/'checkpoints/epoch_000080.pth'
    torch.save(state, checkpoint)
    pointer = {'schema_version': 1, 'epoch': 80, 'path': str(checkpoint.relative_to(directory)),
               'session': first.name, 'sha256': sessions._digest(checkpoint)}
    sessions._atomic_json(directory/'latest_checkpoint.json', pointer)
    with sessions.experiment_lock(directory):
        next_session, _ = run_session(run=directory, mode='resume', task='q0_only',
                                      config_sha256=config_hash, epochs=81, **components('q0_only'))
    assert sessions.latest_checkpoint(directory)['epoch'] == 81
    assert json.loads((next_session/'session_manifest.json').read_text())['parent_checkpoint']['epoch'] == 80
    assert json.loads((directory/'history.json').read_text())[-1]['epoch'] == 81
    with sessions.experiment_lock(directory):
        after, _ = run_session(run=directory, mode='resume', task='q0_only',
                               config_sha256=config_hash, epochs=82, **components('q0_only'))
    assert sessions.latest_checkpoint(directory)['epoch'] == 82
    assert json.loads((after/'session_manifest.json').read_text())['parent_checkpoint']['epoch'] == 81
