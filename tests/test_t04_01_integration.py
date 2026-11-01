"""Issue T04-01 regression gates; no GPU/network/real dataset required."""
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pytest
import yaml
from tests.test_t04_preflight import fixture
from train_utils import t04_preflight as p


def test_config_schema_rejects_unknown_field(tmp_path, monkeypatch):
    config, settings, _, _ = fixture(tmp_path, monkeypatch)
    settings['unknown_unsafe_option'] = True
    config.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError, match='unknown'):
        p.parse_config(config)


def test_checkpoint_revision_must_be_sha(tmp_path, monkeypatch):
    config, settings, _, _ = fixture(tmp_path, monkeypatch)
    settings['checkpoint_revision'] = 'main'
    config.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError, match='40-character'):
        p.parse_config(config)


def test_no_snapshot_tampering(tmp_path, monkeypatch):
    config, _, _, _ = fixture(tmp_path, monkeypatch)
    out = p.prepare(config, 'tamper')
    assert p.verify(out)
    file = out / 'source' / 'config.original.yml'
    file.chmod(0o644)
    file.write_bytes(file.read_bytes() + b'\n# changed\n')
    with pytest.raises(RuntimeError, match='Source snapshot altered'):
        p.verify(out)


def test_resolved_config_tampering(tmp_path, monkeypatch):
    config, _, _, _ = fixture(tmp_path, monkeypatch)
    out = p.prepare(config, 'tamper-config')
    file = out / 'config.resolved.json'
    file.write_text(file.read_text() + ' ')
    with pytest.raises(RuntimeError, match='Resolved config altered'):
        p.verify(out)


def test_run_id_collision_is_atomic(tmp_path, monkeypatch):
    config, _, _, _ = fixture(tmp_path, monkeypatch)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: _reserve(config), range(2)))
    assert sorted(results) == [False, True]


def _reserve(config):
    try:
        p.prepare(config, 'same-id')
        return True
    except FileExistsError:
        return False


def test_snapshot_contains_execution_modules_when_repo_has_them(tmp_path, monkeypatch):
    # Standalone stage-one fixture has only the core module. Real repo must
    # snapshot both before production run can be attempted.
    config, _, _, _ = fixture(tmp_path, monkeypatch)
    out = p.prepare(config, 'snapshot')
    manifest = json.loads((out / 'run_manifest.json').read_text())
    assert manifest['configuration_sha256'] == p.sha256(out / 'config.resolved.json')
    assert manifest['test_set_evaluated'] is False
    assert 'source_audit' in manifest['inputs']['input_records']
    assert 'train' in manifest['inputs']['input_records']
    assert (out / 'source/train_utils/t04_preflight.py').is_file()
