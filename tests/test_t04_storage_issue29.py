"""Small CPU-only tests for T04 research storage roots, no HPC access required."""
from pathlib import Path
import pytest

from train_utils.t04_storage import load_storage, future_output_root, slurm_log_options, historical_run_path


def config_file(tmp_path):
    root = tmp_path / 'ceph'
    cfg = tmp_path / 'storage.yml'
    cfg.write_text('schema_version: 1\nstorage:\n'
                   f'  research_root: {root}\n'
                   f'  output_root: {root}/training_output\n'
                   f'  slurm_log_root: {root}/logs\n'
                   f'  historical_t04_06_v2: {root}/training_output/T04/t04_06_v2_production\n')
    return cfg


def test_new_experiment_output_is_absolute_and_research_based(tmp_path):
    cfg = load_storage(config_file(tmp_path))
    path = future_output_root('new-test', cfg)
    assert path.is_absolute() and path.is_dir()
    assert path.is_relative_to(cfg.research_root)


def test_slurm_log_options(tmp_path):
    cfg = load_storage(config_file(tmp_path))
    opts = slurm_log_options('clip_s4_e5_seed42', pool='a100', storage=cfg)
    assert opts[0].startswith('--output=') and opts[1].startswith('--error=')
    assert all('/logs/T04/a100/clip_s4_e5_seed42/' in s for s in opts)
    assert (cfg.slurm_log_root/'T04/a100/clip_s4_e5_seed42').is_dir()


def test_reject_unsafe_identifiers(tmp_path):
    cfg = load_storage(config_file(tmp_path))
    with pytest.raises(ValueError):
        slurm_log_options('../escape', pool='a100', storage=cfg)
    with pytest.raises(ValueError):
        slurm_log_options('ok', pool='other', storage=cfg)
    with pytest.raises(ValueError):
        future_output_root('../escape', storage=cfg)


def test_historical_path_resolution_preserves_original(tmp_path):
    cfg = load_storage(config_file(tmp_path))
    repo = tmp_path / 'repo'
    old = repo / 'training_output/T04/t04_06_v2_production/clip/S4/E5/seed-42/run-1'
    resolved = historical_run_path(str(old), repo=repo, storage=cfg)
    assert resolved == cfg.historical_t04_06_v2/'clip/S4/E5/seed-42/run-1'
    with pytest.raises(ValueError):
        historical_run_path('/random/path', repo=repo, storage=cfg)
