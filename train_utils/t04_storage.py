"""T04 storage roots for NEW runs, Slurm logs, and read-only historical relocation.

Issue #29. Do not modify immutable historical run configs/registries.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
import tempfile

import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / 'config/t04_storage.yml'
SAFE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$')


@dataclass(frozen=True)
class T04Storage:
    research_root: Path
    output_root: Path
    slurm_log_root: Path
    historical_t04_06_v2: Path


def load_storage(config_path: Path = CONFIG) -> T04Storage:
    raw = yaml.safe_load(Path(config_path).read_text())
    if not isinstance(raw, dict) or set(raw) != {'schema_version', 'storage'} or raw['schema_version'] != 1:
        raise ValueError('Invalid storage configuration schema')
    data = raw['storage']
    keys = {'research_root', 'output_root', 'slurm_log_root', 'historical_t04_06_v2'}
    if not isinstance(data, dict) or set(data) != keys:
        raise ValueError('Unexpected storage configuration keys')
    paths = {}
    for key in keys:
        value = data[key]
        if not isinstance(value, str) or not value or not Path(value).is_absolute():
            raise ValueError(f'{key} must be a nonempty absolute path')
        if '..' in Path(value).parts:
            raise ValueError(f'{key} contains a parent traversal')
        paths[key] = Path(value).expanduser().resolve(strict=False)
    root = paths['research_root']
    for key in keys - {'research_root'}:
        if not paths[key].is_relative_to(root) or paths[key] == root:
            raise ValueError(f'{key} must be strictly under research_root')
    if len(set(paths.values())) != len(paths):
        raise ValueError('Storage paths must be distinct')
    return T04Storage(**paths)


def ensure_writable(path: Path) -> Path:
    """Fail closed if Ceph is unavailable; do not fall back to home."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    fd, probe = tempfile.mkstemp(prefix='.t04-probe-', dir=path)
    try:
        os.close(fd)
    finally:
        Path(probe).unlink(missing_ok=True)
    return path


def future_output_root(experiment: str, storage: T04Storage | None = None) -> Path:
    if not SAFE.fullmatch(experiment):
        raise ValueError('Unsafe experiment name')
    cfg = storage or load_storage()
    path = cfg.output_root / 'T04' / experiment
    ensure_writable(path)
    return path


def slurm_log_options(experiment: str, *, pool: str, storage: T04Storage | None = None) -> list[str]:
    """Return CLI overrides; --output/--error supersede #SBATCH headers."""
    if not SAFE.fullmatch(experiment) or pool not in ('a100', 'h200'):
        raise ValueError('Unsafe experiment id or GPU pool')
    cfg = storage or load_storage()
    directory = ensure_writable(cfg.slurm_log_root / 'T04' / pool / experiment)
    return [f'--output={directory}/%j.out', f'--error={directory}/%j.err']


def historical_run_path(recorded_run: str, *, repo: Path = REPO,
                        storage: T04Storage | None = None) -> Path:
    """Translate original run root to Ceph, without changing recorded paths/hashes.

    No symlink needed. The caller can read artifacts at this path.
    """
    cfg = storage or load_storage()
    old = Path(repo).absolute() / 'training_output/T04/t04_06_v2_production'
    run = Path(recorded_run)
    if not run.is_absolute() or '..' in run.parts:
        raise ValueError('Invalid historical run path')
    try:
        suffix = run.relative_to(old)
    except ValueError as exc:
        raise ValueError('Run path is outside the known historical root') from exc
    if not suffix.parts:
        raise ValueError('Expected an experiment run, not the root')
    return cfg.historical_t04_06_v2 / suffix

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='T04 Ceph research storage preflight')
    parser.add_argument('command', choices=['check','prepare'])
    args = parser.parse_args()
    cfg = load_storage()
    print('OUTPUT_ROOT', cfg.output_root)
    print('SLURM_LOG_ROOT', cfg.slurm_log_root)
    print('HISTORICAL_V2_ROOT', cfg.historical_t04_06_v2)
    if args.command == 'prepare':
        ensure_writable(cfg.output_root)
        for pool in ('a100', 'h200'):
            ensure_writable(cfg.slurm_log_root / 'T04' / pool)
        print('PASS: writable Ceph storage roots')
