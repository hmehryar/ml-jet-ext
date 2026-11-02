"""T04-01: CPU-only, fail-closed experiment validation and immutable preparation.

No CUDA, torch, transformers, or dataset tensors are loaded here.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid

import yaml

REPO = Path(__file__).resolve().parents[1]
COUNTS = {'train': 11200, 'val': 1400, 'test': 1400}
STRATEGIES = {'S1', 'S2', 'S3', 'S4'}
LOSSES = {'cross_entropy', 'ce_ordinal', 'ce_soft_mae'}
PRECISIONS = {'fp32', 'bf16', 'fp16'}
SAFE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]*$')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _positive(value, name, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not (value >= 0 if allow_zero else value > 0):
        raise ValueError(f'{name} must be {"nonnegative" if allow_zero else "positive"}')
    return value


@dataclass(frozen=True)
class T04Config:
    model: str
    checkpoint: str
    checkpoint_revision: str
    strategy: str
    moments: list
    moment_loss: dict
    precision: str
    learning_rates: dict
    seed: int
    epochs: int
    min_epochs: int
    patience: int
    scheduler: dict
    batch_size: int
    gradient_accumulation: int
    num_workers: int
    dataset_root: str
    output_root: str
    gpu_profile: str
    source_audit_report: str
    fold: int | None = None
    split_protocol: str = 'source_disjoint'
    optimizer: dict | None = None


def parse_config(path: Path) -> T04Config:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError('Expected YAML mapping')
    required = set(T04Config.__dataclass_fields__) - {'fold', 'split_protocol', 'optimizer'}
    missing = required - raw.keys()
    unknown = raw.keys() - set(T04Config.__dataclass_fields__)
    if missing or unknown:
        raise ValueError(f'Missing fields {sorted(missing)}; unknown fields {sorted(unknown)}')
    c = T04Config(**raw)
    if c.split_protocol not in ('aggregate_disjoint', 'source_disjoint'):
        raise ValueError('split_protocol must be aggregate_disjoint or source_disjoint')
    if c.model not in ('clip', 'siglip') or c.strategy not in STRATEGIES:
        raise ValueError('Invalid model or adaptation strategy')
    if not isinstance(c.checkpoint, str) or not c.checkpoint.strip() or not isinstance(c.checkpoint_revision, str) or not c.checkpoint_revision.strip():
        raise ValueError('Checkpoint and pinned revision required')
    if not re.fullmatch(r'[0-9a-fA-F]{40}', c.checkpoint_revision):
        raise ValueError('Use a 40-character immutable checkpoint commit SHA')
    if not isinstance(c.moments, list) or c.moments not in [list(range(1, k + 1)) for k in range(1, 6)]:
        raise ValueError('T04 requires cumulative moments E1-E5')
    if not isinstance(c.moment_loss, dict) or c.moment_loss.get('type') not in LOSSES:
        raise ValueError('Unsupported moment loss')
    if set(c.moment_loss) - {'type', 'ordinal_weight', 'mae_weight'}:
        raise ValueError('Unknown loss settings')
    if c.moment_loss['type'] == 'ce_ordinal':
        _positive(c.moment_loss.get('ordinal_weight'), 'ordinal_weight', allow_zero=True)
    if c.moment_loss['type'] == 'ce_soft_mae':
        _positive(c.moment_loss.get('mae_weight'), 'mae_weight', allow_zero=True)
    if c.precision not in PRECISIONS:
        raise ValueError('Invalid precision')
    if not isinstance(c.learning_rates, dict) or set(c.learning_rates) != {'backbone', 'adapter', 'heads'}:
        raise ValueError('learning_rates must define backbone, adapter and heads')
    for key, value in c.learning_rates.items():
        _positive(value, f'learning_rates.{key}')
    if c.optimizer is not None:
        if not isinstance(c.optimizer, dict) or set(c.optimizer) != {'type', 'weight_decay'}:
            raise ValueError('optimizer must contain exactly type and weight_decay')
        if c.optimizer['type'] not in ('adam', 'adamw'):
            raise ValueError('optimizer.type must be adam or adamw')
        import math
        decay = c.optimizer['weight_decay']
        if isinstance(decay, bool) or not isinstance(decay, (int, float)) or not math.isfinite(decay) or decay < 0:
            raise ValueError('optimizer.weight_decay must be finite and nonnegative')
        if c.optimizer['type'] == 'adam' and decay != 0:
            raise ValueError('Legacy Adam requires weight_decay=0')
    for key in ('seed', 'epochs', 'min_epochs', 'patience', 'batch_size', 'gradient_accumulation', 'num_workers'):
        value = getattr(c, key)
        if type(value) is not int or value < (0 if key in ('seed', 'num_workers') else 1):
            raise ValueError(f'{key} must be an integer in range')
    if c.min_epochs > c.epochs:
        raise ValueError('min_epochs exceeds epochs')
    if c.fold is not None and (type(c.fold) is not int or c.fold not in range(5)):
        raise ValueError('fold must be null or 0..4')
    if not isinstance(c.scheduler, dict) or c.scheduler.get('type') != 'ReduceLROnPlateau':
        raise ValueError('T04-01 requires an explicit ReduceLROnPlateau scheduler')
    if c.scheduler.get('monitor') not in ('val_loss', 'val_joint_accuracy'):
        raise ValueError('Invalid scheduler monitor')
    if c.scheduler.get('mode') != ('min' if c.scheduler['monitor'] == 'val_loss' else 'max'):
        raise ValueError('Scheduler mode conflicts with monitor')
    for name in ('dataset_root', 'output_root', 'gpu_profile', 'source_audit_report'):
        if not isinstance(getattr(c, name), str) or not getattr(c, name).strip():
            raise ValueError(f'{name} required')
    return c


def _file_digest(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return {'path': str(path.resolve()), 'sha256': sha256(path), 'bytes': path.stat().st_size}


def inspect_inputs(c):
    root = Path(c.dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    paths = {'train': root / 'splits/train.csv', 'val': root / 'splits/val.csv', 'test': root / 'splits/test.csv'}
    counts = {}
    sets = {}
    records = {}
    for split, path in paths.items():
        records[split] = _file_digest(path)
        with path.open(newline='') as f:
            reader = csv.DictReader(f)
            if not {'agg_file_path', 'alpha', 'q0', 'config'} <= set(reader.fieldnames or []):
                raise ValueError(f'{path} missing dataset columns')
            ids = set()
            for row in reader:
                rel = Path(row['agg_file_path'])
                if rel.is_absolute() or '..' in rel.parts or not rel.parts or rel.parts[0] != row['config']:
                    raise ValueError(f'Unsafe aggregate path {rel}')
                if str(rel) in ids:
                    raise ValueError(f'Duplicate aggregate {rel}')
                ids.add(str(rel))
            counts[split] = len(ids)
            sets[split] = ids
        if counts[split] != COUNTS[split]:
            raise ValueError(f'{split} has {counts[split]}, expected {COUNTS[split]}')
    if any(sets[a] & sets[b] for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))):
        raise ValueError('Cross-split aggregate overlap')
    manifest = root / 'splits/split_manifest.json'
    records['split_manifest'] = _file_digest(manifest)
    published = json.loads(manifest.read_text())
    hashes = published.get('split_file_sha256')
    if not isinstance(hashes, dict):
        raise ValueError('Published split manifest lacks split_file_sha256')
    for name in COUNTS:
        expected = hashes.get(paths[name].name)
        if expected != records[name]['sha256']:
            raise ValueError(f'Published split hash mismatch for {name}')
    audit = Path(c.source_audit_report).expanduser().resolve()
    records['source_audit'] = _file_digest(audit)
    report = json.loads(audit.read_text())
    # Source membership is reported, not presumed independent from distinct
    # aggregate file paths. Both protocols demand actual hash-bound evidence.
    audited_hashes = report.get('split_sha256')
    if not isinstance(audited_hashes, dict) or any(
            audited_hashes.get(name + '.csv') != records[name]['sha256']
            for name in COUNTS):
        raise ValueError('Audit does not match current split CSV hashes')
    if report.get('published_split_manifest_sha256') not in (None, records['split_manifest']['sha256']):
        raise ValueError('Audit does not match published split manifest hash')
    if not isinstance(report.get('source_id_method'), str) or not report['source_id_method'].strip():
        raise ValueError('Audit missing source-ID methodology')
    overlap = report.get('global_cross_split_overlap')
    per_config = report.get('per_configuration_cross_split_overlap')
    configurations = set(published.get('configurations', {}))
    if type(overlap) is not int or overlap < 0:
        raise ValueError('Audit missing valid global overlap count')
    if not isinstance(per_config, dict) or len(per_config) != 35 or any(
            type(v) is not int or v < 0 for v in per_config.values()
    ) or (configurations and set(per_config) != configurations):
        raise ValueError('Audit missing valid per-configuration counts')
    if sum(per_config.values()) != overlap:
        raise ValueError('Audit global/per-configuration overlap mismatch')
    if c.split_protocol == 'source_disjoint' and overlap != 0:
        raise ValueError('Source-disjoint protocol requires zero cross-split overlap')
    return {
        'dataset_root': str(root), 'counts': counts, 'input_records': records,
        'split_protocol': c.split_protocol,
        'global_cross_split_overlap': overlap,
        'per_configuration_cross_split_overlap': per_config,
        'source_audit_status': ('verified_zero' if overlap == 0 else 'verified_overlap_disclosed'),
        'source_overlap_independence_claim': overlap == 0,
    }


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_info():
    def read(*args):
        try:
            return subprocess.check_output(['git', '-C', str(REPO), *args], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {'commit': read('rev-parse', 'HEAD'), 'status': read('status', '--porcelain'), 'diff': read('diff', '--binary', 'HEAD')}


def preflight(config_path):
    cfg = parse_config(config_path)
    inputs = inspect_inputs(cfg)
    return cfg, inputs


def prepare(config_path, run_id=None):
    cfg, inputs = preflight(config_path)
    run_id = run_id or (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:12])
    if not SAFE.fullmatch(run_id):
        raise ValueError('Invalid run ID')
    out = (Path(cfg.output_root).expanduser().resolve() / cfg.model / cfg.strategy /
           ('E' + str(len(cfg.moments))) / ('seed-' + str(cfg.seed)) / run_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=False)  # atomic reservation, even if destination is empty
    git = git_info()
    try:
        source = out / 'source'
        source.mkdir()
        for folder in ('models', 'data', 'train_utils'):
            for path in (REPO / folder).glob('*.py'):
                dest = source / path.relative_to(REPO)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
        for path in (REPO / 'config.py', REPO / 'train_t04.ipynb'):
            if path.is_file():
                dest = source / path.relative_to(REPO)
                shutil.copy2(path, dest)
        (source / 'config.original.yml').write_bytes(Path(config_path).read_bytes())
        for folder in ('models', 'data', 'train_utils'):
            (source / folder).mkdir(exist_ok=True)
            (source / folder / '__init__.py').touch(exist_ok=True)
        hashes = {str(p.relative_to(source)): sha256(p) for p in sorted(source.rglob('*')) if p.is_file()}
        (out / 'source_manifest.json').write_text(json.dumps(hashes, indent=2))
        (out / 'config.resolved.json').write_text(json.dumps(asdict(cfg), indent=2))
        manifest = {'schema_version': 2, 'state': 'prepared', 'created_utc': datetime.now(timezone.utc).isoformat(),
                    'run_id': run_id, 'configuration': asdict(cfg), 'inputs': inputs,
                    'git': {'commit': git['commit'], 'status': git['status']},
                    'source_hashes': hashes, 'python': sys.version,
                    'packages': {name: _package_version(name) for name in ('torch', 'transformers', 'numpy', 'pandas', 'PyYAML')},
                    'runtime_gpu_details': 'capture on allocated compute node in T04-04',
                    'trainable_parameters': 'capture after construction in T04-03',
                    'test_set_evaluated': False}
        manifest['configuration_sha256'] = sha256(out / 'config.resolved.json')
        (out / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
        (out / 'working_tree.patch').write_text(git['diff'] or '')
        for p in source.rglob('*'):
            p.chmod(0o555 if p.is_dir() else 0o444)
        source.chmod(0o555)
        return out
    except Exception:
        # Retain failed reservation for forensic inspection; never silently delete it.
        (out / 'PREPARATION_FAILED').write_text('Preparation failed; do not reuse this run ID.\n')
        raise


def verify(out):
    out = Path(out)
    manifest = json.loads((out / 'run_manifest.json').read_text())
    source = out / 'source'
    if sha256(out / 'config.resolved.json') != manifest['configuration_sha256']:
        raise RuntimeError('Resolved config altered')
    actual = {str(p.relative_to(source)): sha256(p) for p in source.rglob('*') if p.is_file()}
    if actual != manifest['source_hashes']:
        raise RuntimeError('Source snapshot altered')
    for record in manifest['inputs']['input_records'].values():
        if sha256(Path(record['path'])) != record['sha256']:
            raise RuntimeError(f'Input changed: {record["path"]}')
    return True


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path)
    p.add_argument('--check', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--run-id')
    p.add_argument('--verify', type=Path)
    args = p.parse_args(argv)
    if args.verify:
        verify(args.verify)
        print('T04 manifest verification: PASS')
        return
    if not args.config or args.check == args.prepare_only:
        p.error('Provide --config and exactly one of --check / --prepare-only')
    if args.check:
        cfg, input_metadata = preflight(args.config)
        print('T04 preflight: PASS', cfg.model, cfg.strategy, cfg.moments, input_metadata['counts'])
    else:
        print('Prepared:', prepare(args.config, args.run_id))


if __name__ == '__main__':
    main()
