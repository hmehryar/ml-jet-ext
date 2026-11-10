"""T04-10: fail-closed, CPU-only preparation of isolated single-head runs.

No import of torch, datasets, or T04-07 training modules. No jobs submitted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone

import yaml

TASKS = ('alpha_only', 'q0_only')
MODELS = ('clip', 'siglip')  # ViT excluded until independently validated parity protocol.
STRATEGIES = ('S1', 'S2', 'S3', 'S4')
SEEDS = (42, 123, 2026)
SAFE_ID = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_.-]*$')
PINNED_REVISION = re.compile(r'^[0-9a-f]{40}$')
SPLITS = ('train.csv', 'val.csv', 'test.csv')
SOURCE_DIRS = ('models', 'train_utils', 'data')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _fail(condition, message):
    if not condition:
        raise ValueError(message)


def load_settings(config_path):
    data = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    _fail(type(data) is dict, 'Configuration must be a YAML mapping')
    required = {
        'model', 'task', 'checkpoint', 'checkpoint_revision', 'strategy',
        'moments', 'seed', 'epochs', 'precision', 'optimizer', 'learning_rates',
        'batch_size', 'num_workers', 'dataset_root', 'output_root',
        'source_audit_report', 'split_protocol', 'training_protocol',
        'gpu_profile'
    }
    optional = {'fold', 'scheduler'}
    _fail(set(data) <= required | optional and required <= set(data),
          f'Config fields: missing={sorted(required - set(data))}; unexpected={sorted(set(data) - required - optional)}')
    _fail(data['model'] in MODELS, 'Stage 2 only supports CLIP/SigLIP; ViT parity is a separate task')
    _fail(data['task'] in TASKS, 'Select alpha_only or q0_only')
    _fail(data['strategy'] in STRATEGIES, 'Unknown adaptation strategy')
    _fail(data['seed'] in SEEDS and type(data['seed']) is int, 'Unrecognized pilot seed')
    _fail(data['training_protocol'] == 'fixed_budget_v2' and type(data['epochs']) is int
          and data['epochs'] == 80, 'Require the T04-06 v2 80-epoch fixed-budget protocol')
    _fail(data['precision'] in ('fp32', 'bf16'), 'Only fp32/bf16 supported by T04-10 stage 2')
    _fail(type(data['moments']) is list and data['moments'] in
          [list(range(1, k + 1)) for k in range(1, 6)], 'Require E1-E5 cumulative moments')
    _fail(type(data['checkpoint']) is str and data['checkpoint'].strip(), 'Missing checkpoint')
    _fail(type(data['checkpoint_revision']) is str and
          bool(PINNED_REVISION.fullmatch(data['checkpoint_revision'])), 'Require pinned 40-hex revision')
    expected = ('openai/clip-vit-base-patch16' if data['model'] == 'clip'
                else 'google/siglip-base-patch16-224')
    _fail(data['checkpoint'] == expected, 'Unrecognized architecture checkpoint')
    _fail(data['optimizer'] == {'type': 'adamw', 'weight_decay': 0.01},
          'Pilot requires AdamW weight decay 0.01')
    _fail(type(data['learning_rates']) is dict and
          set(data['learning_rates']) == {'backbone', 'adapter', 'heads'} and
          all(type(v) in (float, int) and 0 < v < 1 for v in data['learning_rates'].values()),
          'Positive parameter-group learning rates required')
    for name in ('batch_size', 'num_workers'):
        minimum = 0 if name == 'num_workers' else 1
        _fail(type(data[name]) is int and data[name] >= minimum, f'Invalid {name}')
    _fail(data.get('fold') is None, 'Initial pilot uses the fixed split, not a CV fold')
    _fail(data['split_protocol'] in ('aggregate_disjoint', 'source_disjoint'), 'Invalid split protocol')
    _fail(type(data['gpu_profile']) is str and data['gpu_profile'].strip(), 'Missing GPU profile')
    sch = data.get('scheduler', {'type':'ReduceLROnPlateau', 'factor':0.5, 'patience':4, 'min_lr':1e-8})
    _fail(type(sch) is dict and set(sch) == {'type', 'factor', 'patience', 'min_lr'}
          and sch['type'] == 'ReduceLROnPlateau' and type(sch['patience']) is int
          and sch['patience'] >= 1 and 0 < sch['factor'] < 1 and sch['min_lr'] >= 0,
          'Expected validation-loss ReduceLROnPlateau scheduler')
    data['scheduler'] = sch
    for field in ('dataset_root', 'output_root', 'source_audit_report'):
        _fail(type(data[field]) is str and data[field].strip(), f'{field} required')
    _fail('t04_10_single_head' in Path(data['output_root']).parts,
          'Output must be in a dedicated t04_10_single_head namespace')
    return data


def inspect_inputs(data):
    root = Path(data['dataset_root']).expanduser().resolve()
    _fail(root.is_dir(), f'Missing dataset root: {root}')
    split_dir = root / 'splits'
    records, paths = {}, {}
    for name in SPLITS:
        csv_path = split_dir / name
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        with csv_path.open(newline='') as handle:
            reader = csv.DictReader(handle)
            _fail({'agg_file_path', 'alpha', 'q0', 'config'} <= set(reader.fieldnames or ()),
                  f'{name}: missing required columns')
            members = set()
            for row in reader:
                p = Path(row['agg_file_path'])
                _fail(bool(p.parts) and not p.is_absolute() and '..' not in p.parts and
                      p.parts[0] == row['config'], f'{name}: unsafe aggregate path')
                _fail(str(p) not in members, f'{name}: duplicate aggregate path')
                members.add(str(p))
            _fail(bool(members), f'{name}: empty split')
        paths[name] = members
        records[name] = {'path': str(csv_path), 'sha256': sha256(csv_path), 'count': len(members)}
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i+1:]:
            _fail(not paths[left] & paths[right], f'Aggregate overlap: {left}/{right}')
    manifest = split_dir / 'split_manifest.json'
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    published = json.loads(manifest.read_text())
    for name in SPLITS:
        _fail(published.get('split_file_sha256', {}).get(name) == records[name]['sha256'],
              f'Split manifest hash differs: {name}')
    records['split_manifest'] = {'path': str(manifest), 'sha256': sha256(manifest)}
    audit = Path(data['source_audit_report']).expanduser().resolve()
    if not audit.is_file():
        raise FileNotFoundError(audit)
    report = json.loads(audit.read_text())
    for name in SPLITS:
        _fail(report.get('split_sha256', {}).get(name) == records[name]['sha256'],
              f'Source audit does not match split hash: {name}')
    _fail(bool(report.get('source_id_method')), 'Audit lacks source-ID methodology')
    overlap = report.get('global_cross_split_overlap')
    _fail(type(overlap) is int and overlap >= 0, 'Audit must explicitly report source overlap')
    by_config = report.get('per_configuration_cross_split_overlap')
    _fail(type(by_config) is dict and bool(by_config) and
          all(type(v) is int and v >= 0 for v in by_config.values()) and
          sum(by_config.values()) == overlap,
          'Audit per-configuration source overlaps must sum to the global count')
    if data['split_protocol'] == 'source_disjoint':
        _fail(overlap == 0, 'Cannot claim source-disjoint split with source overlap')
    records['source_audit'] = {'path': str(audit), 'sha256': sha256(audit)}
    return {'inputs': records, 'source_overlap': overlap, 'split_protocol': data['split_protocol']}


def _git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def prepare(config_path, *, repo_root=None, run_id=None):
    cfg = load_settings(config_path)
    audited = inspect_inputs(cfg)
    repo = Path(repo_root or Path(__file__).resolve().parents[1]).resolve()
    _fail((repo / 'models/t04_10_single_head.py').is_file(), 'T04-10 Stage 1 missing')
    _fail((repo / 'train_utils/t04_10_runner.py').is_file(), 'T04-10 Stage 2 runner missing')
    branch = _git(repo, 'branch', '--show-current')
    _fail(branch == 'feat/t04-10-single-head-pilot', 'Prepare only from dedicated T04-10 branch')
    _fail(not _git(repo, 'status', '--porcelain', '--untracked-files=no'),
          'Tracked working tree contains uncommitted changes')
    commit = _git(repo, 'rev-parse', 'HEAD')
    run_id = run_id or f'{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:10]}'
    _fail(bool(SAFE_ID.fullmatch(run_id)), 'Unsafe run ID')
    root = Path(cfg['output_root']).expanduser().resolve()
    relative = Path(cfg['model']) / cfg['task'] / f'E{len(cfg["moments"])}' / f'seed-{cfg["seed"]}' / run_id
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=False)  # atomic output reservation
    try:
        source = destination / 'source'
        for folder in SOURCE_DIRS:
            for file in (repo / folder).glob('*.py'):
                path = source / file.relative_to(repo)
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, path)
            (source / folder / '__init__.py').touch(exist_ok=True)
        hashes = {str(p.relative_to(source)): sha256(p) for p in source.rglob('*') if p.is_file()}
        config_raw = (source / 'config.original.yml')
        config_raw.write_bytes(Path(config_path).read_bytes())
        hashes['config.original.yml'] = sha256(config_raw)
        (destination / 'config.resolved.json').write_text(json.dumps(cfg, sort_keys=True, indent=2))
        manifest = {
            'schema': 1, 'issue': 30, 'status': 'prepared', 'run_id': run_id,
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'git_commit': commit, 'git_branch': branch,
            'source_sha256': hashes, 'inputs': audited,
            'config_sha256': sha256(destination / 'config.resolved.json'),
            'test_set_evaluated': False,
        }
        (destination / 'run_manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2))
        return destination
    except Exception:
        (destination / 'PREPARATION_FAILED').write_text('Do not reuse this reservation.\n')
        raise


def verify(directory):
    run = Path(directory).resolve()
    manifest = json.loads((run / 'run_manifest.json').read_text())
    _fail(manifest['schema'] == 1 and manifest['issue'] == 30, 'Unexpected T04-10 manifest')
    _fail(sha256(run / 'config.resolved.json') == manifest['config_sha256'], 'Config modified')
    source = run / 'source'
    actual = {str(p.relative_to(source)): sha256(p) for p in source.rglob('*')
              if p.is_file() and not ('__pycache__' in p.parts and p.suffix == '.pyc')}
    _fail(actual == manifest['source_sha256'], 'Snapshot sources modified')
    for record in manifest['inputs']['inputs'].values():
        _fail(sha256(record['path']) == record['sha256'], f'Input modified: {record["path"]}')
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description='T04-10 source-audited CPU preflight')
    parser.add_argument('--config')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--verify')
    parser.add_argument('--run-id')
    args = parser.parse_args(argv)
    if args.verify:
        verify(args.verify)
        print('T04-10 verification: PASS')
        return
    if not args.config or args.check == args.prepare_only:
        parser.error('Provide --config and exactly one of --check/--prepare-only')
    if args.check:
        config = load_settings(args.config)
        audit = inspect_inputs(config)
        print('T04-10 preflight: PASS', config['model'], config['task'], audit['split_protocol'],
              'source overlap:', audit['source_overlap'])
    else:
        run = prepare(args.config, run_id=args.run_id)
        print('T04-10 prepared run:', run)
        print('No job submitted; use the held-by-default Slurm script after GPU validation.')


if __name__ == '__main__':
    main()
