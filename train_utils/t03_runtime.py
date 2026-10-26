"""T03 configuration, exclusive run reservation and frozen source preparation.

Preparation uses no accelerator libraries and is safe for lightweight staging.
No dataset tensors or checkpoint objects are loaded here.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid


REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def archive_metadata(path, target):
    """Copy and hash large split CSVs in one pass over shared storage."""
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as source, target.open('xb') as dest:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            dest.write(block)
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('Split metadata changed while archiving: ' + str(path))
    return digest.hexdigest()


def verify_split_checksums(dataset_root, input_records, manifest_path=None):
    """Compare selected split hashes to the published manifest, without resplitting.

    Older datasets without a manifest remain supported. A present but incomplete
    or inconsistent manifest is an error, rather than a reason to skip checks.
    input_records contains the hashes already computed while archiving metadata.
    """
    split_root = Path(dataset_root).resolve() / 'splits'
    manifest_path = Path(manifest_path) if manifest_path else split_root / 'split_manifest.json'
    if not manifest_path.is_file():
        return {'status': 'not_available', 'verified_files': []}
    expected = json.loads(manifest_path.read_text()).get('split_file_sha256')
    if not isinstance(expected, dict):
        raise ValueError('Split manifest is missing split_file_sha256')
    verified = []
    for key in ('train_split', 'validation_split', 'test_split'):
        item = input_records[key]
        name = Path(item['path']).resolve().relative_to(split_root).as_posix()
        if name not in expected or item['sha256'] != expected[name]:
            raise ValueError(f'Published split checksum mismatch: {name}; do not regenerate splits')
        verified.append(name)
    return {'status': 'verified', 'verified_files': verified}


def resolve_scheduler(spec):
    resolved = dict(spec)
    if resolved.get('type', 'ReduceLROnPlateau') != 'ReduceLROnPlateau':
        raise ValueError('T03 requires ReduceLROnPlateau')
    resolved['type'] = 'ReduceLROnPlateau'
    resolved.setdefault('monitor', 'val_joint_accuracy')
    directions = {'val_joint_accuracy': 'max', 'val_loss': 'min'}
    if resolved['monitor'] not in directions:
        raise ValueError('Scheduler monitor must be val_joint_accuracy or val_loss')
    resolved.setdefault('mode', directions[resolved['monitor']])
    if resolved['mode'] != directions[resolved['monitor']]:
        raise ValueError('Scheduler monitor/direction mismatch; explicitly use '
                         '--scheduler-monitor val_loss --scheduler-mode min or '
                         '--scheduler-monitor val_joint_accuracy --scheduler-mode max')
    resolved.setdefault('factor', 0.5)
    resolved.setdefault('patience', 4)
    resolved.setdefault('min_lr', 0.0)
    return resolved


def reserve_run(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)  # atomic: even an empty preexisting run is occupied
    (path / 'reservation.json').write_text(json.dumps({
        'pid': os.getpid(), 'created_utc': datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    return path


def _git(repo):
    def read(*args):
        try:
            return subprocess.check_output(['git', '-C', str(repo), *args],
                                           text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {'commit': read('rev-parse', 'HEAD'),
            'status': read('status', '--porcelain', '--untracked-files=normal'),
            'diff': read('diff', '--binary', 'HEAD')}


def freeze_source(repo, out, raw_config, resolved):
    """Copy project code, export notebook, and seal its content by hash and mode."""
    from tools.notebook_to_py import convert
    source = out / 'source'
    source.mkdir()
    paths = [repo / 'config.py', repo / 'train_t03.ipynb']
    for folder in ('models', 'data', 'train_utils', 'tools', 'tests'):
        paths.extend(sorted((repo / folder).glob('*.py')))
    paths.extend(sorted((repo / 'slurm').glob('*')))
    originals = {path: sha256(path) for path in paths if path.is_file()}
    for path in paths:
        if not path.is_file():
            continue
        target = source / path.relative_to(repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    if any(sha256(path) != digest or sha256(source / path.relative_to(repo)) != digest
           for path, digest in originals.items()):
        raise RuntimeError('Source changed while snapshotting; prepare again from a stable checkout')
    # Resolve project packages here even if an unrelated installed distribution
    # happens to have the same top-level name.
    for folder in ('models', 'data', 'train_utils', 'tools', 'tests'):
        (source / folder / '__init__.py').touch()
    (source / 'config.original.yml').write_bytes(raw_config)
    (source / 'config.resolved.json').write_text(json.dumps(resolved, indent=2))
    convert(source / 'train_t03.ipynb', source / 'train_t03_run.py')
    hashes = {str(p.relative_to(source)): sha256(p)
              for p in sorted(source.rglob('*')) if p.is_file()}
    (out / 'source_manifest.json').write_text(json.dumps(hashes, indent=2))
    for path in source.rglob('*'):
        path.chmod(0o555 if path.is_dir() else 0o444)
    source.chmod(0o555)
    return hashes


def verify_source(out):
    out = Path(out)
    expected = json.loads((out / 'source_manifest.json').read_text())
    actual = {str(p.relative_to(out / 'source')): sha256(p)
              for p in (out / 'source').rglob('*') if p.is_file()}
    if actual != expected:
        raise RuntimeError('Frozen source hash verification failed')


def parser():
    p = argparse.ArgumentParser(description='Prepare or execute an isolated T03 run')
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--dataset-root', type=Path, help='Override MLJET_DATASET_ROOT and YAML dataset_root')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--output-root', type=Path, help='Override MLJET_OUTPUT_ROOT and YAML output_dir')
    p.add_argument('--workers', type=int)
    p.add_argument('--device', default='auto', help='auto (CUDA required), cpu, cuda, cuda:N')
    p.add_argument('--profile', choices=('local', 'hpc'), default='local')
    p.add_argument('--run-id')
    p.add_argument('--epochs', type=int, help='Explicit pilot epoch override')
    p.add_argument('--scheduler-monitor', choices=('val_joint_accuracy', 'val_loss'))
    p.add_argument('--scheduler-mode', choices=('min', 'max'))
    p.add_argument('--prepare-only', action='store_true', help='Freeze now; execute later on compute node')
    p.add_argument('--validation-suite', action='store_true',
                   help='Checkpoint transfer for C=1..5, then one full train/validation epoch')
    return p


def prepare(args, repo=REPO_ROOT):
    import yaml
    from config import get_config
    repo = Path(repo).resolve()
    raw = args.config.read_bytes()
    original = yaml.safe_load(raw)
    scheduler = dict(original.get('scheduler') or {})
    for field in ('monitor', 'mode'):
        override = getattr(args, 'scheduler_' + field)
        if override is not None:
            scheduler[field] = override
    scheduler = resolve_scheduler(scheduler)
    overrides = {'scheduler': scheduler}
    for key, value in [('dataset_root', args.dataset_root), ('output_dir', args.output_root),
                       ('num_workers', args.workers), ('epochs', args.epochs)]:
        if value is not None:
            overrides[key] = str(value.expanduser().resolve()) if isinstance(value, Path) else value
    cfg = get_config(str(args.config), overrides=overrides)
    if cfg.dataset_type != 'moment' or original.get('transfer_mode') != 'jetbench_full':
        raise ValueError('This runner requires moment / jetbench_full configuration')
    if cfg.moments != list(range(1, len(cfg.moments) + 1)):
        raise ValueError('T03 requires cumulative moments starting with M1')
    if cfg.num_workers < 0 or cfg.epochs < 1:
        raise ValueError('workers must be nonnegative and epochs positive')
    checkpoint = args.checkpoint or original.get('checkpoint_path')
    if not checkpoint:
        raise ValueError('Missing checkpoint_path / --checkpoint')
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    # Only metadata is read; held-out test tensors and labels are never evaluated.
    inputs = {'checkpoint': checkpoint, 'train_split': Path(cfg.train_csv),
              'validation_split': Path(cfg.val_csv), 'test_split': Path(cfg.test_csv)}
    manifest = Path(cfg.dataset_root_dir) / 'splits' / 'split_manifest.json'
    if manifest.is_file():
        inputs['split_manifest'] = manifest
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    name = str(cfg.model_tag)
    run_id = args.run_id or (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:12])
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', run_id):
        raise ValueError('model_tag and run-id must be safe single path components')
    root = Path(args.output_root or os.environ.get('MLJET_OUTPUT_ROOT')
                or original.get('output_dir', 'training_output'))
    out = root.expanduser().resolve() / args.profile / name / ('seed-' + str(cfg.seed)) / run_id
    reserve_run(out)
    archived_inputs = out / 'input_metadata'
    archived_inputs.mkdir()
    input_records = {}
    for key, path in inputs.items():
        digest = (sha256(path) if key == 'checkpoint' else
                  archive_metadata(path, archived_inputs / path.name))
        input_records[key] = {'path': str(path.resolve()), 'sha256': digest}
    split_verification = verify_split_checksums(
        cfg.dataset_root_dir, input_records,
        manifest_path=archived_inputs / 'split_manifest.json')
    cfg.output_dir = str(out)
    resolved = dict(vars(cfg), checkpoint_path=str(checkpoint), device=args.device,
                    profile=args.profile, run_id=run_id, transfer_mode='jetbench_full',
                    validation_suite=args.validation_suite)
    # Legacy config uses NaN for an absent normalization maximum. Moment data
    # never uses that field; record null so provenance remains standard JSON.
    if isinstance(resolved.get('global_max'), float) and not math.isfinite(resolved['global_max']):
        resolved['global_max'] = None
    if args.validation_suite:
        resolved['epochs'] = 1
    git = _git(repo)
    hashes = freeze_source(repo, out, raw, resolved)
    (out / 'config.original.yml').write_bytes(raw)
    (out / 'config.resolved.json').write_text(json.dumps(resolved, indent=2))
    (out / 'working_tree.patch').write_text(git.pop('diff') or '')
    provenance = {'schema_version': 1, 'status': 'prepared', 'git': git,
                  'created_utc': datetime.now(timezone.utc).isoformat(),
                  'resolved': resolved, 'source_hashes': hashes,
                  'inputs': input_records, 'split_verification': split_verification,
                  'selection': {'monitor': 'val_joint_accuracy', 'mode': 'max'},
                  'scheduler': scheduler, 'held_out_test_evaluated': False}
    (out / 'run_manifest.json').write_text(json.dumps(provenance, indent=2))
    print('Prepared run:', out, flush=True)
    return out


def execute(out):
    out = Path(out).resolve()
    verify_source(out)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg')
    env['PYTHONPATH'] = str(out / 'source')  # never import from the mutable checkout
    return subprocess.call([sys.executable, '-B', str(out / 'source' / 'train_t03_run.py'),
                            '--prepared-run', str(out)], cwd=out / 'source', env=env)


def main(argv=None):
    args = parser().parse_args(argv)
    out = prepare(args)
    if not args.prepare_only:
        raise SystemExit(execute(out))


if __name__ == '__main__':
    main()
