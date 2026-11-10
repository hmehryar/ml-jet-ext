"""T04-10 immutable continuation protocol (no fixed maximum epoch).

Existing 80-epoch snapshots and original configs are never edited. A separate,
source-hashed protocol snapshot executes append-only sessions in the original run.
Only the new session/global history/checkpoint pointer are written on execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

BASE_COMMIT = 'c866804e38eef508e83297c111ab4318278eeb7e'
BRANCH = 'feat/t04-10-single-head-pilot'
ROOT = Path('/rs/rs_grp_mljet/ml-jet-ext/training_output/T04/t04_10_single_head')
MATRIX = {('clip', 'alpha_only'): 'E3', ('clip', 'q0_only'): 'E4',
          ('siglip', 'alpha_only'): 'E2', ('siglip', 'q0_only'): 'E4'}
SEEDS = (42, 123, 2026)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for piece in iter(lambda: f.read(1024 * 1024), b''):
            h.update(piece)
    return h.hexdigest()


def source_hashes(source):
    source = Path(source)
    return {str(p.relative_to(source)): digest(p) for p in sorted(source.rglob('*.py'))
            if '__pycache__' not in p.parts}


def validate_target(target):
    if type(target) is not int or target < 1:
        raise ValueError('Target must be a positive integer number of TOTAL epochs')
    return target


def current_git(repo):
    repo = Path(repo).resolve()
    head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    branch = subprocess.check_output(['git', '-C', str(repo), 'branch', '--show-current'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    if branch != BRANCH or dirty:
        raise ValueError('Require clean tracked T04-10 feature branch before freezing continuation protocol')
    base = subprocess.run(['git', '-C', str(repo), 'merge-base', '--is-ancestor', BASE_COMMIT, head], check=False)
    if base.returncode:
        raise ValueError('Stage 6 base commit missing from checkout history')
    return head


def prepare_protocol(repo, root=ROOT):
    repo, root = Path(repo).resolve(), Path(root).resolve()
    commit = current_git(repo)
    key = 'continuation-' + commit[:12]
    folder = root / '_continuation_protocols' / key
    if folder.exists():
        verify_protocol(folder)
        if json.loads((folder/'protocol_manifest.json').read_text())['source_commit'] != commit:
            raise ValueError('Existing protocol uses another commit')
        return folder
    folder.parent.mkdir(parents=True, exist_ok=True)
    candidate = Path(tempfile.mkdtemp(prefix='.build-continuation-', dir=folder.parent))
    try:
        source = candidate / 'source'
        for sub in ('train_utils', 'models', 'data'):
            for src in sorted((repo / sub).glob('*.py')):
                dst = source / sub / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            (source / sub / '__init__.py').touch(exist_ok=True)
        hashes = source_hashes(source)
        manifest = dict(schema=1, name='t04_10_unbounded_continuation', source_commit=commit,
                        baseline_commit=BASE_COMMIT, source_sha256=hashes,
                        target_policy='positive_integer_per_session',
                        parent_policy='verified_latest_checkpoint')
        (candidate / 'protocol_manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
        os.replace(candidate, folder)
    except BaseException:
        shutil.rmtree(candidate, ignore_errors=True)
        raise
    verify_protocol(folder)
    return folder


def verify_protocol(protocol):
    protocol = Path(protocol).resolve()
    record = json.loads((protocol/'protocol_manifest.json').read_text())
    if record.get('schema') != 1 or record.get('name') != 't04_10_unbounded_continuation':
        raise ValueError('Unsupported continuation protocol')
    actual = source_hashes(protocol/'source')
    if actual != record['source_sha256']:
        raise ValueError('Continuation protocol source SHA-256 mismatch')
    if 'train_utils/t04_10_extension.py' not in actual:
        raise ValueError('Missing extension runner in frozen protocol')
    return record


def expected_runs(root=ROOT):
    root = Path(root).resolve()
    for (model, task), moment in MATRIX.items():
        for seed in SEEDS:
            runid = f't04_10_{model}_{task}_seed{seed}_session_v1'
            yield root/model/task/moment/f'seed-{seed}'/runid


def verify_run(run, target, *, root=ROOT):
    """Read-only preflight; calls original run's SHA verifier and latest checkpoint verifier."""
    target = validate_target(target)
    root, run = Path(root).resolve(), Path(run).resolve()
    if run not in set(expected_runs(root)):
        raise ValueError('Run outside frozen 12-run matrix')
    from train_utils.t04_10_preflight import verify
    from train_utils.t04_10_sessions import latest_checkpoint
    verify(run)
    manifest = json.loads((run/'run_manifest.json').read_text())
    config = json.loads((run/'config.resolved.json').read_text())
    if manifest['git_commit'] != BASE_COMMIT or config['epochs'] != 80:
        raise ValueError('Expected audited Stage 6 baseline, not a different experiment')
    if manifest.get('test_set_evaluated') is not False:
        raise ValueError('Test set was used')
    pointer = latest_checkpoint(run)
    if pointer is None or pointer['epoch'] < 80:
        raise ValueError('Continuation requires completed 80-epoch baseline checkpoint')
    if target <= pointer['epoch']:
        raise ValueError(f'Target {target} must exceed last committed epoch {pointer["epoch"]}')
    return run, config, pointer


def execute(run, protocol, target, root=ROOT):
    """Execute from frozen protocol source, never from mutable working-tree modules."""
    protocol = Path(protocol).resolve()
    here = Path(__file__).resolve()
    if here != protocol / 'source/train_utils/t04_10_extension.py':
        raise RuntimeError('Execution must use frozen continuation protocol source')
    record = verify_protocol(protocol)
    run, cfg, _pointer = verify_run(run, target, root=root)
    import torch
    from train_utils.t04_10_runner import build_run
    from train_utils.t04_10_session_training import train_session, _recover_last_session
    from train_utils.t04_10_sessions import experiment_lock, latest_checkpoint, start_session, finish_session, _atomic_json
    if not torch.cuda.is_available():
        raise RuntimeError('Production continuation requires CUDA')
    device = torch.device('cuda')
    if cfg['precision'] == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA BF16 not supported')
    # Full execution lock covers parent selection, new session creation and training.
    with experiment_lock(run):
        _, cfg, parent = verify_run(run, target, root=root)
        _recover_last_session(run, parent)
        session = start_session(run, mode='resume', target_epochs=target)
        session_manifest = session / 'session_manifest.json'
        info = json.loads(session_manifest.read_text())
        info['continuation_protocol'] = dict(source_commit=record['source_commit'],
                                             manifest_sha256=digest(protocol/'protocol_manifest.json'),
                                             protocol_path=str(protocol))
        _atomic_json(session_manifest, info)
        print(f'T04_10_CONTINUATION session={session.name} parent_epoch={parent["epoch"]} target_epoch={target}', flush=True)
        try:
            train, val, model, optimizer, scheduler = build_run(cfg, device)
            def seed_epoch(epoch):
                if train.generator is None:
                    raise RuntimeError('Deterministic training generator missing')
                train.generator.manual_seed(int(cfg['seed']) + epoch)
            outcome = train_session(run=run, session=session, mode='resume', model=model,
                       train_loader=train, val_loader=val, optimizer=optimizer,
                       scheduler=scheduler, task=cfg['task'],
                       config_sha256=json.loads((run/'run_manifest.json').read_text())['config_sha256'],
                       epochs=target, device=device, precision=cfg['precision'],
                       reset_loader_seed=seed_epoch, microbatch_size=cfg['batch_size'])
        except BaseException:
            try:
                finish_session(run, session, status='interrupted')
            except Exception:
                pass
            raise
        print('T04-10 continuation session completed: ' + session.name + ' ' + str(outcome), flush=True)
        print('SESSION_COMPLETED=' + str(session), flush=True)
        return outcome


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare-protocol', action='store_true')
    p.add_argument('--repo')
    p.add_argument('--root', default=str(ROOT))
    p.add_argument('--protocol')
    p.add_argument('--run')
    p.add_argument('--target-epochs', type=int)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args(argv)
    if args.prepare_protocol:
        if not args.repo:
            p.error('--prepare-protocol requires --repo')
        print('CONTINUATION_PROTOCOL=' + str(prepare_protocol(args.repo, args.root)))
        return
    if not args.run or not args.protocol or args.target_epochs is None:
        p.error('Need --run, --protocol and --target-epochs')
    record = verify_protocol(args.protocol)
    run, cfg, pointer = verify_run(args.run, args.target_epochs, root=args.root)
    print(json.dumps(dict(status='CONTINUATION_GUARD_PASS', run=str(run),
          current_epoch=pointer['epoch'], target_epoch=args.target_epochs,
          model=cfg['model'], task=cfg['task'], seed=cfg['seed'],
          protocol_commit=record['source_commit']), sort_keys=True), flush=True)
    if args.execute:
        execute(run, args.protocol, args.target_epochs, root=args.root)
    else:
        print('PREVIEW_ONLY: no session created, no job submitted')

if __name__ == '__main__':
    main()
