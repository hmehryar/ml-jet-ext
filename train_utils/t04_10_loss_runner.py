"""Immutable T04-10 single-head loss-ablation session executor.

New experiment identities; training ALWAYS starts from the original pretrained/scratch
initialization, never from a CE-trained checkpoint. Resume preserves loss identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from train_utils.loss_config import resolve_moment_loss
from train_utils.t04_10_preflight import verify

ALLOWED = {"physical_mae", "ce_ordinal", "ce_physical_mae"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def verify_ablation(run, target):
    from train_utils.t04_10_sessions import latest_checkpoint
    run = Path(run).resolve()
    verify(run)
    config = json.loads((run/'config.resolved.json').read_text())
    manifest = json.loads((run/'run_manifest.json').read_text())
    loss_path = run/'loss_ablation_manifest.json'
    payload = json.loads(loss_path.read_text())
    if payload.get('schema') != 1 or payload.get('loss',{}).get('type') not in ALLOWED:
        raise ValueError('Invalid loss experiment manifest')
    if resolve_moment_loss(payload['loss']) != payload['resolved_loss']:
        raise ValueError('Loss specification is not canonical')
    if payload.get('source_commit') != manifest.get('git_commit'):
        raise ValueError('Loss source commit mismatch')
    if payload.get('config_sha256') != manifest.get('config_sha256') or payload.get('run_manifest_sha256') != sha(run/'run_manifest.json'):
        raise ValueError('Loss run identity changed')
    if payload.get('run_id') != manifest.get('run_id'):
        raise ValueError('Loss run id mismatch')
    if config['model'] not in ('clip','siglip') or config['task'] not in ('alpha_only','q0_only'):
        raise ValueError('Unsupported single-head configuration')
    if type(target) is not int or target < 1:
        raise ValueError('Target must be a positive total epoch count')
    parent = latest_checkpoint(run)
    if parent and target <= parent['epoch']:
        raise ValueError('Target must exceed current committed epoch')
    return config, payload, parent, sha(loss_path)


def execute(run, target):
    import torch
    from train_utils.t04_10_runner import build_run
    from train_utils.t04_10_session_training import train_session, _recover_last_session
    from train_utils.t04_10_sessions import (experiment_lock, start_session,
                                              finish_session, latest_checkpoint, _atomic_json)
    from train_utils.ordinal_loss import build_moment_criterion
    run = Path(run).resolve()
    if Path(__file__).resolve() != run/'source/train_utils/t04_10_loss_runner.py':
        raise RuntimeError('Must execute from immutable experiment snapshot')
    cfg, payload, parent, loss_hash = verify_ablation(run, target)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required for production')
    device = torch.device('cuda')
    if cfg['precision'] == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 not supported')
    with experiment_lock(run):
        cfg, payload, parent, loss_hash = verify_ablation(run, target)
        mode = 'resume' if parent else 'fresh'
        if parent:
            _recover_last_session(run, parent)
        session = start_session(run, mode=mode, target_epochs=target)
        info = json.loads((session/'session_manifest.json').read_text())
        info['loss_ablation_manifest_sha256'] = loss_hash
        info['loss_spec'] = payload['resolved_loss']
        _atomic_json(session/'session_manifest.json', info)
        print(f'T04_10_LOSS_START mode={mode} session={session.name} loss={payload["loss"]} '
              f'parent_epoch={parent["epoch"] if parent else 0} target_epoch={target}', flush=True)
        try:
            train, val, model, optimizer, scheduler = build_run(cfg, device)
            key = 'alpha_output' if cfg['task']=='alpha_only' else 'q0_output'
            criterion = build_moment_criterion(payload['loss'], device=device)[key]
            def seed_epoch(epoch):
                if train.generator is None:
                    raise RuntimeError('Deterministic train loader generator missing')
                train.generator.manual_seed(int(cfg['seed'])+epoch)
            outcome = train_session(run=run, session=session, mode=mode,
                model=model, train_loader=train, val_loader=val,
                optimizer=optimizer, scheduler=scheduler, task=cfg['task'],
                config_sha256=json.loads((run/'run_manifest.json').read_text())['config_sha256'],
                epochs=target, device=device, precision=cfg['precision'],
                reset_loader_seed=seed_epoch, microbatch_size=cfg['batch_size'],
                criterion=criterion, loss_sha256=loss_hash)
        except BaseException:
            try: finish_session(run, session, status='interrupted')
            except Exception: pass
            raise
        print('T04_10_LOSS_SESSION_COMPLETED=' + str(session), flush=True)
        return outcome


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--target-epochs', type=int, required=True)
    p.add_argument('--execute', action='store_true')
    a=p.parse_args()
    cfg, payload, parent, loss_hash=verify_ablation(a.run, a.target_epochs)
    print(json.dumps({'status':'LOSS_GUARD_PASS','run':str(Path(a.run).resolve()),
        'model':cfg['model'],'task':cfg['task'],'seed':cfg['seed'],
        'loss':payload['loss'],'current_epoch':parent['epoch'] if parent else 0,
        'target_epoch':a.target_epochs,'loss_sha256':loss_hash}, sort_keys=True),flush=True)
    if a.execute:
        execute(a.run,a.target_epochs)
    else:
        print('PREVIEW_ONLY: no sessions or jobs created')

if __name__=='__main__': main()
