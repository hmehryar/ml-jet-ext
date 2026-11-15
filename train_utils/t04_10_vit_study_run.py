"""Immutable 10-run ViT E1-E5 CE smoke-test execution; target 2 then resume.

Read-only verification for --check, GPU execution only with --execute.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from train_utils.t04_10_preflight import verify, sha256
from train_utils.loss_config import resolve_moment_loss
from train_utils.t04_10_sessions import latest_checkpoint

ALLOWED = {'cross_entropy', 'physical_mae', 'ce_physical_mae', 'ce_ordinal'}

def validate(run, target):
    run = Path(run).resolve()
    verify(run)
    cfg = json.loads((run/'config.resolved.json').read_text())
    base = json.loads((run/'run_manifest.json').read_text())
    lm_file = run/'vit_loss_manifest.json'
    lm = json.loads(lm_file.read_text())
    loss = lm['loss']
    if lm.get('schema') != 1 or loss.get('type') not in ALLOWED:
        raise ValueError('Invalid ViT loss specification')
    if resolve_moment_loss(loss) != lm['resolved_loss']:
        raise ValueError('Resolved loss differs from frozen manifest')
    if lm['run_id'] != base['run_id'] or lm['source_commit'] != base['git_commit']:
        raise ValueError('ViT run provenance mismatch')
    if lm['run_manifest_sha256'] != sha256(run/'run_manifest.json'):
        raise ValueError('Run manifest modified')
    if lm['config_sha256'] != base['config_sha256']:
        raise ValueError('Configuration fingerprint mismatch')
    if (cfg['model'] != 'vit' or cfg['task'] not in ('alpha_only','q0_only')
        or cfg['strategy'] != 'S4' or cfg['moments'] not in [list(range(1, k+1)) for k in range(1,6)]
        or cfg.get('vit_initialization') != 'random'
        or cfg['checkpoint'] != 'timm/vit_tiny_patch16_224_scratch'):
        raise ValueError('Incorrect ViT single-head scratch S4/E5 protocol')
    if lm.get('initialization') != 'random ViT-Tiny; no pretrained or transfer checkpoint':
        raise ValueError('ViT initialization provenance mismatch')
    if not isinstance(target,int) or isinstance(target,bool) or target < 1:
        raise ValueError('Target must be positive')
    parent = latest_checkpoint(run)
    if parent is not None and parent['epoch'] >= target:
        raise ValueError('Target does not exceed committed epoch')
    return cfg, lm, parent, sha256(lm_file)

def execute(run, target):
    import torch
    from torch import nn
    from train_utils.t04_10_runner import build_run
    from train_utils.t04_10_session_training import train_session, _recover_last_session
    from train_utils.t04_10_sessions import experiment_lock, start_session, finish_session, _atomic_json
    from train_utils.ordinal_loss import build_moment_criterion
    run = Path(run).resolve()
    if Path(__file__).resolve() != run/'source/train_utils/t04_10_vit_study_run.py':
        raise RuntimeError('Training must run from immutable prepared source snapshot')
    cfg, lm, parent, loss_hash = validate(run, target)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    if cfg['precision'] == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 not supported')
    with experiment_lock(run):
        cfg, lm, parent, loss_hash = validate(run, target)
        mode = 'resume' if parent else 'fresh'
        if parent:
            _recover_last_session(run, parent)
        session = start_session(run, mode=mode, target_epochs=target)
        info = json.loads((session/'session_manifest.json').read_text())
        info['vit_loss_manifest_sha256'] = loss_hash
        info['loss_spec'] = lm['resolved_loss']
        _atomic_json(session/'session_manifest.json', info)
        print(f'T04_10_VIT_START session={session.name} mode={mode} loss={lm["loss"]} parent_epoch={parent["epoch"] if parent else 0} target={target}',flush=True)
        try:
            device = torch.device('cuda')
            train, val, model, optimizer, scheduler = build_run(cfg, device)
            if lm['loss']['type'] == 'cross_entropy':
                criterion = nn.CrossEntropyLoss()
            else:
                key = 'alpha_output' if cfg['task']=='alpha_only' else 'q0_output'
                criterion = build_moment_criterion(lm['loss'], device=device)[key]
            def seed_epoch(epoch):
                if train.generator is None:
                    raise RuntimeError('Missing deterministic train loader generator')
                train.generator.manual_seed(int(cfg['seed'])+epoch)
            outcome = train_session(run=run,session=session,mode=mode,model=model,
                train_loader=train,val_loader=val,optimizer=optimizer,scheduler=scheduler,
                task=cfg['task'],config_sha256=json.loads((run/'run_manifest.json').read_text())['config_sha256'],
                epochs=target,device=device,precision=cfg['precision'],reset_loader_seed=seed_epoch,
                microbatch_size=cfg['batch_size'],criterion=criterion,loss_sha256=loss_hash)
        except BaseException:
            try:finish_session(run,session,status='interrupted')
            except Exception:pass
            raise
        print('T04_10_VIT_SESSION_COMPLETED',str(session),'epoch',outcome['epoch'],flush=True)
        return outcome

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True)
    p.add_argument('--target',type=int,required=True)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args(argv)
    cfg,lm,parent,h=validate(args.run,args.target)
    print(json.dumps({'status':'VIT_GUARD_PASS','run':str(Path(args.run).resolve()),
        'task':cfg['task'],'seed':cfg['seed'],'loss':lm['loss'],
        'epoch':parent['epoch'] if parent else 0,'target':args.target,
        'loss_sha256':h},sort_keys=True),flush=True)
    if args.execute:execute(args.run,args.target)
    else:print('PREVIEW_ONLY: no GPU training')

if __name__=='__main__':main()
