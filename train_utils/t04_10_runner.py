"""T04-10 standalone immutable-snapshot training runner (no T04-07 entrypoint edits)."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch

from data.moment_loader import get_moment_dataloaders
from models.t04_10_single_head import build_clip_siglip_single_head, build_trainable_optimizer
from train_utils.t04_10_checkpoint import fit_fixed_budget
from train_utils.t04_10_preflight import verify


def build_run(cfg, device):
    if cfg['model'] not in ('clip', 'siglip'):
        raise ValueError('ViT parity has not been established in T04-10')
    if cfg['task'] not in ('alpha_only', 'q0_only'):
        raise ValueError('Invalid single-head task')
    seed = cfg['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    train, val, _unused_test = get_moment_dataloaders(
        cfg['dataset_root'], moments=cfg['moments'], batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'], seed=seed, fold=None,
        pin_memory=(device.type == 'cuda'), validate_files=True,
        check_normalization=True)
    model = build_clip_siglip_single_head(
        architecture=cfg['model'], task=cfg['task'], channels=len(cfg['moments']),
        strategy=cfg['strategy'], checkpoint=cfg['checkpoint'],
        revision=cfg['checkpoint_revision'], pretrained=(cfg['strategy'] != 'S4'))
    model = model.to(device)
    lr = cfg['learning_rates']
    optimizer = build_trainable_optimizer(model, lr_backbone=lr['backbone'],
        lr_adapter=lr['adapter'], lr_head=lr['heads'],
        weight_decay=cfg['optimizer']['weight_decay'])
    sch = cfg['scheduler']
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=sch['factor'],
        patience=sch['patience'], min_lr=sch['min_lr'])
    return train, val, model, optimizer, scheduler


def execute(prepared_run, mode):
    run = Path(prepared_run).resolve()
    if Path(__file__).resolve() != run / 'source/train_utils/t04_10_runner.py':
        raise RuntimeError('Run only from the source snapshot inside the prepared run')
    verify(run)
    print('T04 manifest verification: PASS', flush=True)
    cfg = json.loads((run / 'config.resolved.json').read_text())
    if mode not in ('fresh', 'resume'):
        raise ValueError('Mode must be fresh or resume')
    if not torch.cuda.is_available():
        raise RuntimeError('Production requires CUDA. Use --check for CPU-only preflight.')
    device = torch.device('cuda')
    if cfg['precision'] == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA device does not support BF16')
    lock = run / 'training.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, 'w') as handle:
            handle.write(f'pid={os.getpid()} mode={mode}\n')
        print(f'EXECUTING {mode} {run} {run / "training"}', flush=True)
        train, val, model, optimizer, scheduler = build_run(cfg, device)
        # A separate loader generator is reseeded by epoch; resume does not
        # require reproducing prior DataLoader constructor RNG consumption.
        def seed_epoch(epoch):
            if train.generator is None:
                raise RuntimeError('Deterministic training generator missing')
            train.generator.manual_seed(int(cfg['seed']) + epoch)
        outcome = fit_fixed_budget(
            model=model, train_loader=train, val_loader=val, optimizer=optimizer,
            task=cfg['task'], output_dir=run / 'training',
            config_sha256=json.loads((run / 'run_manifest.json').read_text())['config_sha256'],
            epochs=cfg['epochs'], device=device, precision=cfg['precision'],
            scheduler=scheduler, resume=(mode == 'resume'), reset_loader_seed=seed_epoch,
            microbatch_size=cfg['batch_size'])
        (run / 'training_outcome.json').write_text(json.dumps(outcome, indent=2))
        history = json.loads((run / 'training/history.json').read_text())
        best = max(history, key=lambda row: row['val']['macro_f1'])
        session_outcome = {
            'epoch': outcome['epoch'], 'best_epoch': best['epoch'],
            'best_metric': best['val']['macro_f1'], 'early_stopped': False,
            'checkpoint': outcome['last_path']}
        print('T04-10 session completed: training ' + str(session_outcome), flush=True)
        print('SESSION_COMPLETED=' + str(run / 'training'), flush=True)
        print('T04-10 training complete:', json.dumps(outcome), flush=True)
        return outcome
    finally:
        # Even an exception releases only this process' exclusive lock.
        lock.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared-run', required=True)
    parser.add_argument('--mode', choices=('fresh', 'resume'), required=True)
    args = parser.parse_args(argv)
    execute(args.prepared_run, args.mode)

if __name__ == '__main__':
    main()
