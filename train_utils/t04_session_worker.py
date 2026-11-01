"""Execute exactly the frozen T04 snapshot for a session.

Separate from the legacy runner; this file exists only in newly prepared snapshots.
"""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from data.moment_loader import get_moment_dataloaders
from train_utils.t04_preflight import verify
from train_utils.t04_runner import FACTORIES, _runtime_details
from train_utils.ordinal_loss import build_moment_criterion
from train_utils.t03_runtime import resolve_scheduler
from train_utils.t04_session_training import train_session


def execute(run, session, mode, target_epochs):
    run, session = Path(run).resolve(), Path(session).resolve()
    verify(run)
    if Path(__file__).resolve() != run / 'source/train_utils/t04_session_worker.py':
        raise RuntimeError('Session worker must execute immutable source snapshot')
    if session.parent != run / 'sessions' or not (session / 'session_manifest.json').is_file():
        raise RuntimeError('Unknown session')
    cfg = json.loads((run / 'config.resolved.json').read_text())
    if cfg['strategy'] != 'S1' or cfg['gradient_accumulation'] != 1 or cfg['precision'] != 'fp32':
        raise NotImplementedError('T04-02 supports only S1 FP32 accumulation=1')
    if cfg['moment_loss']['type'] != 'cross_entropy' or cfg['learning_rates']['adapter'] != cfg['learning_rates']['heads']:
        raise NotImplementedError('T04-02 supports only equal adapter/head LR and cross entropy')
    settings = {'_strategy': cfg['strategy'], 'model': cfg['model'], 'seed': cfg['seed'],
        'dataset_root': cfg['dataset_root'], 'moments': cfg['moments'],
        'batch_size': cfg['batch_size'], 'num_workers': cfg['num_workers'],
        'learning_rate': cfg['learning_rates']['heads'], 'moment_loss': cfg['moment_loss'],
        '_checkpoint': cfg['checkpoint'], '_checkpoint_revision': cfg['checkpoint_revision']}
    if settings['_strategy'] != 'S1' or cfg['precision'] != 'fp32':
        raise NotImplementedError('Only S1 FP32 prepared runs are currently supported')
    if cfg['min_epochs'] > target_epochs:
        raise ValueError('target_epochs must be >= min_epochs')
    if cfg['min_epochs'] > 1 and cfg['min_epochs'] > cfg['epochs']:
        raise ValueError('invalid config epoch constraints')
    if not torch.cuda.is_available():
        raise RuntimeError('GPU session training requires CUDA')
    device = torch.device('cuda')
    seed = int(settings['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    train_loader, val_loader, _ = get_moment_dataloaders(
        dataset_root=settings['dataset_root'], moments=settings['moments'],
        batch_size=settings['batch_size'], num_workers=settings['num_workers'],
        seed=seed, validate_files=True, check_normalization=True)
    factory, backbone = FACTORIES[settings['model']]
    model, optimizer = factory(backbone=backbone,
        input_shape=(len(settings['moments']), 32, 32), dataset_type='moment',
        freeze_backbone=True, pretrained=True,
        learning_rate=float(settings['learning_rate']),
        checkpoint=settings['_checkpoint'], revision=settings['_checkpoint_revision'])
    model = model.to(device)
    spec = resolve_scheduler(cfg['scheduler'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=spec['mode'], factor=spec['factor'],
        patience=spec['patience'], min_lr=spec['min_lr'])
    criterion = build_moment_criterion(settings['moment_loss'], device=device)
    (session / 'runtime_environment.json').write_text(json.dumps(_runtime_details(model, device), indent=2))
    retention = os.environ.get('T04_CHECKPOINT_RETENTION', 'last_best')
    if retention not in ('last_best', 'all'):
        raise ValueError('Invalid T04_CHECKPOINT_RETENTION: use last_best or all')
    outcome = train_session(run=run, session=session, mode=mode,
        target_epochs=target_epochs, seed=seed, min_epochs=cfg['min_epochs'],
        patience=cfg['patience'], train_loader=train_loader, val_loader=val_loader,
        model=model, optimizer=optimizer, scheduler=scheduler, criterion=criterion,
        device=device, loss_weights={'alpha_output':1.0,'q0_output':1.0},
        checkpoint_retention=retention)
    (session / 'result.json').write_text(json.dumps(outcome, indent=2))
    print('T04-02 session completed:', session.name, outcome, flush=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--prepared-run', required=True)
    p.add_argument('--session', required=True)
    p.add_argument('--mode', choices=['fresh','resume'], required=True)
    p.add_argument('--target-epochs', type=int, required=True)
    args = p.parse_args(argv)
    execute(args.prepared_run, args.session, args.mode, args.target_epochs)


if __name__ == '__main__':
    main()
