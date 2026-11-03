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
from train_utils.t04_precision import make_scaler
from train_utils.t04_optimizers import build_t04_optimizer, learning_rate_summary



def validate_strategy_settings(cfg):
    """Fail closed on unsupported training options before allocating model memory."""
    strategy = cfg.get('strategy')
    if strategy not in ('S1', 'S2', 'S3', 'S4'):
        raise ValueError('Unknown T04 adaptation strategy')
    if cfg.get('precision') not in ('fp32','bf16','fp16') or type(cfg.get('gradient_accumulation')) is not int or cfg['gradient_accumulation'] < 1:
        raise ValueError('Invalid precision or gradient accumulation')
    if cfg.get('moment_loss', {}).get('type') != 'cross_entropy':
        raise NotImplementedError('Additional losses belong to T04-07')
    rates = cfg.get('learning_rates', {})
    if cfg.get('optimizer') is None:
        # Historical T04-03 behavior remains unchanged for old manifests.
        if rates.get('adapter') != rates.get('heads'):
            raise NotImplementedError('Separate adapter/head learning rates require optimizer configuration')
        if strategy != 'S1' and rates.get('backbone') != rates.get('heads'):
            raise NotImplementedError('Separate backbone learning rates require optimizer configuration')
    return strategy


def build_strategy_model(cfg, factories):
    """Create trainability mask before the optimizer; enforce S4 scratch."""
    strategy = validate_strategy_settings(cfg)
    factory, backbone = factories[cfg['model']]
    model, optimizer = factory(
        backbone=backbone, input_shape=(len(cfg['moments']), 32, 32),
        dataset_type='moment', freeze_backbone=(strategy == 'S1'),
        pretrained=(strategy != 'S4'), strategy=strategy,
        learning_rate=float(cfg['learning_rates']['heads']),
        checkpoint=cfg['checkpoint'], revision=cfg['checkpoint_revision'])
    if cfg.get('optimizer') is not None:
        spec = cfg['optimizer']
        optimizer = build_t04_optimizer(
            model, cfg['learning_rates'], optimizer_type=spec['type'],
            weight_decay=spec['weight_decay'])
    if getattr(model, 'strategy', None) != strategy:
        raise RuntimeError('Factory failed to apply configured strategy')
    if getattr(model, 'pretrained_backbone', None) is not (strategy != 'S4'):
        raise RuntimeError('Factory initialization source disagrees with strategy')
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    used = [id(p) for group in optimizer.param_groups for p in group['params']]
    if not expected or len(used) != len(set(used)) or set(used) != expected:
        raise RuntimeError('Optimizer parameters do not match adaptation mask')
    return model, optimizer


def resolve_dataloader_workers(cfg, *, env=None):
    """Resolve the effective worker count without mutating the frozen config.

    Only the explicit T04 v2 execution profile may override DataLoader
    workers. All other experiments retain their configured value.
    """
    env = os.environ if env is None else env
    configured = cfg["num_workers"]
    profile = env.get("T04_EXECUTION_PROFILE")

    if profile is None:
        return configured, "frozen_config"

    if profile != "cpu1_workers0_v1":
        raise ValueError("Unknown T04 execution profile")

    if cfg.get("training_protocol") != "fixed_budget_v2":
        raise ValueError("Execution profile requires fixed_budget_v2")

    if env.get("SLURM_CPUS_PER_TASK") != "1":
        raise ValueError("Execution profile requires exactly one Slurm CPU")

    if env.get("T04_DATALOADER_WORKERS") != "0":
        raise ValueError("Execution profile requires zero DataLoader workers")

    return 0, profile


def execute(run, session, mode, target_epochs):
    run, session = Path(run).resolve(), Path(session).resolve()
    verify(run)
    if Path(__file__).resolve() != run / 'source/train_utils/t04_session_worker.py':
        raise RuntimeError('Session worker must execute immutable source snapshot')
    if session.parent != run / 'sessions' or not (session / 'session_manifest.json').is_file():
        raise RuntimeError('Unknown session')
    cfg = json.loads((run / 'config.resolved.json').read_text())
    from train_utils.t04_training_termination import validate_protocol
    protocol_id = cfg.get('training_protocol', 'early_stop_v1')
    protocol_hash = cfg.get('training_protocol_hash')
    configured_epochs = cfg['epochs']
    validate_protocol(protocol_id, protocol_hash, configured_epochs)
    validate_strategy_settings(cfg)
    settings = {'model': cfg['model'], 'seed': cfg['seed'],
        'dataset_root': cfg['dataset_root'], 'moments': cfg['moments'],
        'batch_size': cfg['batch_size'], 'num_workers': cfg['num_workers'],
        'moment_loss': cfg['moment_loss']}

    effective_workers, execution_profile = resolve_dataloader_workers(cfg)
    settings['num_workers'] = effective_workers

    # Append-only execution metadata; never modify frozen configuration.
    execution_record = {
        'execution_profile': execution_profile,
        'configured_num_workers': cfg['num_workers'],
        'effective_num_workers': effective_workers,
        'slurm_cpus_per_task': os.environ.get('SLURM_CPUS_PER_TASK'),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
    }
    with (session / 'execution_profile.json').open('x') as stream:
        json.dump(execution_record, stream, indent=2, sort_keys=True)

    # min_epochs controls only legacy early-stopping eligibility.
    # Fixed-budget v2 can stop a Slurm segment before min_epochs.
    if cfg.get('training_protocol', 'early_stop_v1') != 'fixed_budget_v2' and cfg['min_epochs'] > target_epochs:
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
    model, optimizer = build_strategy_model(cfg, FACTORIES)
    model = model.to(device)
    spec = resolve_scheduler(cfg['scheduler'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=spec['mode'], factor=spec['factor'],
        patience=spec['patience'], min_lr=spec['min_lr'])
    criterion = build_moment_criterion(settings['moment_loss'], device=device)
    scaler = make_scaler(device, cfg['precision'])
    runtime = _runtime_details(model, device)
    runtime['adaptation'] = {
        'strategy': cfg['strategy'], 'pretrained_backbone': cfg['strategy'] != 'S4',
        'trainable_transformer_indices': model.unfrozen_transformer_indices,
        'frozen_backbone': cfg['strategy'] == 'S1',
        'optimizer_trainable_parameter_count': sum(
            p.numel() for group in optimizer.param_groups for p in group['params']),
        'optimizer_type': cfg.get('optimizer', {}).get('type', 'adam') if cfg.get('optimizer') else 'adam',
        'optimizer_learning_rates': learning_rate_summary(optimizer) if cfg.get('optimizer') else
            {'legacy': float(optimizer.param_groups[0]['lr'])}}
    (session / 'runtime_environment.json').write_text(json.dumps(runtime, indent=2))
    retention = os.environ.get('T04_CHECKPOINT_RETENTION', 'last_best')
    if retention not in ('last_best', 'all'):
        raise ValueError('Invalid T04_CHECKPOINT_RETENTION: use last_best or all')
    outcome = train_session(run=run, session=session, mode=mode,
        target_epochs=target_epochs, seed=seed, min_epochs=cfg['min_epochs'],
        patience=cfg['patience'], protocol_id=protocol_id, protocol_hash=protocol_hash,
        configured_epochs=configured_epochs,
        train_loader=train_loader, val_loader=val_loader,
        model=model, optimizer=optimizer, scheduler=scheduler, criterion=criterion,
        device=device, loss_weights={'alpha_output':1.0,'q0_output':1.0},
        checkpoint_retention=retention, scaler=scaler,
        precision=cfg['precision'], accumulation=cfg['gradient_accumulation'],
        microbatch_size=cfg['batch_size'])
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
