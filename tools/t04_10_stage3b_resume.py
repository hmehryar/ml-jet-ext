"""T04-10 #30 Stage 3B: real-backbone, 2-epoch fresh/resume GPU smoke.

Uses *only* train/validation aggregates. Never starts T04-07 or production jobs.
Run only under the dedicated Stage-3B Slurm allocation. One case per invocation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data.moment_loader import MomentJetDataset
from models.t04_10_single_head import (
    TASK_CLASSES, TASK_OUTPUT, build_clip_siglip_single_head,
    build_trainable_optimizer,
)
from train_utils.t04_10_checkpoint import fit_fixed_budget
from train_utils.t04_10_preflight import inspect_inputs, load_settings, sha256
from train_utils.t04_10_training import run_single_head_epoch


def check_metrics(metrics, *, task: str, expected_n: int):
    if metrics['task'] != task or metrics['n'] != expected_n:
        raise AssertionError('Task or sample count mismatch')
    keys = ('accuracy', 'macro_f1', 'top1_accuracy', 'top2_accuracy',
            'top3_accuracy', 'within_1_class_accuracy', 'within_2_class_accuracy')
    for key in keys:
        value = metrics[key]
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise AssertionError(f'Invalid metric {key}: {value}')
    if not (metrics['top1_accuracy'] <= metrics['top2_accuracy'] <=
            metrics['top3_accuracy']):
        raise AssertionError('Top-k monotonicity failed')
    if abs(metrics['top1_accuracy'] - metrics['accuracy']) > 1e-12:
        raise AssertionError('Legacy accuracy and top-1 differ')
    for key in ('physical_mae', 'physical_rmse', 'loss'):
        if not math.isfinite(metrics[key]) or metrics[key] < 0:
            raise AssertionError(f'Invalid {key}: {metrics[key]}')
    cm = metrics['confusion_matrix']
    classes = TASK_CLASSES[task]
    if len(cm) != classes or any(len(row) != classes for row in cm):
        raise AssertionError('Wrong confusion-matrix dimensions')
    if sum(sum(row) for row in cm) != expected_n:
        raise AssertionError('Confusion matrix count mismatch')
    if 'joint_accuracy' in metrics:
        raise AssertionError('Unexpected joint-head metric')


def validate_state(state, *, task: str, epoch: int, config_sha256: str):
    if state['schema'] != 1 or state['task'] != task or state['epoch'] != epoch:
        raise AssertionError('Checkpoint metadata mismatch')
    if state['config_sha256'] != config_sha256:
        raise AssertionError('Checkpoint configuration mismatch')
    if len(state['history']) != epoch:
        raise AssertionError('Checkpoint history truncated')
    if not state['optimizer']['state']:
        raise AssertionError('AdamW optimizer state was not saved')
    if state['scheduler'] is None or 'best' not in state['scheduler']:
        raise AssertionError('ReduceLROnPlateau state missing')
    if not all(k in state['rng'] for k in ('python', 'numpy', 'torch', 'cuda')):
        raise AssertionError('RNG state incomplete')
    if torch.cuda.is_available() and state['rng']['cuda'] is None:
        raise AssertionError('CUDA RNG state not saved')
    for key in state['model']:
        if 'alpha_head' in key or 'q0_head' in key:
            raise AssertionError('Inactive head leaked into checkpoint')
    for row in state['history']:
        check_metrics(row['train'], task=task, expected_n=row['train']['n'])
        check_metrics(row['val'], task=task, expected_n=row['val']['n'])


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders(cfg, *, train_samples: int, val_samples: int):
    root = Path(cfg['dataset_root'])
    # In particular, never construct a test loader or load test samples.
    train_ds = MomentJetDataset(root / 'splits/train.csv', root, cfg['moments'],
                                validate_files=False, check_normalization=True)
    val_ds = MomentJetDataset(root / 'splits/val.csv', root, cfg['moments'],
                              validate_files=False, check_normalization=True)
    if train_samples > len(train_ds) or val_samples > len(val_ds):
        raise ValueError('Requested subset larger than split')
    if set(map(str, train_ds.paths[:train_samples])) & set(map(str, val_ds.paths[:val_samples])):
        raise ValueError('Aggregate path overlap in smoke subsets')
    train_loader = DataLoader(Subset(train_ds, range(train_samples)), batch_size=1,
                              shuffle=True, num_workers=0,
                              generator=torch.Generator().manual_seed(cfg['seed']))
    val_loader = DataLoader(Subset(val_ds, range(val_samples)), batch_size=1,
                            shuffle=False, num_workers=0)
    val_ids = [str(p.relative_to(root)) for p in val_ds.paths[:val_samples]]
    if len(set(val_ids)) != len(val_ids):
        raise AssertionError('Duplicate validation IDs')
    return train_loader, val_loader, val_ids


def build_model_optimizer_scheduler(cfg):
    model = build_clip_siglip_single_head(
        architecture=cfg['model'], task=cfg['task'], channels=len(cfg['moments']),
        strategy=cfg['strategy'], checkpoint=cfg['checkpoint'],
        revision=cfg['checkpoint_revision'], pretrained=True).cuda()
    if model.classification_head.out_features != TASK_CLASSES[cfg['task']]:
        raise AssertionError('Wrong classifier output dimension')
    if any('alpha_head' in name or 'q0_head' in name for name, _ in model.named_parameters()):
        raise AssertionError('An unused classifier was instantiated')
    lr = cfg['learning_rates']
    optimizer = build_trainable_optimizer(
        model, lr_backbone=lr['backbone'], lr_adapter=lr['adapter'],
        lr_head=lr['heads'], weight_decay=cfg['optimizer']['weight_decay'])
    sch = cfg['scheduler']
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=sch['factor'], patience=sch['patience'],
        min_lr=sch['min_lr'])
    return model, optimizer, scheduler


def execute(config_path, output_root, *, train_samples=6, val_samples=6):
    cfg = load_settings(config_path)
    if cfg['strategy'] != 'S3' or cfg['precision'] != 'bf16':
        raise ValueError('Real pretrained GPU Stage 3B uses S3 and BF16 only')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA BF16 device required')
    if not 1 <= train_samples <= 32 or not 1 <= val_samples <= 32:
        raise ValueError('Smoke subsets must have 1..32 samples')
    audit = inspect_inputs(cfg)  # Enforces split/audit integrity without test inference.
    output_root = Path(output_root).resolve()
    # Independent namespace, never the T04-07 root or an 80-epoch production run.
    if 't04_10_single_head' not in output_root.parts or 'stage3b_resume' not in output_root.parts:
        raise ValueError('Use a dedicated /rs/.../t04_10_single_head/stage3b_resume path')
    if not output_root.is_dir():
        raise FileNotFoundError('Output directory must be created before the job')
    case = f"{cfg['model']}_{cfg['task']}"
    case_root = output_root / case
    case_root.mkdir(exist_ok=False)
    config_hash = hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
    seed_everything(cfg['seed'])
    tr, val, sample_ids = make_loaders(cfg, train_samples=train_samples, val_samples=val_samples)
    model, optimizer, scheduler = build_model_optimizer_scheduler(cfg)
    seed_epoch = lambda epoch: tr.generator.manual_seed(cfg['seed'] + epoch)

    first = fit_fixed_budget(
        model=model, train_loader=tr, val_loader=val, optimizer=optimizer,
        task=cfg['task'], output_dir=case_root / 'training',
        config_sha256=config_hash, epochs=1, device='cuda', precision='bf16',
        scheduler=scheduler, reset_loader_seed=seed_epoch)
    latest = Path(first['last_path'])
    if not latest.is_file() or not Path(first['best_path']).is_file():
        raise AssertionError('Epoch-1 best/last checkpoint missing')
    state1 = torch.load(latest, map_location='cuda', weights_only=False)
    validate_state(state1, task=cfg['task'], epoch=1, config_sha256=config_hash)
    check_metrics(state1['history'][0]['val'], task=cfg['task'], expected_n=val_samples)
    step1 = _optimizer_steps(state1)
    del model, optimizer, scheduler, state1
    torch.cuda.empty_cache()

    # Reinitialize the full model from a pinned checkpoint to emulate a new process.
    # fit_fixed_budget(resume=True) restores full model, optimizer, scheduler & RNG.
    seed_everything(cfg['seed'] + 999)
    model, optimizer, scheduler = build_model_optimizer_scheduler(cfg)
    second = fit_fixed_budget(
        model=model, train_loader=tr, val_loader=val, optimizer=optimizer,
        task=cfg['task'], output_dir=case_root / 'training',
        config_sha256=config_hash, epochs=2, device='cuda', precision='bf16',
        scheduler=scheduler, resume=True, reset_loader_seed=seed_epoch)
    state2 = torch.load(second['last_path'], map_location='cuda', weights_only=False)
    validate_state(state2, task=cfg['task'], epoch=2, config_sha256=config_hash)
    if _optimizer_steps(state2) <= step1:
        raise AssertionError('Optimizer steps did not advance after resume')
    if state2['history'][0]['val'] != json.loads((case_root/'training/history.json').read_text())[0]['val']:
        raise AssertionError('Epoch-1 metric history changed after resume')
    for i, row in enumerate(state2['history']):
        check_metrics(row['train'], task=cfg['task'], expected_n=train_samples)
        check_metrics(row['val'], task=cfg['task'], expected_n=val_samples)

    # Reload the best checkpoint to verify validation metrics and exports, not test.
    best_file = Path(second['best_path'])
    best = torch.load(best_file, map_location='cuda', weights_only=False)
    model.load_state_dict(best['model'], strict=True)
    selected_epoch = int(best['epoch'])
    val_export = case_root / 'validation_predictions.json'
    metrics = run_single_head_epoch(
        val, model, task=cfg['task'], device='cuda', precision='bf16',
        prediction_export_path=val_export, sample_ids=sample_ids,
        export_metadata={
            'run_id': case_root.parent.name + '/' + case,
            'architecture': cfg['model'], 'strategy': cfg['strategy'],
            'moments': cfg['moments'], 'checkpoint_sha256': sha256(best_file),
            'split': 'val'})
    check_metrics(metrics, task=cfg['task'], expected_n=val_samples)
    expected = best['history'][selected_epoch-1]['val']
    for key in ('accuracy','macro_f1','top1_accuracy','top2_accuracy',
                'top3_accuracy','physical_mae','physical_rmse','loss'):
        if abs(metrics[key] - expected[key]) > 2e-5:
            raise AssertionError(f'Reloaded best checkpoint mismatch for {key}: '
                                 f'{metrics[key]} vs {expected[key]}')
    export = json.loads(val_export.read_text())
    if len(export['predictions']) != val_samples or export['metadata']['split'] != 'val':
        raise AssertionError('Invalid validation-only prediction artifact')
    if any(len(rec['logits']) != TASK_CLASSES[cfg['task']] for rec in export['predictions']):
        raise AssertionError('Wrong exported head dimensions')
    summary = {
        'status': 'PASS', 'issue': 30, 'phase': 'stage3b_resume',
        'architecture': cfg['model'], 'task': cfg['task'],
        'epochs': 2, 'resumed_from_epoch': 1, 'best_epoch': selected_epoch,
        'train_samples': train_samples, 'validation_samples': val_samples,
        'source_overlap_disclosed': audit['source_overlap'],
        'checkpoint_sha256': sha256(best_file), 'last_sha256': sha256(latest),
        'validation_metrics': metrics,
        'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
    }
    (case_root/'summary.json').write_text(json.dumps(summary, indent=2))
    print('STAGE3B_RESULT ' + json.dumps({k: v for k, v in summary.items()
                                         if k not in ('validation_metrics',)}), flush=True)
    return summary


def _optimizer_steps(state):
    steps = []
    for entry in state['optimizer']['state'].values():
        if 'step' in entry:
            steps.append(int(entry['step']))
    if not steps:
        raise AssertionError('AdamW step counters absent')
    return max(steps)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output-root', required=True)
    p.add_argument('--train-samples', type=int, default=6)
    p.add_argument('--val-samples', type=int, default=6)
    a = p.parse_args(argv)
    execute(a.config, a.output_root, train_samples=a.train_samples, val_samples=a.val_samples)


if __name__ == '__main__':
    main()
