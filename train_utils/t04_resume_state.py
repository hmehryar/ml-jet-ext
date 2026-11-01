"""T04-02 durable epoch-boundary state for *new* resumable experiments.

This module does not retrofit earlier T04-01 checkpoints that lack RNG and
best/stagnation state. No device allocation occurs on import.
"""
from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch


class ResumeError(RuntimeError):
    pass


def capture_rng():
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    if not isinstance(state, dict) or not {'python', 'numpy', 'torch_cpu', 'torch_cuda'} <= state.keys():
        raise ResumeError('Checkpoint has no complete RNG state')
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'].cpu())
    if state['torch_cuda'] is not None:
        if not torch.cuda.is_available() or len(state['torch_cuda']) != torch.cuda.device_count():
            raise ResumeError('CUDA device topology differs from saved RNG state')
        torch.cuda.set_rng_state_all([s.cpu() for s in state['torch_cuda']])


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_torch_save(state, destination):
    """A failed serialization never replaces a valid previous checkpoint."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f'.{destination.name}.', suffix='.tmp', dir=str(destination.parent))
    try:
        with os.fdopen(fd, 'wb') as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, destination)
        sync_dir(destination.parent)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def sync_dir(path):
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def checkpoint_state(*, epoch, model, optimizer, scheduler, scaler, best_metric,
                     best_epoch, stagnant, manifest_sha256, config_sha256,
                     metrics_row, metrics_columns, metric_mode='max', stopped=False):
    if epoch < 1 or best_epoch > epoch or stagnant < 0:
        raise ValueError('Invalid epoch or early-stopping counters')
    if metric_mode not in ('max', 'min'):
        raise ValueError('Invalid metric direction')
    return {
        'resume_schema': 1, 'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'best_metric': float(best_metric), 'best_epoch': int(best_epoch),
        'stagnant': int(stagnant), 'stopped': bool(stopped),
        'metric_mode': metric_mode, 'rng_state': capture_rng(),
        'manifest_sha256': manifest_sha256, 'config_sha256': config_sha256,
        'metrics_row': dict(metrics_row), 'metrics_columns': list(metrics_columns),
        'sampler_protocol': 'deterministic_per_epoch_seed_v1',
    }


def load_checkpoint(path, *, map_location='cpu', manifest_sha256, config_sha256):
    path = Path(path)
    if not path.is_file():
        raise ResumeError(f'Checkpoint absent: {path}')
    try:
        state = torch.load(path, map_location=map_location, weights_only=False)
    except Exception as exc:
        raise ResumeError(f'Checkpoint unreadable: {path}') from exc
    if not isinstance(state, dict) or state.get('resume_schema') != 1:
        raise ResumeError('Checkpoint is not a T04-02 resumable checkpoint')
    if state.get('manifest_sha256') != manifest_sha256 or state.get('config_sha256') != config_sha256:
        raise ResumeError('Run identity or immutable config changed')
    required = {'epoch', 'model_state_dict', 'optimizer_state_dict', 'rng_state',
                'best_metric', 'best_epoch', 'stagnant', 'metrics_row', 'metrics_columns',
                'sampler_protocol', 'stopped', 'metric_mode'}
    if not required <= state.keys() or state['sampler_protocol'] != 'deterministic_per_epoch_seed_v1':
        raise ResumeError('Missing resumable state or unsupported sampler protocol')
    if type(state['epoch']) is not int or state['epoch'] < 1:
        raise ResumeError('Invalid checkpoint epoch')
    if state['best_epoch'] > state['epoch'] or state['stagnant'] < 0:
        raise ResumeError('Invalid early-stop state')
    return state


def restore_training(state, *, model, optimizer, scheduler=None, scaler=None):
    """Call after model, optimizer, scheduler and loader construction."""
    if (scheduler is None) != (state.get('scheduler_state_dict') is None):
        raise ResumeError('Scheduler identity differs')
    if (scaler is None) != (state.get('scaler_state_dict') is None):
        raise ResumeError('AMP scaler identity differs')
    model.load_state_dict(state['model_state_dict'], strict=True)
    optimizer.load_state_dict(state['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(state['scheduler_state_dict'])
    if scaler is not None:
        scaler.load_state_dict(state['scaler_state_dict'])
    restore_rng(state['rng_state'])
    return state['epoch'], state['best_metric'], state['best_epoch'], state['stagnant']


def reconcile_epoch_csv(path, state):
    """Rebuild missing last CSV row after crash; never accept divergent rows.

    The checkpoint is the authoritative COMMITTED epoch. A later CSV row
    indicates an incomplete next epoch and is rejected, not silently deleted.
    """
    path = Path(path)
    cols = state['metrics_columns']
    last = {key: str(state['metrics_row'][key]) for key in cols}
    if not path.exists():
        rows = []
    else:
        with path.open(newline='') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != cols:
                raise ResumeError('CSV columns changed')
            rows = list(reader)
    epochs = []
    for row in rows:
        try:
            epochs.append(int(row['epoch']))
        except (KeyError, ValueError) as exc:
            raise ResumeError('Malformed CSV epoch') from exc
    if epochs != list(range(1, len(epochs) + 1)):
        raise ResumeError('CSV contains gaps or duplicate epochs')
    if len(rows) > state['epoch']:
        raise ResumeError('CSV is ahead of last committed checkpoint')
    if len(rows) < state['epoch'] - 1:
        raise ResumeError('Missing too many CSV rows to recover safely')
    if len(rows) == state['epoch']:
        if rows[-1] != last:
            raise ResumeError('Committed CSV row disagrees with checkpoint')
        return
    if len(rows) == state['epoch'] - 1:
        with path.open('a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols)
            if not rows:
                w.writeheader()
            w.writerow(last)
            f.flush()
            os.fsync(f.fileno())


def epoch_seed(base_seed, epoch):
    """Stable seed for each epoch; independent of prior worker RNG consumption."""
    if type(base_seed) is not int or type(epoch) is not int or epoch < 1:
        raise ValueError('Expected integer seed and positive epoch')
    return (base_seed + 1000003 * epoch) % (2**63 - 1)


def seed_loader_epoch(loader, base_seed, epoch):
    """Requires an explicit shuffle Generator attached to the DataLoader.

    For num_workers>0 worker seeding also depends on this generator. The caller
    must use the same multiprocessing/data-loading configuration on resume.
    """
    gen = getattr(loader, 'generator', None)
    if gen is None:
        raise ResumeError('Training DataLoader must have an explicit torch.Generator')
    gen.manual_seed(epoch_seed(base_seed, epoch))
