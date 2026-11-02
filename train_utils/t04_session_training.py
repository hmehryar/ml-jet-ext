"""T04-02: epoch-boundary resumable training for *prepared* T04 S1 runs.

The original moment_training.run_moment_training_loop remains unchanged for T03.
Each epoch is first committed as a uniquely named immutable checkpoint.
With the default last_best retention policy, earlier checkpoints from the
CURRENT session are pruned only after publication of a new pointer.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import torch

from train_utils.moment_training import train_moment_epoch, evaluate_moment
from train_utils.t04_telemetry import cuda_reset, epoch_telemetry, save_session_epoch_telemetry
from train_utils.t04_precision import run_epoch
from train_utils.t04_resume_state import (atomic_torch_save, checkpoint_state,
    load_checkpoint, restore_training, reconcile_epoch_csv, seed_loader_epoch,
    ResumeError, sha256)
from train_utils.t04_sessions import (latest_checkpoint, publish_checkpoint,
    finish_session, SessionError)

COLUMNS = ['epoch','train_loss','train_alpha_accuracy','train_q0_accuracy',
           'train_joint_accuracy','train_alpha_loss','train_q0_loss','val_loss',
           'val_alpha_loss','val_q0_loss','val_alpha_accuracy','val_q0_accuracy',
           'val_joint_accuracy','val_alpha_macro_f1','val_q0_macro_f1','lr']
COLUMNS += [f'{sp}_{tag}_{component}_loss' for sp in ('train','val')
            for tag in ('alpha','q0') for component in ('ce','ordinal')]
COLUMNS += [f'{sp}_{tag}_{metric}' for sp in ('train','val')
            for tag in ('alpha','q0') for metric in ('mae','rmse','normalized_mae')]


def _record(epoch, train, val, optimizer):
    row = dict(epoch=epoch, train_loss=train['loss'],
        train_alpha_accuracy=train['alpha']['accuracy'],
        train_q0_accuracy=train['q0']['accuracy'], train_joint_accuracy=train['accuracy'],
        train_alpha_loss=train['loss_alpha'], train_q0_loss=train['loss_q0'],
        val_loss=val['loss'], val_alpha_loss=val['loss_alpha'],
        val_q0_loss=val['loss_q0'], val_alpha_accuracy=val['alpha']['accuracy'],
        val_q0_accuracy=val['q0']['accuracy'], val_joint_accuracy=val['accuracy'],
        val_alpha_macro_f1=val['alpha']['f1'], val_q0_macro_f1=val['q0']['f1'],
        lr=optimizer.param_groups[0]['lr'])
    for split, metrics in (('train', train), ('val', val)):
        for tag in ('alpha','q0'):
            for comp in ('ce','ordinal'):
                row[f'{split}_{tag}_{comp}_loss'] = metrics.get(f'loss_{tag}_{comp}', '')
            for metric in ('mae','rmse','normalized_mae'):
                row[f'{split}_{tag}_{metric}'] = metrics[tag].get(metric, '')
    return row


def _append_csv(path, row):
    path = Path(path)
    with path.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if f.tell() == 0:
            w.writeheader()
        w.writerow(row)
        f.flush()
        import os
        os.fsync(f.fileno())


def _metric_history(run, checkpoint):
    """Authoritative checkpoint covers the last CSV row; previous rows are verified for ordering."""
    csv_path = Path(run) / 'moment_epoch_metrics.csv'
    reconcile_epoch_csv(csv_path, checkpoint)
    return csv_path


def _retain_checkpoints(run, session, checkpoint, *, improved, retention):
    """Prune only this active session after latest pointer securely advances.

    The best epoch is represented by a separate immutable hard link, which
    remains valid even after pruning its original epoch recovery filename.
    Existing completed sessions are never inspected or changed here.
    """
    if retention == 'all':
        return
    if retention != 'last_best':
        raise ValueError('Unknown checkpoint retention policy')
    directory = session / 'checkpoints'
    # Best copy must already exist before old files are removed.
    best_files = sorted(directory.glob('best_epoch_*.pth'))
    if improved:
        best = directory / ('best_epoch_%06d.pth' % int(checkpoint.stem.split('_')[-1]))
        if not best.exists():
            raise RuntimeError('Missing best checkpoint after commit')
        for old in best_files:
            if old != best:
                old.unlink()
    # Never prune the file that latest_checkpoint.json currently references.
    current = latest_checkpoint(run)
    if current is None or Path(current['absolute_path']).resolve() != checkpoint.resolve():
        raise RuntimeError('Cannot prune before the latest checkpoint is committed')
    for old in directory.glob('epoch_*.pth'):
        if old != checkpoint:
            old.unlink()


def train_session(*, run, session, mode, target_epochs, seed, min_epochs, patience,
                  train_loader, val_loader, model, optimizer, scheduler, criterion,
                  device, scaler=None, loss_weights=None, checkpoint_retention='last_best',
                  precision='fp32', accumulation=1, microbatch_size=None):
    run, session = Path(run).resolve(), Path(session).resolve()
    if checkpoint_retention not in ('last_best', 'all'):
        raise ValueError('checkpoint_retention must be last_best or all')
    if mode not in ('fresh','resume'):
        raise ValueError('Invalid mode')
    if type(target_epochs) is not int or target_epochs < 1:
        raise ValueError('Invalid target epochs')
    if min_epochs < 1 or patience < 1:
        raise ValueError('Invalid early-stop parameters')
    # Bind each checkpoint to the prepared experiment identity. The run manifest
    # is deliberately not rewritten during training.
    manifest_hash = sha256(run / 'run_manifest.json')
    config_hash = sha256(run / 'config.resolved.json')
    history = run / 'moment_epoch_metrics.csv'
    best_metric, best_epoch, stagnant, start_epoch = -1., -1, 0, 0
    if mode == 'resume':
        pointer = latest_checkpoint(run)
        if not pointer:
            raise SessionError('Resume requested without latest checkpoint')
        state = load_checkpoint(pointer['absolute_path'], map_location=device,
            manifest_sha256=manifest_hash, config_sha256=config_hash)
        if state['epoch'] != pointer['epoch']:
            raise ResumeError('Checkpoint/pointer epoch disagreement')
        if state.get('stopped'):
            raise ResumeError('Cannot resume a training trajectory terminated by early stopping')
        start_epoch, best_metric, best_epoch, stagnant = restore_training(state,
            model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler)
        _metric_history(run, state)
    elif history.exists() or latest_checkpoint(run) is not None:
        raise SessionError('Fresh session refuses existing training history')

    if start_epoch >= target_epochs:
        raise ResumeError('Target epoch count must exceed last completed epoch')
    if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        raise ValueError('Only ReduceLROnPlateau is supported by prepared T04-02')
    # Preserve epoch-wise metric trace; no duplicate epochs can be appended.
    stop = False
    for epoch in range(start_epoch + 1, target_epochs + 1):
        seed_loader_epoch(train_loader, seed, epoch)
        telemetry_start = cuda_reset(device)
        if precision == 'fp32' and accumulation == 1:
            tr = train_moment_epoch(train_loader, model, criterion, optimizer, device, loss_weights)
            va = evaluate_moment(val_loader, model, criterion, device, loss_weights)
        else:
            tr = run_epoch(train_loader, model, criterion, device, loss_weights,
                           optimizer, precision=precision, accumulation=accumulation, scaler=scaler)
            va = run_epoch(val_loader, model, criterion, device, loss_weights,
                           precision=precision)
        if scheduler is not None:
            scheduler_settings = json.loads((run / 'config.resolved.json').read_text())['scheduler']
            scheduler.step(va['loss'] if scheduler_settings['monitor'] == 'val_loss' else va['accuracy'])
        metric = float(va['accuracy'])
        improved = metric > best_metric
        if improved:
            best_metric, best_epoch, stagnant = metric, epoch, 0
        else:
            stagnant += 1
        stop = epoch >= min_epochs and stagnant >= patience
        row = _record(epoch, tr, va, optimizer)
        state = checkpoint_state(epoch=epoch, model=model, optimizer=optimizer,
            scheduler=scheduler, scaler=scaler, best_metric=best_metric,
            best_epoch=best_epoch, stagnant=stagnant, stopped=stop,
            manifest_sha256=manifest_hash, config_sha256=config_hash,
            metrics_row=row, metrics_columns=COLUMNS)
        checkpoint = session / 'checkpoints' / ('epoch_%06d.pth' % epoch)
        atomic_torch_save(state, checkpoint)
        # Preserve an independently named best-epoch checkpoint before the
        # latest-pointer commit, so pruning cannot lose a new best model.
        if improved and checkpoint_retention == 'last_best':
            best_path = session / 'checkpoints' / ('best_epoch_%06d.pth' % epoch)
            os.link(checkpoint, best_path)  # same filesystem, never mutates checkpoint
        # Pointer is the commit; CSV may be reconstructed after a crash.
        publish_checkpoint(run, session, checkpoint, epoch=epoch)
        _append_csv(history, row)
        telemetry = epoch_telemetry(telemetry_start, device, optimizer, tr, va,
                                    microbatch_size, accumulation)
        save_session_epoch_telemetry(session, epoch, telemetry)
        print('[T04-04] epoch telemetry: ' + json.dumps(telemetry, sort_keys=True), flush=True)
        _retain_checkpoints(run, session, checkpoint, improved=improved,
                            retention=checkpoint_retention)
        print('[T04-02] Epoch %d / %d: val joint %.5f (best %.5f @ %d)' %
            (epoch, target_epochs, metric, best_metric, best_epoch), flush=True)
        if stop:
            break
    # Session CSV is a *copy of the committed interval*, not an editable global log.
    with history.open(newline='') as f:
        rows = list(csv.DictReader(f))
    interval = rows[start_epoch:epoch]
    local_history = session / 'moment_epoch_metrics.csv'
    with local_history.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(interval)
    finish_session(run, session, status='completed')
    return {'epoch': epoch, 'best_epoch': best_epoch, 'best_metric': best_metric,
            'early_stopped': stop, 'checkpoint': str(checkpoint)}
