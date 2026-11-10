"""T04-10 single-head adaptation of the T04-02/T04-07 session training contract.

Append-only sessions; SHA-256-verified checkpoint pointer; continuation training; no T04-07 writes; best-model selection is task validation macro-F1.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import tempfile

import torch

from train_utils.t04_10_checkpoint import _rng_state, _restore_rng, _atomic_save
from train_utils.t04_10_training import run_single_head_epoch
from train_utils.t04_10_sessions import (latest_checkpoint, publish_checkpoint,
                                         finish_session, SessionError, _atomic_json)
from train_utils.t04_telemetry import cuda_reset, epoch_telemetry, save_session_epoch_telemetry

FIELDS = ('epoch', 'train_loss', 'train_accuracy', 'train_macro_f1',
          'val_loss', 'val_accuracy', 'val_macro_f1', 'val_top1_accuracy',
          'val_top2_accuracy', 'val_top3_accuracy', 'val_within_1_class_accuracy',
          'val_within_2_class_accuracy', 'val_physical_mae', 'val_physical_rmse',
          'best_val_macro_f1')


def _rows(history):
    for item in history:
        tr, val = item['train'], item['val']
        yield dict(epoch=item['epoch'], train_loss=tr['loss'],
                   train_accuracy=tr['accuracy'], train_macro_f1=tr['macro_f1'],
                   val_loss=val['loss'], val_accuracy=val['accuracy'],
                   val_macro_f1=val['macro_f1'],
                   val_top1_accuracy=val['top1_accuracy'],
                   val_top2_accuracy=val['top2_accuracy'],
                   val_top3_accuracy=val['top3_accuracy'],
                   val_within_1_class_accuracy=val['within_1_class_accuracy'],
                   val_within_2_class_accuracy=val['within_2_class_accuracy'],
                   val_physical_mae=val['physical_mae'],
                   val_physical_rmse=val['physical_rmse'],
                   best_val_macro_f1=item['best_val_macro_f1'])


def _atomic_csv(path, history):
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix='.metrics-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', newline='', encoding='utf-8') as out:
            writer = csv.DictWriter(out, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(_rows(history))
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _verify_history(history, epoch, task, config_sha256):
    if not isinstance(history, list) or len(history) != epoch:
        raise SessionError('Checkpoint history length/epoch mismatch')
    if any(r.get('epoch') != i for i, r in enumerate(history, 1)):
        raise SessionError('Checkpoint history epoch sequence invalid')
    if any(r.get('val', {}).get('task') != task for r in history):
        raise SessionError('Checkpoint task history mismatch')
    if any(not isinstance(r.get('best_val_macro_f1'), (float, int)) for r in history):
        raise SessionError('Missing best macro-F1')


def _recover_last_session(run, pointer):
    """After abrupt SIGKILL, mark old session interrupted before reserving new one.

    Never touches completed sessions or their checkpoints.
    """
    if not pointer:
        return
    sessions_dir = Path(run) / 'sessions'
    for previous in sorted(sessions_dir.glob('session-*')):
        if not previous.is_dir():
            continue
        path = previous / 'session_manifest.json'
        info = json.loads(path.read_text())
        if info['status'] in ('reserved', 'running'):
            finish_session(run, previous, status='interrupted')
        elif info['status'] not in ('interrupted', 'completed'):
            raise SessionError('Invalid prior session state')


def run_session(*, run, mode, model, train_loader, val_loader, optimizer,
                scheduler, task, config_sha256, epochs, device='cpu', precision='fp32',
                reset_loader_seed=None, microbatch_size=None):
    """Reserve and execute one session while caller holds experiment_lock.

    Accepts any positive target epoch. Recovery starts at the latest published checkpoint epoch.
    """
    from train_utils.t04_10_sessions import start_session
    run = Path(run).resolve()
    if mode == 'resume':
        _recover_last_session(run, latest_checkpoint(run))
    session = start_session(run, mode=mode, target_epochs=epochs)
    try:
        result = train_session(run=run, session=session, mode=mode,
                               model=model, train_loader=train_loader,
                               val_loader=val_loader, optimizer=optimizer,
                               scheduler=scheduler, task=task,
                               config_sha256=config_sha256, epochs=epochs,
                               device=device, precision=precision,
                               reset_loader_seed=reset_loader_seed,
                               microbatch_size=microbatch_size)
        return session, result
    except BaseException:
        try:
            finish_session(run, session, status='interrupted')
        except Exception:
            pass
        raise


def train_session(*, run, session, mode, model, train_loader, val_loader,
                  optimizer, scheduler, task, config_sha256, epochs=80,
                  device='cpu', precision='fp32', reset_loader_seed=None,
                  microbatch_size=None):
    """Train an already reserved session with T04-07-compatible publication."""
    run, session = Path(run).resolve(), Path(session).resolve()
    if task not in ('alpha_only', 'q0_only') or getattr(model, 'task', None) != task:
        raise ValueError('Single-head task mismatch')
    if type(epochs) is not int or epochs < 1:
        raise ValueError('Target epochs must be a positive integer')
    manifest = json.loads((run / 'run_manifest.json').read_text())
    if config_sha256 != manifest['config_sha256']:
        raise SessionError('Configuration hash does not match prepared manifest')
    if mode not in ('fresh', 'resume'):
        raise ValueError('Invalid session mode')
    pointer = latest_checkpoint(run)
    history = []
    best_f1, best_epoch, start = -1.0, 0, 0
    if mode == 'resume':
        if pointer is None or pointer['epoch'] >= epochs:
            raise SessionError('No valid resumable checkpoint')
        saved = torch.load(pointer['absolute_path'], map_location=device, weights_only=False)
        if (saved.get('schema') != 2 or saved.get('task') != task or
            saved.get('config_sha256') != config_sha256 or
            saved.get('epoch') != pointer['epoch']):
            raise SessionError('Checkpoint training identity mismatch')
        if saved.get('scheduler') is None and scheduler is not None:
            raise SessionError('Missing scheduler state')
        if saved.get('scheduler') is not None and scheduler is None:
            raise SessionError('Unexpected scheduler state')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(saved['scheduler'])
        _restore_rng(saved['rng'])
        history = saved['history']
        start = int(saved['epoch'])
        _verify_history(history, start, task, config_sha256)
        best_f1 = float(saved['best_f1'])
        best_epoch = int(saved['best_epoch'])
        if best_f1 != history[best_epoch-1]['val']['macro_f1']:
            raise SessionError('Best metric/checkpoint inconsistency')
        # Checkpoint is authoritative: repair histories if interrupted after publish.
        _atomic_json(run / 'history.json', history)
        _atomic_csv(run / 'single_head_epoch_metrics.csv', history)
    elif pointer is not None or (run / 'history.json').exists():
        raise SessionError('Fresh session refuses existing results')
    if session.parent != run / 'sessions':
        raise SessionError('Session directory is outside run')
    if (session / 'session_manifest.json').is_file():
        info = json.loads((session / 'session_manifest.json').read_text())
        if info.get('mode') != mode or info.get('target_epochs') != epochs:
            raise SessionError('Incorrect session reservation')
    else:
        raise SessionError('Session was not reserved')
    model.to(device)
    for epoch in range(start + 1, epochs + 1):
        if reset_loader_seed:
            reset_loader_seed(epoch)
        epoch_start = cuda_reset(device)
        counts = {}
        train = run_single_head_epoch(train_loader, model, task=task, device=device,
                                      precision=precision, optimizer=optimizer,
                                      telemetry_counts=counts)
        val = run_single_head_epoch(val_loader, model, task=task, device=device,
                                    precision=precision)
        if scheduler is not None:
            scheduler.step(val['loss'])
        improved = val['macro_f1'] > best_f1
        if improved:
            best_f1, best_epoch = val['macro_f1'], epoch
        row = dict(epoch=epoch, train=train, val=val,
                   best_val_macro_f1=best_f1)
        history.append(row)
        state = dict(schema=2, task=task, epoch=epoch, best_epoch=best_epoch,
                     config_sha256=config_sha256, model=model.state_dict(),
                     optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict() if scheduler else None,
                     rng=_rng_state(), best_f1=best_f1, history=history)
        checkpoint = session / 'checkpoints' / ('epoch_%06d.pth' % epoch)
        _atomic_save(state, checkpoint)
        if improved:
            os.link(checkpoint, session / 'checkpoints' / ('best_epoch_%06d.pth' % epoch))
        publish_checkpoint(run, session, checkpoint, epoch=epoch)
        # A crash after pointer commit can be recovered from checkpoint history.
        _atomic_json(run / 'history.json', history)
        _atomic_csv(run / 'single_head_epoch_metrics.csv', history)
        telemetry = epoch_telemetry(epoch_start, device, optimizer,
                                   {'samples': train['n'], **counts}, val,
                                   microbatch_size=microbatch_size, accumulation=1)
        save_session_epoch_telemetry(session, epoch, telemetry)
        print('[T04-04] epoch telemetry: ' + json.dumps(telemetry, sort_keys=True), flush=True)
        # Same retention policy as T04-07: keep active latest + best; never touch past sessions.
        for stale in (session / 'checkpoints').glob('epoch_*.pth'):
            if stale != checkpoint:
                stale.unlink()
        if improved:
            for stale in (session / 'checkpoints').glob('best_epoch_*.pth'):
                if stale.name != ('best_epoch_%06d.pth' % epoch):
                    stale.unlink()
        tag = 'alpha' if task == 'alpha_only' else 'q0'
        print(f'[T04-10] Epoch {epoch} / {epochs}: val {tag} macro-F1 '
              f'{val["macro_f1"]:.5f} (best {best_f1:.5f} @ {best_epoch})', flush=True)
    # Write read-only per-session slice after completion.
    session_rows = history[start:epochs]
    _atomic_csv(session / 'single_head_epoch_metrics.csv', session_rows)
    finish_session(run, session, status='completed')
    return dict(epoch=epochs, best_epoch=best_epoch, best_metric=best_f1,
                early_stopped=False, checkpoint=str(checkpoint),
                session=str(session))
