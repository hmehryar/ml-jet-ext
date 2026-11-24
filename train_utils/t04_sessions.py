"""T04-02 Stage 2: append-only execution sessions for prepared experiments.

CPU-only. Checkpoint writing/training integration belongs to Stage 3.
Only a caller holding experiment_lock may create or complete sessions.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


class SessionError(RuntimeError):
    pass


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _inside(root, rel):
    rel = Path(rel)
    if rel.is_absolute() or '..' in rel.parts or not rel.parts:
        raise SessionError('Unsafe checkpoint path')
    target = (root / rel).resolve()
    if not target.is_relative_to(root.resolve()):
        raise SessionError('Checkpoint path escapes run directory')
    return target


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise SessionError('Missing or unreadable JSON: {}'.format(path)) from exc


def _validate_run(run):
    run = Path(run).expanduser().resolve()
    if not (run / 'run_manifest.json').is_file() or not (run / 'config.resolved.json').is_file():
        raise SessionError('Not an existing T04 prepared experiment')
    # Verify immutable provenance with the existing T04-01 module where available.
    from train_utils.t04_preflight import verify
    verify(run)
    return run


@contextmanager
def experiment_lock(run):
    """Nonblocking advisory lock; caller must hold for the *entire* execution.

    Unlike a permanent sentinel, the kernel releases this lock on process death.
    Note: all T04-02 executors must cooperate by acquiring this same lock.
    """
    run = _validate_run(run)
    fd = os.open(str(run / '.session-execution.lock'), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SessionError('Another execution owns this experiment') from exc
        os.ftruncate(fd, 0)
        os.write(fd, ('pid={} acquired={}\n'.format(os.getpid(), _utc())).encode())
        os.fsync(fd)
        yield run
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def latest_checkpoint(run):
    run = _validate_run(run)
    pointer_path = run / 'latest_checkpoint.json'
    if not pointer_path.exists():
        return None
    pointer = _read_json(pointer_path)
    if pointer.get('schema_version') != 1 or type(pointer.get('epoch')) is not int or pointer['epoch'] < 1:
        raise SessionError('Invalid latest checkpoint pointer')
    target = _inside(run, pointer.get('path', ''))
    if not target.is_file() or _digest(target) != pointer.get('sha256'):
        raise SessionError('Latest checkpoint missing or has mismatched SHA-256')
    return dict(pointer, absolute_path=str(target))


def start_session(run, *, mode, target_epochs, parent=None):
    """Reserve the next session. Call only while experiment_lock is held.

    mode='fresh' creates session-001; mode='resume' demands a committed parent.
    The caller must verify the checkpoint's embedded training identity in Stage 3.
    """
    run = _validate_run(run)
    if mode not in ('fresh', 'resume'):
        raise SessionError('Session mode must be fresh or resume')
    if type(target_epochs) is not int or target_epochs < 1:
        raise SessionError('target_epochs must be positive integer')
    sessions = run / 'sessions'
    sessions.mkdir(exist_ok=True)
    existing = sorted(p for p in sessions.iterdir() if p.is_dir() and re.fullmatch(r'session-\d{3,}', p.name))
    pointer = latest_checkpoint(run)
    if mode == 'fresh':
        if existing or pointer or any((run / n).exists() for n in ('last_model.pth', 'moment_epoch_metrics.csv')):
            raise SessionError('Fresh mode refuses an already used experiment')
        if parent is not None:
            raise SessionError('Fresh session cannot specify parent')
    else:
        if not pointer:
            raise SessionError('Resume requires a verified last committed checkpoint')
        if parent is not None and parent != pointer['path']:
            raise SessionError('Resume parent is not latest committed checkpoint')
        if target_epochs <= pointer['epoch']:
            raise SessionError('Target epochs must exceed completed epoch')
    number = max([int(p.name.split('-')[1]) for p in existing], default=0) + 1
    name = 'session-{:03d}'.format(number)
    directory = sessions / name
    directory.mkdir(exist_ok=False)
    (directory / 'checkpoints').mkdir()
    data = {
        'schema_version': 1, 'session': name, 'status': 'reserved',
        'mode': mode, 'created_utc': _utc(), 'target_epochs': target_epochs,
        'parent_checkpoint': ({k: pointer[k] for k in ('path', 'sha256', 'epoch')}
                              if pointer else None),
        'run_manifest_sha256': _digest(run / 'run_manifest.json'),
        'config_sha256': _digest(run / 'config.resolved.json'),
    }
    _atomic_json(directory / 'session_manifest.json', data)
    return directory


def publish_checkpoint(run, session, checkpoint, *, epoch):
    """Publish a fully written checkpoint. Caller holds experiment lock.

    Pointer update is atomic. Older session checkpoints are never overwritten.
    Epoch can advance within the same active session (rolling recovery path).
    """
    run = _validate_run(run)
    session = Path(session).resolve()
    if session.parent != (run / 'sessions').resolve() or not re.fullmatch(r'session-\d{3,}', session.name):
        raise SessionError('Session does not belong to experiment')
    manifest_path = session / 'session_manifest.json'
    manifest = _read_json(manifest_path)
    if manifest.get('status') not in ('reserved', 'running'):
        raise SessionError('Session is not writable')
    if type(epoch) is not int or epoch < 1 or epoch > manifest['target_epochs']:
        raise SessionError('Invalid checkpoint epoch')
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_relative_to((session / 'checkpoints').resolve()) or not checkpoint.is_file():
        raise SessionError('Checkpoint must exist inside this session checkpoints directory')
    prev = latest_checkpoint(run)
    if prev and epoch <= prev['epoch']:
        raise SessionError('Checkpoint epoch must advance monotonically')
    new = {'schema_version': 1, 'path': str(checkpoint.relative_to(run)),
           'sha256': _digest(checkpoint), 'epoch': epoch,
           'session': session.name, 'published_utc': _utc()}
    _atomic_json(run / 'latest_checkpoint.json', new)
    manifest['status'] = 'running'
    manifest['last_committed_epoch'] = epoch
    manifest['last_checkpoint_sha256'] = new['sha256']
    _atomic_json(manifest_path, manifest)
    return new


def finish_session(run, session, *, status):
    """Finalize a session, retaining it for audit. Valid: completed/interrupted."""
    run = _validate_run(run)
    session = Path(session).resolve()
    if session.parent != (run / 'sessions').resolve():
        raise SessionError('Session not owned by experiment')
    path = session / 'session_manifest.json'
    info = _read_json(path)
    if info.get('status') not in ('reserved', 'running') or status not in ('completed', 'interrupted'):
        raise SessionError('Invalid session state transition')
    info['status'] = status
    info['closed_utc'] = _utc()
    _atomic_json(path, info)
    return info
