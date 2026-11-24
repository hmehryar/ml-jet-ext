"""T04-06 Stage 2: transactional, append-only-attempt operational registry.

No Slurm submission, process spawning, or immutable-run reservation here.
This is a sidecar to the Stage-1 read-only planning registry.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import uuid
from train_utils.t04_v2_relocation import matches_original_digest

VALID = {'planned', 'submitted', 'running', 'completed', 'failed', 'resumed', 'ready_to_resume'}
TRANSITIONS = {
    'planned': {'submitted'},
    'submitted': {'running', 'failed'},
    'running': {'completed', 'failed', 'ready_to_resume'},
    'completed': {'resumed'},
    'failed': {'resumed'},
    'resumed': {'running', 'failed'},
    'ready_to_resume': {'resumed'},
}
SHA = re.compile(r'^[0-9a-f]{64}$')
JOB = re.compile(r'^\d+(?:_\d+)?$')


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()


def atomic_replace(path, doc):
    path = Path(path)
    payload = (json.dumps(doc, indent=2, sort_keys=True) + '\n').encode()
    fd, name = tempfile.mkstemp(prefix='.t04-06-', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
        directory_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextmanager
def locked(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with (path.parent / (path.name + '.lock')).open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _load_plan(repo, plan_path):
    plan = json.loads(Path(plan_path).read_text())
    if plan.get('kind') == 'T04-06-v2-plan':
        from tools.t04_06_v2_matrix import verify
    elif plan.get('kind') == 'T04-06-plan':
        from tools.t04_06_matrix import verify
    else:
        raise ValueError('Unknown T04-06 plan kind')
    # Bootstrap only before any experiment run is reserved.
    verify(repo, plan_path)
    plan = json.loads(Path(plan_path).read_text())
    return plan


def bootstrap(repo, plan_path, runtime_path, git_commit):
    repo, plan_path, runtime_path = Path(repo).resolve(), Path(plan_path).resolve(), Path(runtime_path).resolve()
    if not re.fullmatch(r'[0-9a-f]{40}', git_commit):
        raise ValueError('git commit must be a full 40-character SHA')
    with locked(runtime_path):
        if runtime_path.exists():
            raise FileExistsError('Operational registry exists: ' + str(runtime_path))
        plan = _load_plan(repo, plan_path)
        entries = {}
        for row in plan['entries']:
            if row['id'] in entries:
                raise ValueError('Duplicate experiment ID')
            entries[row['id']] = {
                'id': row['id'], 'config': row['config'], 'config_sha256': row['config_sha256'],
                'run': row['run'], 'git_commit': git_commit,
                'checkpoint_revision': None, 'status': 'planned',
                'attempts': [], 'events': [{'at': now(), 'event': 'planned'}],
            }
        doc = {'schema': 1, 'kind': 'T04-06-operational',
               'plan_sha256': sha(plan_path), 'plan_entries_sha256': plan['plan_sha256'],
               'plan_path': str(plan_path), 'repo': str(repo),
               'entries': entries}
        atomic_replace(runtime_path, doc)
    return len(entries)


def validate(doc):
    if doc.get('schema') != 1 or doc.get('kind') != 'T04-06-operational':
        raise ValueError('Invalid operational registry schema')
    if not SHA.fullmatch(doc.get('plan_sha256', '')):
        raise ValueError('Invalid plan hash')
    entries = doc.get('entries')
    if not isinstance(entries, dict) or len(entries) != 120:
        raise ValueError('Expected exactly 120 entries')
    runs = set()
    for key, entry in entries.items():
        if key != entry.get('id') or entry.get('status') not in VALID:
            raise ValueError('Corrupt experiment entry: ' + str(key))
        if entry['run'] in runs:
            raise ValueError('Duplicate output directory')
        runs.add(entry['run'])
        if not SHA.fullmatch(entry['config_sha256']):
            raise ValueError('Invalid config hash')
        active = [a for a in entry['attempts'] if a.get('status') in ('submitted','running','resumed')]
        if len(active) > 1:
            raise ValueError('Duplicate active attempts')
        if entry['status'] in ('submitted','running','resumed') and len(active) != 1:
            raise ValueError('Active state without active attempt')
        if entry['status'] not in ('submitted','running','resumed') and active:
            raise ValueError('Inactive state with active attempt')
    return entries


def read(runtime_path):
    path = Path(runtime_path)
    doc = json.loads(path.read_text())
    validate(doc)
    if sha(doc['plan_path']) != doc['plan_sha256']:
        raise ValueError('Planning registry hash changed')
    for entry in doc['entries'].values():
        cfg = Path(doc['repo']) / entry['config']
        if not matches_original_digest(cfg, entry['config_sha256']):
            raise ValueError('Config changed: ' + str(cfg))
    return doc


def transition(runtime_path, identity, state, *, attempt_id=None, job_id=None,
               checkpoint_sha256=None, session=None, note=None):
    if state not in VALID or state == 'planned':
        raise ValueError('Invalid transition target')
    if note is not None and len(note) > 1024:
        raise ValueError('Note too long')
    with locked(runtime_path):
        doc = read(runtime_path)
        if identity not in doc['entries']:
            raise KeyError(identity)
        entry = doc['entries'][identity]
        prior = entry['status']
        if state not in TRANSITIONS[prior]:
            raise ValueError(f'Illegal transition {prior} -> {state}')
        if state in ('submitted', 'resumed'):
            if not job_id or not JOB.fullmatch(job_id):
                raise ValueError('Submission requires Slurm job ID')
            if state == 'resumed' and not checkpoint_sha256:
                # Recovery of a pre-execution failure, explicitly marked as fresh.
                # The CLI separately verifies the frozen run has no session or checkpoint.
                if not (prior == 'failed' and note and
                        note.startswith('T04-06 fresh-retry-no-checkpoint ') and
                        entry['attempts'] and
                        entry['attempts'][-1].get('checkpoint_sha256') is None and
                        entry['attempts'][-1].get('parent_checkpoint_sha256') is None):
                    raise ValueError('Resume requires verified checkpoint SHA-256')
            if checkpoint_sha256 and not SHA.fullmatch(checkpoint_sha256):
                raise ValueError('Invalid checkpoint SHA')
            if any(a['slurm_job_id'] == job_id for e in doc['entries'].values() for a in e['attempts']):
                raise ValueError('Duplicate Slurm job identity')
            attempt_id = str(uuid.uuid4())
            entry['attempts'].append({'id': attempt_id, 'slurm_job_id': job_id,
                'status': state, 'session': session, 'parent_checkpoint_sha256': checkpoint_sha256,
                'created_at': now(), 'ended_at': None})
        else:
            if not attempt_id or entry['attempts'][-1]['id'] != attempt_id:
                raise ValueError('Transition requires latest attempt ID')
            current = entry['attempts'][-1]
            if current['status'] != prior:
                raise ValueError('Attempt state mismatch')
            if state in ('completed', 'ready_to_resume') and (not checkpoint_sha256 or not SHA.fullmatch(checkpoint_sha256)):
                raise ValueError('Completed requires verified checkpoint SHA-256')
            current['status'] = state
            if state in ('completed', 'failed', 'ready_to_resume'):
                current['ended_at'] = now()
                if checkpoint_sha256:
                    current['checkpoint_sha256'] = checkpoint_sha256
                if session:
                    current['session'] = session
        entry['status'] = state
        entry['events'].append({'at': now(), 'event': state, 'attempt_id': attempt_id,
                                 'job_id': job_id, 'note': note})
        validate(doc)
        atomic_replace(runtime_path, doc)
        return attempt_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, default=Path('training_output/T04/t04_06/registry.runtime.json'))
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('bootstrap')
    p.add_argument('--repo', type=Path, default=Path.cwd())
    p.add_argument('--plan', type=Path, default=Path('training_output/T04/t04_06/registry.json'))
    p.add_argument('--commit', required=True)
    sub.add_parser('verify')
    sub.add_parser('summary')
    p = sub.add_parser('transition')
    p.add_argument('--id', required=True)
    p.add_argument('--to', required=True, choices=sorted(VALID - {'planned'}))
    p.add_argument('--attempt-id')
    p.add_argument('--job-id')
    p.add_argument('--checkpoint-sha256')
    p.add_argument('--session')
    p.add_argument('--note')
    args = parser.parse_args()
    if args.command == 'bootstrap':
        print('BOOTSTRAPPED', bootstrap(args.repo,args.plan,args.runtime,args.commit), 'planned entries')
    elif args.command == 'transition':
        print('ATTEMPT_ID', transition(args.runtime,args.id,args.to,
              attempt_id=args.attempt_id,job_id=args.job_id,
              checkpoint_sha256=args.checkpoint_sha256,session=args.session,note=args.note))
    else:
        doc = read(args.runtime)
        counts = {k: sum(x['status'] == k for x in doc['entries'].values()) for k in sorted(VALID)}
        print('PASS: operational registry verified; counts=' + json.dumps(counts,sort_keys=True))

if __name__ == '__main__':
    main()
