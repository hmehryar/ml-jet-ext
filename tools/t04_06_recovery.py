"""T04-06 Stage 3C: read-only, fail-closed resume/recovery eligibility.

Never changes registry, run directory, sessions, checkpoint pointer or Slurm.
A positive report is advisory, NOT authorization to submit a duplicate job.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from tools.t04_06_registry import read, sha
from train_utils.t04_preflight import verify
from train_utils.t04_sessions import latest_checkpoint

ACTIVE_REGISTRY = {'submitted', 'running', 'resumed'}
ACTIVE_SESSION = {'reserved', 'running'}
ALLOWED_CLOSED_SESSION = {'completed', 'interrupted'}


def decision(entry, *, slurm_state=None, load_state=None):
    """Assess one experiment. load_state is injectable for CPU-only unit tests.

    A terminal scheduler observation must be supplied independently before
    permitting recovery from a failed job. Never assume no queued task exists.
    """
    outcome = dict(id=entry['id'], status=entry['status'], eligible=False,
                   action='block', reason=None, checkpoint_epoch=None,
                   checkpoint_sha256=None, target_epochs=None)
    def block(reason):
        outcome['reason'] = reason
        return outcome

    if entry['status'] in ACTIVE_REGISTRY:
        return block('active registry attempt: reconcile scheduler first')
    if entry['status'] == 'planned':
        return block('planned experiment: use fresh preparation, not resume')
    if entry['status'] not in ('failed', 'completed'):
        return block('unsupported registry state')
    if not entry.get('attempts'):
        return block('missing audited execution attempt')
    if slurm_state is None:
        return block('scheduler terminal state not independently confirmed')
    state = slurm_state.upper()
    if state not in ('COMPLETED','FAILED','TIMEOUT','CANCELLED','PREEMPTED',
                     'NODE_FAIL','OUT_OF_MEMORY'):
        return block('scheduler still active or ambiguous: '+state)

    run = Path(entry['run'])
    try:
        verify(run)
        cfg = json.loads((run/'config.resolved.json').read_text())
        outcome['target_epochs'] = cfg['epochs']
        if sha(run/'source/config.original.yml') != entry['config_sha256']:
            return block('prepared config does not match operational registry')
        pointer = latest_checkpoint(run)
        if pointer is None:
            return block('no valid committed checkpoint; no in-place fresh retry')
        epoch = pointer['epoch']
        outcome.update(checkpoint_epoch=epoch, checkpoint_sha256=pointer['sha256'])
        if epoch > cfg['epochs']:
            return block('checkpoint epoch exceeds training horizon')
        sessions_dir = run/'sessions'
        sessions = sorted(sessions_dir.glob('session-*')) if sessions_dir.is_dir() else []
        if not sessions:
            return block('no session audit history')
        audit = []
        for session in sessions:
            d = json.loads((session/'session_manifest.json').read_text())
            if d.get('session') != session.name:
                return block('session name mismatch: '+session.name)
            if d.get('status') in ACTIVE_SESSION:
                return block('unfinished reserved/running session requires manual audit: '+session.name)
            if d.get('status') not in ALLOWED_CLOSED_SESSION:
                return block('invalid session status: '+session.name)
            audit.append(d)
        # Last completed checkpoint must correspond to its claimed session.
        matching = [s for s in audit if s['session'] == pointer['session']]
        if len(matching) != 1:
            return block('checkpoint session not found')
        if matching[0]['status'] == 'completed':
            result = json.loads((sessions_dir/pointer['session']/'result.json').read_text())
            if result.get('epoch') != epoch:
                return block('completed session result epoch differs')
            if result.get('early_stopped'):
                return block('early stopping already terminated this trajectory')
        # Read checkpoint on CPU using the repository's validated schema.
        if load_state is None:
            from train_utils.t04_resume_state import load_checkpoint
            state_obj = load_checkpoint(pointer['absolute_path'], map_location='cpu',
                manifest_sha256=sha(run/'run_manifest.json'),
                config_sha256=sha(run/'config.resolved.json'))
        else:
            state_obj = load_state(pointer, run)
        if state_obj.get('epoch') != epoch:
            return block('checkpoint payload epoch mismatch')
        if state_obj.get('stopped'):
            return block('checkpoint says early-stopped; resume prohibited')
        if cfg['precision'] == 'fp16' and state_obj.get('scaler_state_dict') is None:
            return block('missing FP16 scaler state')
        if state_obj.get('optimizer_state_dict') is None or state_obj.get('scheduler_state_dict') is None:
            return block('missing optimizer or scheduler state')
        if not state_obj.get('rng_state'):
            return block('missing RNG state')
        metric_file = run/'moment_epoch_metrics.csv'
        with metric_file.open(newline='') as f:
            rows = list(csv.DictReader(f))
        epochs = [int(row['epoch']) for row in rows]
        # An absent last CSV row may be recoverable by the checkpoint restore
        # code, but manual reconciliation is required before launching.
        if epochs != list(range(1, epoch + 1)):
            return block('CSV epoch history incomplete/divergent; reconcile manually')
        if epoch == cfg['epochs']:
            return block('configured epoch horizon already completed')
        if entry['status'] == 'completed':
            return block('completed registry entry: explicit continuation policy required')
        outcome.update(eligible=True, action='review_resume', reason='valid checkpoint and closed session; manually authorize a new attempt')
        return outcome
    except Exception as exc:
        return block('evidence verification failed: '+str(exc)[:240])


def inspect(runtime, *, states=None, load_state=None):
    doc = read(runtime)
    states = states or {}
    return [decision(e, slurm_state=states.get(e['attempts'][-1]['slurm_job_id']) if e['attempts'] else None,
                     load_state=load_state) for e in doc['entries'].values()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['inspect'])
    parser.add_argument('--runtime', type=Path,
                        default=Path('training_output/T04/t04_06/registry.runtime.json'))
    parser.add_argument('--output', type=Path, help='New report path; refuses overwrite')
    args = parser.parse_args()
    # No sacct inference: production jobs require authenticated observation
    # supplied in the later Stage 3D ownership handshake.
    reports = inspect(args.runtime.resolve())
    for r in reports:
        if r['status'] != 'planned':
            print(json.dumps(r,sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as f:
            json.dump({'schema':1,'inspection_only':True,'entries':reports},f,indent=2)
            f.write('\n')
    eligible=sum(x['eligible'] for x in reports)
    print('RECOVERY_INSPECT',len(reports),'entries;',eligible,'eligible; no mutations')


if __name__ == '__main__':
    main()
