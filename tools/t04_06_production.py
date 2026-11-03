"""T04-06 Stage 3E: fail-closed sequential waves of held E5 seed42 jobs.

No automatic release, retry, requeue, config mutation, or second-wave
submission while a prior wave is not fully reconciled.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from tools.t04_06_registry import locked, read, atomic_replace
from tools.t04_06_gate import dry_run, prepare_one, submit_held, intent
from tools.t04_06_launcher import selected
from tools.t04_06_reconcile import query_sacct

DEFAULT = Path('training_output/T04/t04_06/registry.runtime.json')
FIRST = 'clip_s2_e5_seed42'


def candidates(doc):
    return [e for e in selected(doc) if e['id'] != FIRST]


def records(runtime):
    doc = read(runtime)
    return doc, candidates(doc)


def assert_prior_terminal(doc, tasks):
    """Conservative gate: do not overlap waves or unverified tasks.

    Every prior-wave task must be in ready_to_resume/completed and Slurm must
    independently confirm terminal successful status. A failed attempt is
    NOT auto-retried or mistaken for capacity.
    """
    prior = [e for e in tasks if e['status'] != 'planned' or e['attempts']]
    if not prior:
        return
    for e in prior:
        if e['status'] not in ('ready_to_resume', 'completed') or not e['attempts']:
            raise RuntimeError('Previous wave not reconciled: '+e['id']+' '+e['status'])
    ids = [e['attempts'][-1]['slurm_job_id'] for e in prior]
    observed = query_sacct(ids)
    for e in prior:
        job = e['attempts'][-1]['slurm_job_id']
        if observed.get(job) != {'state':'COMPLETED', 'exit_code':'0:0'}:
            raise RuntimeError('Previous job Slurm status ambiguous: '+job)


def plan(runtime, count):
    if type(count) is not int or not 1 <= count <= 4:
        raise ValueError('Wave size must be 1..4')
    doc, tasks = records(runtime)
    if doc['entries'][FIRST]['status'] not in ('ready_to_resume','completed'):
        raise RuntimeError('First accepted CLIP S2 experiment must be terminal partial or completed')
    assert_prior_terminal(doc, tasks)
    remaining = [e for e in tasks if e['status'] == 'planned' and not e['attempts']]
    for e in remaining:
        if intent(runtime,e['id']).exists():
            raise RuntimeError('Unresolved submission intent: '+e['id'])
    batch = remaining[:count]
    for e in batch:
        print('CANDIDATE', e['id'], e['run'], flush=True)
    print('WAVE_PLAN', len(batch), 'of',len(remaining),'remaining; no mutations')
    return batch


def stage(runtime, count):
    """Reserve runs for at most count experiments; no Slurm submission."""
    runtime = Path(runtime).resolve()
    with locked(runtime.parent/'stage3e.wave'):
        batch = plan(runtime,count)
        if not batch:
            raise ValueError('Nothing left to prepare')
        for e in batch:
            dry_run(runtime,e['id'])
        for e in batch:
            prepare_one(runtime,e['id'])
        print('PREPARED_WAVE',len(batch),'no Slurm submissions')


def submit_wave(runtime, count, script):
    """At most four HELD jobs. Never auto-releases.

    Each job uses existing durable per-identity intent fencing. If a partial
    sbatch/registry update fails, operator must inspect rather than retry.
    """
    runtime = Path(runtime).resolve()
    with locked(runtime.parent/'stage3e.wave'):
        batch = plan(runtime,count)
        if not batch:
            raise ValueError('No pending jobs')
        for e in batch:
            # Fail before first sbatch if any run is missing or provenance changed.
            from train_utils.t04_preflight import verify
            run = Path(e['run'])
            if not run.exists():
                raise RuntimeError('Run needs prepare-wave before submit: '+e['id'])
            verify(run)
            frozen = run/'source/config.original.yml'
            if frozen.read_bytes() != (Path(read(runtime)['repo'])/e['config']).read_bytes():
                raise RuntimeError('Frozen config differs: '+e['id'])
        for e in batch:
            submit_held(runtime,e['id'],script)
        print('HELD_WAVE',len(batch),'jobs; inspect ownership before manual release')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['plan','prepare-wave','submit-held-wave','status'])
    p.add_argument('--runtime',type=Path,default=DEFAULT)
    p.add_argument('--count',type=int,default=3)
    p.add_argument('--script',type=Path,default=Path('slurm/t04_06_single.slurm'))
    args=p.parse_args()
    runtime=args.runtime.resolve()
    if args.command=='plan': plan(runtime,args.count)
    elif args.command=='prepare-wave': stage(runtime,args.count)
    elif args.command=='submit-held-wave': submit_wave(runtime,args.count,args.script)
    else:
        _,tasks=records(runtime)
        for e in tasks:
            print(e['id'],e['status'],e['attempts'][-1]['slurm_job_id'] if e['attempts'] else '-')

if __name__=='__main__': main()
