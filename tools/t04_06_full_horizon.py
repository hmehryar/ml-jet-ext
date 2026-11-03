"""T04-06 Stage 3F/1: bounded, auditable full-matrix production wave launcher.

All launches are HELD and require an explicit human Slurm release. No resetting
of runtime state, no auto retry, no automatic release, no test-set evaluation.
Use `cycle` after reconciling a wave to propose the next bounded wave.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess

from tools.t04_06_registry import read, locked, atomic_replace, transition, sha
from tools.t04_06_reconcile import query_sacct, evidence, reconcile
from train_utils.t04_preflight import preflight, prepare, verify
from train_utils.t04_sessions import latest_checkpoint

DEFAULT = Path('training_output/T04/t04_06/registry.runtime.json')
MAX_ACTIVE = 4
MAX_SEGMENT = 24
SAFE_ID = re.compile(r'^[a-z0-9_]+$')


def _eligible(entry, scope):
    if entry['status'] not in ('planned', 'ready_to_resume'):
        return False
    if scope == 'e5' and '_e5_' not in entry['id']:
        return False
    return True


def _order(entry):
    # Finish the previously accepted seed42 runs first; then deterministic matrix.
    priority = 0 if entry['status'] == 'ready_to_resume' else 1
    return (priority, entry['id'])


def _latest_job(entry):
    return entry['attempts'][-1]['slurm_job_id'] if entry['attempts'] else None


def _target(entry, repo, segment_epochs):
    cfg_path = Path(repo) / entry['config']
    run = Path(entry['run'])
    if sha(cfg_path) != entry['config_sha256']:
        raise RuntimeError('Registry config hash differs: ' + entry['id'])
    config, _ = preflight(cfg_path)
    if config.epochs != 80 or config.min_epochs != 12 or config.patience != 16:
        raise RuntimeError('Unexpected production training policy: ' + entry['id'])
    if entry['status'] == 'planned':
        if entry['attempts']:
            raise RuntimeError('Planned entry has previous attempts: ' + entry['id'])
        if run.exists():
            verify(run)
            if (run/'source/config.original.yml').read_bytes() != cfg_path.read_bytes():
                raise RuntimeError('Prepared config mismatch')
            if (run/'sessions').exists() or latest_checkpoint(run) is not None:
                raise RuntimeError('Planned experiment already has session/checkpoint')
        # Always satisfy min_epochs even if the segment size is smaller.
        return {'id':entry['id'],'mode':'fresh','from_epoch':0,
                'target':max(config.min_epochs, min(segment_epochs, config.epochs)),
                'run':str(run),'parent_sha256':None}

    if not run.exists():
        raise RuntimeError('Ready-to-resume run missing: '+entry['id'])
    verify(run)
    if (run/'source/config.original.yml').read_bytes() != cfg_path.read_bytes():
        raise RuntimeError('Frozen configuration mismatch: '+entry['id'])
    ptr = latest_checkpoint(run)
    if ptr is None or not entry['attempts']:
        raise RuntimeError('Resume lacks verified checkpoint/attempt')
    parent = entry['attempts'][-1]
    if parent.get('checkpoint_sha256') != ptr['sha256'] or parent.get('session') != ptr['session']:
        raise RuntimeError('Parent SHA/session differ from registry: '+entry['id'])
    if ptr['epoch'] >= config.epochs:
        raise RuntimeError('Experiment already at maximum horizon: '+entry['id'])
    obs = query_sacct([parent['slurm_job_id']]).get(parent['slurm_job_id'])
    if obs != {'state':'COMPLETED','exit_code':'0:0'}:
        raise RuntimeError('Parent Slurm attempt not terminal-successful: '+entry['id'])
    proof = evidence(entry)
    if not proof.get('resume_eligible') or proof.get('epoch') != ptr['epoch']:
        raise RuntimeError('Parent is not verified partial segment: '+entry['id'])
    # Verify serialized optimizer, scheduler, epoch and RNG, not just pointer SHA.
    from train_utils.t04_resume_state import load_checkpoint
    state = load_checkpoint(ptr['absolute_path'], map_location='cpu',
            manifest_sha256=sha(run/'run_manifest.json'),
            config_sha256=sha(run/'config.resolved.json'))
    if state['epoch'] != ptr['epoch'] or state.get('stopped'):
        raise RuntimeError('Checkpoint already stopped or epoch mismatch')
    if state.get('optimizer_state_dict') is None or not state.get('rng_state'):
        raise RuntimeError('Missing optimizer/RNG checkpoint state')
    if config.scheduler and state.get('scheduler_state_dict') is None:
        raise RuntimeError('Missing scheduler checkpoint state')
    if config.precision == 'fp16' and state.get('scaler_state_dict') is None:
        raise RuntimeError('Missing FP16 scaler state')
    return {'id':entry['id'],'mode':'resume','from_epoch':ptr['epoch'],
            'target':min(config.epochs, ptr['epoch'] + segment_epochs),
            'run':str(run), 'parent_sha256':ptr['sha256']}


def fence(runtime, item):
    """Per-experiment, per-session submission intent; never reuse for a retry."""
    identity = item['id']
    if not SAFE_ID.fullmatch(identity):
        raise RuntimeError('Unsafe identity')
    return Path(runtime).resolve().parent / (
        'stage3f.' + identity + '.from-' + str(item['from_epoch']) + '.intent.json')


def active_count(doc):
    active = [e for e in doc['entries'].values()
              if e['status'] in ('submitted','running','resumed')]
    if len(active) > MAX_ACTIVE:
        raise RuntimeError('Existing active slots exceed bound')
    # Never infer free capacity from squeue disappearance: only reconciled
    # registry state frees a slot, which also verifies checkpoint provenance.
    return len(active)


def plan(runtime, *, scope='e5', count=4, segment_epochs=24):
    if scope not in ('e5','all') or type(count) is not int or not 1 <= count <= MAX_ACTIVE:
        raise ValueError('Invalid scope/count; max concurrent jobs is 4')
    if type(segment_epochs) is not int or not 1 <= segment_epochs <= MAX_SEGMENT:
        raise ValueError('segment_epochs must be 1..24')
    doc = read(runtime)
    active = active_count(doc)
    slots = min(count, MAX_ACTIVE-active)
    candidates = sorted((e for e in doc['entries'].values() if _eligible(e,scope)),key=_order)
    items = []
    for entry in candidates:
        if len(items) >= slots:
            break
        item = _target(entry, doc['repo'], segment_epochs)
        if fence(runtime,item).exists():
            raise RuntimeError('Unresolved/reused submission intent: '+str(fence(runtime,item)))
        items.append(item)
    return doc, items


def prepare_wave(runtime, scope, count, segment_epochs):
    """Reserve only new planned experiments; never create new resume runs."""
    with locked(Path(runtime).resolve().parent/'stage3f.orchestrator'):
        doc, items = plan(runtime,scope=scope,count=count,segment_epochs=segment_epochs)
        for item in items:
            if item['mode']=='fresh':
                entry=doc['entries'][item['id']]
                cfg=Path(doc['repo'])/entry['config']
                run=Path(item['run'])
                if not run.exists():
                    actual=Path(prepare(cfg,run.name)).resolve()
                    if actual != run.resolve():
                        raise RuntimeError('Prepared output differs from registry')
                verify(run)
                print('PREPARED',item['id'],run)
            else:
                print('PRESERVED',item['id'],'epoch',item['from_epoch'])
        print('PREPARED_WAVE',len(items),'no Slurm submissions')


def submit_wave(runtime,scope,count,segment_epochs,script):
    with locked(Path(runtime).resolve().parent/'stage3f.orchestrator'):
        doc, items=plan(runtime,scope=scope,count=count,segment_epochs=segment_epochs)
        if not items:
            raise RuntimeError('No reconciled capacity or eligible experiments')
        script=Path(script).resolve()
        if not script.is_file():
            raise FileNotFoundError(script)
        # Pin exactly two CPUs and thread settings before creating an intent.
        script_text=script.read_text()
        if '#SBATCH --cpus-per-task=2' not in script_text or 'OMP_NUM_THREADS=1' not in script_text or 'MKL_NUM_THREADS=1' not in script_text:
            raise RuntimeError('Slurm resource profile differs from accepted 2-CPU configuration')
        subprocess.run(['bash','-n',str(script)],check=True)
        for item in items:
            run=Path(item['run'])
            if not run.is_dir():
                raise RuntimeError('Missing prepared run; run prepare-wave first: '+item['id'])
            verify(run)
            if item['mode']=='fresh' and ((run/'sessions').exists() or latest_checkpoint(run)):
                raise RuntimeError('Fresh run was already executed')
        for item in items:
            path=fence(runtime,item)
            # Fence is persisted BEFORE any external scheduler side effect.
            atomic_replace(path, {'schema':1,'phase':'before_sbatch',**item,'job_id':None})
            command=['sbatch','--parsable','--hold',
                '--export=ALL,T04_06_RUN='+item['run']+',T04_06_MODE='+item['mode']+',T04_06_TARGET='+str(item['target']),
                str(script)]
            try:
                result=subprocess.run(command,cwd=doc['repo'],check=True,text=True,capture_output=True)
                job=result.stdout.strip().split(';')[0]
                if not re.fullmatch(r'\d+',job):
                    raise RuntimeError('Unexpected sbatch response')
                atomic_replace(path,{'schema':1,'phase':'held_submitted',**item,'job_id':job})
                state='resumed' if item['mode']=='resume' else 'submitted'
                transition(runtime,item['id'],state,job_id=job,
                    checkpoint_sha256=item['parent_sha256'],
                    note='T04-06 Stage 3F held segment to epoch '+str(item['target']))
                atomic_replace(path, {'schema':1,'phase':'registered_held',**item,'job_id':job})
                print('HELD_REGISTERED',job,item['id'],item['mode'],
                      'epochs',str(item['from_epoch'])+'->'+str(item['target']),flush=True)
            except Exception:
                print('SUBMISSION_UNCERTAIN',item['id'],path,flush=True)
                raise
        print('HELD_WAVE',len(items),'manual verification/release required')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('plan','prepare-wave','submit-held-wave','reconcile-inspect','reconcile-apply','status'))
    p.add_argument('--runtime',type=Path,default=DEFAULT)
    p.add_argument('--scope',choices=('e5','all'),default='e5')
    p.add_argument('--count',type=int,default=4)
    p.add_argument('--segment-epochs',type=int,default=24)
    p.add_argument('--script',type=Path,default=Path('slurm/t04_06_single.slurm'))
    args=p.parse_args()
    runtime=args.runtime.resolve()
    if args.command in ('reconcile-inspect','reconcile-apply'):
        reports=reconcile(runtime,apply=args.command=='reconcile-apply')
        for report in reports: print(json.dumps(report,sort_keys=True))
        print('RECONCILED',len(reports),'entries',args.command)
    elif args.command=='status':
        doc=read(runtime)
        from collections import Counter
        print('STATES',json.dumps(dict(Counter(e['status'] for e in doc['entries'].values())),sort_keys=True))
        print('ACTIVE_SLOTS',active_count(doc),'/',MAX_ACTIVE)
    elif args.command=='plan':
        _, items=plan(runtime,scope=args.scope,count=args.count,segment_epochs=args.segment_epochs)
        for x in items:
            print('CANDIDATE',x['id'],x['mode'],'epoch',x['from_epoch'],'->',x['target'])
        print('WAVE_PLAN',len(items),'held jobs maximum; no mutations')
    elif args.command=='prepare-wave':
        prepare_wave(runtime,args.scope,args.count,args.segment_epochs)
    else:
        submit_wave(runtime,args.scope,args.count,args.segment_epochs,args.script)


if __name__=='__main__':
    main()
