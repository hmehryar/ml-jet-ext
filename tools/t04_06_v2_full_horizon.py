"""T04-06 v2: independent fixed-budget held-only production wave launcher.

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
from tools.t04_06_v2_reconcile import query_sacct, evidence, reconcile
from train_utils.t04_preflight import preflight, prepare, verify
from train_utils.t04_sessions import latest_checkpoint

from tools.t04_06_v2_matrix import PLAN, RUNTIME, OUTPUT, KIND
from tools.t04_06_slurm_capacity import inspect as inspect_capacity
LEGACY_RUNTIME = Path('training_output/T04/t04_06/registry.runtime.json')
from train_utils.t04_training_termination import validate_protocol

DEFAULT = RUNTIME
MAX_ACTIVE = 13
MAX_SEGMENT = 80
SAFE_ID = re.compile(r'^[a-z0-9_]+$')


PROFILE_ONE = 'cpu1_workers0_v1'
PROFILE_TWO = 'frozen_config_cpu2_v1'

def _profile(item):
    run = Path(item['run'])
    source = run / 'source/train_utils/t04_session_worker.py'
    if item['mode'] == 'fresh' and not run.exists():
        return PROFILE_ONE
    if not source.is_file():
        raise RuntimeError('Frozen worker missing: '+str(run))
    return PROFILE_ONE if 'def resolve_dataloader_workers(' in source.read_text() else PROFILE_TWO

def _active_profile(entry):
    for event in reversed(entry.get('events', [])):
        marker = 'resource_profile='
        note = event.get('note') or ''
        if marker in note:
            profile = note.split(marker, 1)[1].split()[0]
            if profile not in (PROFILE_ONE, PROFILE_TWO):
                raise RuntimeError('Unknown registered profile: '+profile)
            return profile
    return PROFILE_TWO  # conservative historical reservation

def _cost(profile):
    if profile not in (PROFILE_ONE, PROFILE_TWO):
        raise RuntimeError('Unknown profile')
    return (2 if profile == PROFILE_TWO else 1, 12 * 1024, 1)

def _reserve(resources, profile):
    cost = _cost(profile)
    if any(resources[i] < cost[i] for i in range(3)):
        return False
    for i in range(3):resources[i] -= cost[i]
    return True

def _check_script(path, profile):
    path = Path(path)
    if not path.is_file():raise FileNotFoundError(path)
    contents=path.read_text()
    expected = '#SBATCH --cpus-per-task=' + ('1' if profile == PROFILE_ONE else '2')
    required=(expected,'#SBATCH --mem=12G',
              '#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_1g.10gb:1',
              'OMP_NUM_THREADS=1','MKL_NUM_THREADS=1')
    if any(x not in contents for x in required):
        raise RuntimeError('Slurm profile mismatch: '+str(path))
    marker='T04_EXECUTION_PROFILE=cpu1_workers0_v1 T04_DATALOADER_WORKERS=0'
    if (marker in contents) != (profile == PROFILE_ONE):
        raise RuntimeError('Worker profile mismatch: '+str(path))
    subprocess.run(['bash','-n',str(path)],check=True)

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
    if config.training_protocol != 'fixed_budget_v2':
        raise RuntimeError('Not a fixed-budget v2 configuration: ' + entry['id'])
    validate_protocol(config.training_protocol, config.training_protocol_hash, config.epochs)
    if Path(config.output_root) != Path(OUTPUT):
        raise RuntimeError('Unexpected v2 output root: ' + entry['id'])
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
                'target':min(segment_epochs, config.epochs),
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
        'stage3f_v2.' + identity + '.from-' + str(item['from_epoch']) + '.intent.json')


def active_count(doc):
    active = [e for e in doc['entries'].values()
              if e['status'] in ('submitted','running','resumed')]
    if len(active) > MAX_ACTIVE:
        raise RuntimeError('Existing active slots exceed bound')
    # Never infer free capacity from squeue disappearance: only reconciled
    # registry state frees a slot, which also verifies checkpoint provenance.
    return len(active)


def _require_v2(doc, runtime):
    repo = Path(doc['repo']).resolve()
    if Path(doc['plan_path']).resolve() != (repo / PLAN).resolve():
        raise RuntimeError('Runtime registry is not bound to the independent v2 plan')
    plan_doc = json.loads(Path(doc['plan_path']).read_text())
    if plan_doc.get('kind') != KIND or plan_doc.get('training_protocol') != 'fixed_budget_v2':
        raise RuntimeError('Unexpected plan identity/protocol')
    if Path(runtime).resolve() != (repo / RUNTIME).resolve():
        raise RuntimeError('Refusing noncanonical v2 runtime path')

def plan(runtime, *, scope='e5', count=4, segment_epochs=80):
    if scope not in ('e5','all') or type(count) is not int or not 1 <= count <= MAX_ACTIVE:
        raise ValueError('Invalid scope/count; maximum concurrent jobs is 13')
    if type(segment_epochs) is not int or not 1 <= segment_epochs <= MAX_SEGMENT:
        raise ValueError('segment_epochs must be 1..80')
    doc = read(runtime)
    _require_v2(doc, runtime)
    active = active_count(doc)
    capacity = inspect_capacity(cpus_per_job=1, mem_gib=12)
    resources=[capacity['free_cpus'],capacity['free_memory_mib'],capacity['free_mig_gpus']]
    # Conservative reservation of registered jobs, including held jobs.
    for e in doc['entries'].values():
        if e['status'] in ('submitted','running','resumed'):
            cost=_cost(_active_profile(e))
            for i in range(3): resources[i] -= cost[i]
    slots = min(count, MAX_ACTIVE-active)
    candidates = sorted((e for e in doc['entries'].values() if _eligible(e,scope)),key=_order)
    items = []
    for entry in candidates:
        if len(items) >= slots:
            break
        item = _target(entry, doc['repo'], segment_epochs)
        item['resource_profile'] = _profile(item)
        if not all(resources[i] >= _cost(item['resource_profile'])[i] for i in range(3)):
            continue
        if fence(runtime,item).exists():
            raise RuntimeError('Unresolved/reused submission intent: '+str(fence(runtime,item)))
        _reserve(resources, item['resource_profile'])
        items.append(item)
    return doc, items


def prepare_wave(runtime, scope, count, segment_epochs):
    """Reserve only new planned experiments; never create new resume runs."""
    with locked(Path(runtime).resolve().parent/'stage3f_v2.orchestrator'):
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
    with locked(Path(runtime).resolve().parent/'stage3f_v2.orchestrator'):
        doc, items=plan(runtime,scope=scope,count=count,segment_epochs=segment_epochs)
        if not items:
            raise RuntimeError('No reconciled capacity or eligible experiments')
        legacy_path=Path(doc['repo']) / LEGACY_RUNTIME
        if legacy_path.is_file():
            legacy=read(legacy_path)
            if active_count(legacy):
                raise RuntimeError('Legacy jobs still active; do not exceed shared four-GPU concurrency')
        script=Path(script).resolve()
        if script != (Path(doc['repo'])/'slurm/t04_06_v2_single.slurm').resolve():
            raise RuntimeError('Refusing noncanonical v2 Slurm script')
        if not script.is_file():
            raise FileNotFoundError(script)
        legacy_script = (Path(doc['repo'])/'slurm/t04_06_v2_legacy_resume.slurm').resolve()
        _check_script(script, PROFILE_ONE)
        _check_script(legacy_script, PROFILE_TWO)
        for item in items:
            run=Path(item['run'])
            if not run.is_dir():
                raise RuntimeError('Missing prepared run; run prepare-wave first: '+item['id'])
            verify(run)
            if item['mode']=='fresh' and ((run/'sessions').exists() or latest_checkpoint(run)):
                raise RuntimeError('Fresh run was already executed')
        for item in items:
            selected_script = (legacy_script if item['resource_profile'] == PROFILE_TWO else script)
            path=fence(runtime,item)
            # Fence is persisted BEFORE any external scheduler side effect.
            atomic_replace(path, {'schema':1,'phase':'before_sbatch',**item,'job_id':None})
            command=['sbatch','--parsable','--hold',
                '--export=ALL,T04_06_RUN='+item['run']+',T04_06_MODE='+item['mode']+',T04_06_TARGET='+str(item['target']),
                str(selected_script)]
            try:
                result=subprocess.run(command,cwd=doc['repo'],check=True,text=True,capture_output=True)
                job=result.stdout.strip().split(';')[0]
                if not re.fullmatch(r'\d+',job):
                    raise RuntimeError('Unexpected sbatch response')
                atomic_replace(path,{'schema':1,'phase':'held_submitted',**item,'job_id':job})
                state='resumed' if item['mode']=='resume' else 'submitted'
                transition(runtime,item['id'],state,job_id=job,
                    checkpoint_sha256=item['parent_sha256'],
                    note='T04-06 fixed_budget_v2 held segment to epoch '+str(item['target'])+' resource_profile='+item['resource_profile'])
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
    p.add_argument('--segment-epochs',type=int,default=80)
    p.add_argument('--script',type=Path,default=Path('slurm/t04_06_v2_single.slurm'))
    args=p.parse_args()
    runtime=args.runtime.resolve()
    if args.command in ('reconcile-inspect','reconcile-apply'):
        _require_v2(read(runtime), runtime)
        reports=reconcile(runtime,apply=args.command=='reconcile-apply')
        for report in reports: print(json.dumps(report,sort_keys=True))
        print('RECONCILED',len(reports),'entries',args.command)
    elif args.command=='status':
        doc=read(runtime)
        _require_v2(doc, runtime)
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
