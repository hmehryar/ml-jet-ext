"""T04-06 v2 user-sized Slurm queues on A100 MIG or H200.

All submit operations use --hold, shared orchestrator lock, exclusive fences,
immutable prepared runs, and the canonical 120-entry operational registry.
The legacy capacity-aware CSC1 wave planner is intentionally not used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess

from tools.t04_06_registry import read, locked, atomic_replace, transition, sha
from tools.t04_06_v2_matrix import RUNTIME, PLAN, OUTPUT
from train_utils.t04_preflight import prepare, verify
from train_utils.t04_sessions import latest_checkpoint

SAFE = re.compile(r'^[a-z0-9][a-z0-9_-]{0,79}$')
JOB = re.compile(r'^\d+$')
POOLS = {
    'a100': {'script': 'slurm/t04_06_v2_queued_a100.slurm',
             'partition': 'ecscp', 'qos': 'express',
             'gres': 'nvidia_a100_80gb_pcie_1g.10gb', 'mem': '12G',
             'time': '3-00:00:00'},
    'h200': {'script': 'slurm/t04_06_v2_queued_h200.slurm',
             'partition': 'gmsap', 'qos': 'gpu',
             'gres': 'nvidia_h200', 'mem': '10G',
             'time': '06:00:00'},
}
ACTIVE = {'submitted','running','resumed'}
QUEUED = Path('training_output/T04/t04_06_v2/queues')
ORCHESTRATOR = Path('training_output/T04/t04_06_v2/stage3f_v2.orchestrator')


def _repo():
    repo=Path.cwd().resolve()
    if not (repo/'tools/t04_06_v2_full_horizon.py').is_file():
        raise RuntimeError('Run from ml-jet-ext repository root')
    return repo


def _doc(repo):
    runtime=repo/RUNTIME
    doc=read(runtime)
    if Path(doc['repo']).resolve()!=repo or Path(doc['plan_path']).resolve()!=(repo/PLAN).resolve():
        raise RuntimeError('Registry belongs to another repository or plan')
    if doc.get('schema')!=1 or len(doc['entries'])!=120:
        raise RuntimeError('Unexpected production registry')
    return doc


def _batch_file(repo, batch):
    if not SAFE.fullmatch(batch):
        raise ValueError('Invalid batch ID (lowercase letters, digits, dash, underscore)')
    return repo/QUEUED/(batch+'.json')


def _all_batches(repo):
    directory=repo/QUEUED
    for p in sorted(directory.glob('*.json')):
        yield json.loads(p.read_text())


def _load(repo, batch):
    doc=json.loads(_batch_file(repo,batch).read_text())
    if doc.get('schema')!=1 or doc.get('batch')!=batch or doc.get('gpu') not in POOLS:
        raise RuntimeError('Invalid queued batch')
    ids=doc.get('experiment_ids')
    if not isinstance(ids,list) or len(ids)!=doc.get('count') or len(ids)!=len(set(ids)):
        raise RuntimeError('Corrupt batch selection')
    return doc


def _fence(repo, identity):
    if not SAFE.fullmatch(identity):
        raise ValueError('Unsafe experiment identity')
    # Same fence name used by original CSC1 launcher: cross-launcher protection.
    return repo/RUNTIME.parent/('stage3f_v2.'+identity+'.from-0.intent.json')


def _script(repo, gpu):
    pool=POOLS[gpu]
    path=(repo/pool['script']).resolve()
    text=path.read_text()
    required=['#SBATCH --partition='+pool['partition'],
              '#SBATCH --qos='+pool['qos'],
              '#SBATCH --gres=gpu:'+pool['gres']+':1',
              '#SBATCH --cpus-per-task=1', '#SBATCH --mem='+pool['mem'],
              '#SBATCH --time='+pool['time'],
              'T04_EXECUTION_PROFILE=cpu1_workers0_v1',
              'T04_DATALOADER_WORKERS=0',
              'tools.t04_04_pilot execute']
    if any(x not in text for x in required):
        raise RuntimeError('Slurm script does not match pinned pool profile: '+str(path))
    subprocess.run(['bash','-n',str(path)],check=True)
    return path


def create(repo, batch, gpu, count):
    if gpu not in POOLS or not 1<=count<=120:
        raise ValueError('Expected --gpu a100|h200 and --count 1..120')
    path=_batch_file(repo,batch)
    with locked(repo/ORCHESTRATOR):
        doc=_doc(repo)
        if path.exists():
            raise FileExistsError('Batch ID already exists; do not replace it')
        reserved={i for b in _all_batches(repo) for i in b['experiment_ids']}
        candidates=sorted(i for i,e in doc['entries'].items()
                          if e['status']=='planned' and i not in reserved)
        if count>len(candidates):
            raise RuntimeError(f'Only {len(candidates)} unreserved planned experiments available')
        selected=candidates[:count]
        # Fail closed if an unresolved historic fence already exists.
        for ident in selected:
            if _fence(repo,ident).exists():
                raise RuntimeError('Experiment has existing submission intent: '+ident)
        _script(repo,gpu)
        path.parent.mkdir(parents=True,exist_ok=True)
        body={'schema':1,'batch':batch,'gpu':gpu,'count':count,
              'experiment_ids':selected,'plan_sha256':doc['plan_sha256'],
              'profile':POOLS[gpu]}
        with path.open('x') as f:
            json.dump(body,f,indent=2,sort_keys=True)
            f.write('\n')
        print('BATCH_CREATED',batch,'gpu='+gpu,'count='+str(count))
        for ident in selected:
            print('SELECTED',ident)


def _check_binding(repo, batch, doc):
    if batch['plan_sha256']!=doc['plan_sha256']:
        raise RuntimeError('Batch no longer matches registry plan')
    if batch['profile']!=POOLS[batch['gpu']]:
        raise RuntimeError('Queued GPU profile changed')
    _script(repo,batch['gpu'])
    for ident in batch['experiment_ids']:
        if ident not in doc['entries']:
            raise RuntimeError('Unknown registry experiment: '+ident)


def _prepared(repo, entry):
    run=Path(entry['run']).resolve()
    config=repo/entry['config']
    if sha(config)!=entry['config_sha256']:
        raise RuntimeError('Immutable config changed: '+entry['id'])
    if not run.is_dir():
        raise RuntimeError('Run not prepared: '+entry['id'])
    verify(run)
    if (run/'source/config.original.yml').read_bytes()!=config.read_bytes():
        raise RuntimeError('Frozen config mismatch: '+entry['id'])
    if (run/'sessions').exists() or latest_checkpoint(run) is not None:
        raise RuntimeError('Fresh run has prior execution: '+entry['id'])
    cfg=json.loads((run/'config.resolved.json').read_text())
    if cfg.get('training_protocol')!='fixed_budget_v2' or cfg['epochs']!=80:
        raise RuntimeError('Not an 80-epoch v2 run: '+entry['id'])
    if Path(cfg['output_root'])!=Path(OUTPUT):
        raise RuntimeError('Wrong production output: '+entry['id'])
    frozen=run/'source/train_utils/t04_session_worker.py'
    if 'def resolve_dataloader_workers(' not in frozen.read_text():
        raise RuntimeError('Frozen worker does not support 1CPU/0worker profile: '+entry['id'])
    return run


def prepare_batch(repo, batch_id):
    with locked(repo/ORCHESTRATOR):
        batch=_load(repo,batch_id)
        doc=_doc(repo)
        _check_binding(repo,batch,doc)
        for ident in batch['experiment_ids']:
            entry=doc['entries'][ident]
            if entry['status']!='planned':
                raise RuntimeError('Cannot prepare non-planned experiment: '+ident)
            run=Path(entry['run']).resolve()
            cfg=repo/entry['config']
            if sha(cfg)!=entry['config_sha256']:
                raise RuntimeError('Config hash mismatch: '+ident)
            if not run.exists():
                actual=prepare(cfg,run.name)
                if actual.resolve()!=run:
                    raise RuntimeError('Prepared run path mismatch: '+ident)
            _prepared(repo,entry)
            print('PREPARED',ident,flush=True)
        print('BATCH_PREPARED',batch_id,len(batch['experiment_ids']))


def _job_record(repo, entry, batch_id, gpu):
    f=_fence(repo,entry['id'])
    if not f.exists():
        raise RuntimeError('Missing submission fence for '+entry['id'])
    data=json.loads(f.read_text())
    if data.get('batch')!=batch_id or data.get('gpu')!=gpu:
        raise RuntimeError('Submission fence belongs to different queue: '+entry['id'])
    if data.get('phase')!='registered_held':
        raise RuntimeError('Unresolved submission, manual review required: '+str(f))
    jid=data.get('job_id')
    if not isinstance(jid,str) or not JOB.fullmatch(jid):
        raise RuntimeError('Bad submission job id for '+entry['id'])
    if entry['status'] not in ACTIVE|{'completed','failed','ready_to_resume'}:
        raise RuntimeError('Entry not submitted for '+entry['id'])
    if not entry['attempts'] or entry['attempts'][-1]['slurm_job_id']!=jid:
        raise RuntimeError('Fence does not match current registered Slurm job: '+entry['id'])
    return jid


def submit_held(repo, batch_id):
    with locked(repo/ORCHESTRATOR):
        batch=_load(repo,batch_id)
        doc=_doc(repo)
        _check_binding(repo,batch,doc)
        script=_script(repo,batch['gpu'])
        # Precheck all remaining runs BEFORE any Slurm calls.
        for ident in batch['experiment_ids']:
            entry=doc['entries'][ident]
            if entry['status']=='planned':
                _prepared(repo,entry)
                if _fence(repo,ident).exists():
                    raise RuntimeError('Unresolved submission intent: '+str(_fence(repo,ident)))
            else:
                _job_record(repo,entry,batch_id,batch['gpu'])
        for ident in batch['experiment_ids']:
            entry=doc['entries'][ident]
            if entry['status']!='planned':
                print('ALREADY_REGISTERED',ident,_job_record(repo,entry,batch_id,batch['gpu']),flush=True)
                continue
            run=str(Path(entry['run']).resolve())
            path=_fence(repo,ident)
            base={'schema':1,'batch':batch_id,'gpu':batch['gpu'],
                  'experiment':ident,'mode':'fresh','target_epochs':80,'run':run}
            # No external side effect until write-ahead intent is durable.
            with path.open('x') as stream:
                json.dump(dict(base,phase='before_sbatch',job_id=None),stream,indent=2)
                stream.flush()
                import os
                os.fsync(stream.fileno())
            args=['sbatch','--parsable','--hold',
                  '--export=ALL,T04_06_RUN='+run+',T04_06_MODE=fresh,T04_06_TARGET=80',str(script)]
            try:
                result=subprocess.run(args,cwd=repo,text=True,capture_output=True,check=True,timeout=60)
                job=result.stdout.strip().split(';')[0]
                if not JOB.fullmatch(job):
                    raise RuntimeError('Unrecognized Slurm job ID: '+repr(result.stdout))
                atomic_replace(path,dict(base,phase='held_submitted',job_id=job))
                transition(repo/RUNTIME,ident,'submitted',job_id=job,
                           note=f'T04-06 v2 multi-queue batch={batch_id} gpu_pool={batch["gpu"]} resource_profile=cpu1_workers0_v1')
                atomic_replace(path,dict(base,phase='registered_held',job_id=job))
                print('HELD_REGISTERED',job,ident,'gpu='+batch['gpu'],flush=True)
            except Exception:
                print('SUBMISSION_UNCERTAIN',ident,'INTENT',path,flush=True)
                raise
        print('ALL_HELD_REGISTERED',batch_id,len(batch['experiment_ids']))


def _scontrol(job):
    result=subprocess.run(['scontrol','show','job','-o',job],text=True,capture_output=True,check=True,timeout=15)
    lines=result.stdout.strip().splitlines()
    if len(lines)!=1:
        raise RuntimeError('Unexpected scontrol response for '+job)
    return lines[0]


def _validate_job(repo, entry, batch, require_hold=True):
    job=_job_record(repo,entry,batch['batch'],batch['gpu'])
    line=_scontrol(job)
    pool=POOLS[batch['gpu']]
    expected=['JobId='+job+' ', 'Partition='+pool['partition'], 'QOS='+pool['qos'],
              'NumCPUs=1 ', 'cpu=1,mem='+pool['mem'], 'gres/gpu:'+pool['gres']+'=1',
              'Command='+str((repo/pool['script']).resolve())]
    if require_hold:
        expected.extend(['JobState=PENDING', 'Reason=JobHeldUser'])
    for part in expected:
        if part not in line:
            raise RuntimeError(f'Job {job} hardware/queue mismatch: missing {part}\n{line}')
    return job


def verify_held(repo, batch_id):
    batch=_load(repo,batch_id)
    doc=_doc(repo)
    _check_binding(repo,batch,doc)
    jobs=[]
    for ident in batch['experiment_ids']:
        jobs.append(_validate_job(repo,doc['entries'][ident],batch,True))
    print('VERIFIED_HELD',batch_id,'gpu='+batch['gpu'],'count='+str(len(jobs)))
    return jobs


def release(repo, batch_id):
    with locked(repo/ORCHESTRATOR):
        jobs=verify_held(repo,batch_id)
        for job in jobs:
            subprocess.run(['scontrol','release',job],check=True,timeout=15)
            print('RELEASED',job,flush=True)
        print('RELEASED_BATCH',batch_id,len(jobs))


def status(repo, batch_id):
    batch=_load(repo,batch_id)
    doc=_doc(repo)
    from collections import Counter
    counts=Counter()
    for ident in batch['experiment_ids']:
        entry=doc['entries'][ident]
        counts[entry['status']]+=1
        job=entry['attempts'][-1]['slurm_job_id'] if entry['attempts'] else '-'
        print(ident,entry['status'],job)
    print('BATCH_STATUS',batch_id,'gpu='+batch['gpu'],'counts='+json.dumps(dict(counts),sort_keys=True))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('create');p.add_argument('--gpu',choices=sorted(POOLS),required=True);p.add_argument('--count',type=int,required=True);p.add_argument('--batch',required=True)
    for action in ('prepare','submit-held','verify-held','release-verified','status'):
        p=sub.add_parser(action);p.add_argument('--batch',required=True)
    from tools.t04_06_v2_retry import register_cli, dispatch
    register_cli(sub)
    args=parser.parse_args(argv)
    repo=_repo()
    if args.command=='create':create(repo,args.batch,args.gpu,args.count)
    elif args.command=='prepare':prepare_batch(repo,args.batch)
    elif args.command=='submit-held':submit_held(repo,args.batch)
    elif args.command=='verify-held':verify_held(repo,args.batch)
    elif args.command=='release-verified':release(repo,args.batch)
    elif args.command.startswith('retry-'):dispatch(args)
    else:status(repo,args.batch)


if __name__=='__main__':
    main()
