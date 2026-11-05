"""Controlled T04-06 recovery of pre-execution H200 failures on A100 MIG.

Standalone helpers wired into t04_06_v2_multi_queue's CLI. Never reset the
canonical registry, modify frozen runs, or re-use an earlier submission fence.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

from tools.t04_06_registry import read, transition, locked, atomic_replace
from tools.t04_06_v2_multi_queue import (
    _repo, _doc, _load, _prepared, _script, _fence, ORCHESTRATOR,
    POOLS, JOB, _scontrol,
)
from tools.t04_06_v2_matrix import RUNTIME

RECOVERY_PREFIX = 'stage3f_v2.retry.'
SCHEMA = 1


def _journal(repo: Path, identity: str, prior_job: str) -> Path:
    if not re.fullmatch(r'[a-z0-9_]+', identity) or not JOB.fullmatch(prior_job):
        raise ValueError('Bad experiment or parent job ID')
    return repo / RUNTIME.parent / (RECOVERY_PREFIX + identity + '.from-' + prior_job + '.intent.json')


def _sacct(jobs):
    if not jobs:
        return {}
    cmd = ['sacct','-X','-n','-P','-j',','.join(jobs),
           '--format=JobIDRaw,State,ExitCode,NodeList']
    r = subprocess.run(cmd,text=True,capture_output=True,check=True,timeout=30)
    result={}
    for line in r.stdout.splitlines():
        fields=line.strip().split('|')
        if len(fields)!=4 or fields[0] not in jobs or fields[0] in result:
            raise RuntimeError('Unexpected or duplicate sacct row: '+repr(line))
        result[fields[0]] = tuple(fields[1:])
    if set(result)!=set(jobs):
        raise RuntimeError('Missing scheduler records: '+repr(sorted(set(jobs)-set(result))))
    return result


def _candidates(repo, batch_id):
    batch=_load(repo,batch_id)
    if batch['gpu']!='h200' or batch['batch']!=batch_id:
        raise RuntimeError('Recovery expects the original H200 batch')
    registry=_doc(repo)
    # No partial automated state repair. Caller reconciles the original attempts first.
    original=[]
    for ident in batch['experiment_ids']:
        e=registry['entries'][ident]
        if e['status'] not in {'failed','resumed','running','completed'}:
            raise RuntimeError('Unreconciled/unsupported source state: '+ident+' '+e['status'])
        attempts=e['attempts']
        if not attempts:
            raise RuntimeError('Missing H200 source attempt: '+ident)
        first=attempts[0]
        if first['status']!='failed' or not JOB.fullmatch(str(first['slurm_job_id'])):
            raise RuntimeError('Unexpected initial H200 attempt: '+ident)
        original.append((ident,e,first))
    return batch,registry,original


def _retry_proof(repo, identity, entry, first, observation):
    if len(entry['attempts'])!=1 or entry['status']!='failed':
        raise RuntimeError('Not a fresh-retry candidate: '+identity)
    state, code, node=observation
    if (state,code,node)!=('FAILED','0:53','msa3'):
        raise RuntimeError(f'Unexpected original Slurm result: {identity}: {(state,code,node)}')
    if first.get('checkpoint_sha256') or first.get('parent_checkpoint_sha256') or first.get('session'):
        raise RuntimeError('Original attempt contains checkpoint/session evidence: '+identity)
    # Strict prepared-run validation: refuses any session directory or checkpoint.
    run=_prepared(repo,entry)
    if _fence(repo,identity).is_file():
        fence=json.loads(_fence(repo,identity).read_text())
        if fence.get('job_id')!=first['slurm_job_id'] or fence.get('phase')!='registered_held':
            raise RuntimeError('Original submission fence mismatch: '+identity)
    else:
        raise RuntimeError('Original submission fence missing: '+identity)
    return run


def inspect(repo, batch_id):
    batch,registry,records=_candidates(repo,batch_id)
    initial={first['slurm_job_id']:None for _,_,first in records}
    observations=_sacct(list(initial))
    counts={'failed_eligible':0,'already_retried':0}
    for ident,entry,first in records:
        if entry['status']=='failed':
            _retry_proof(repo,ident,entry,first,observations[first['slurm_job_id']])
            counts['failed_eligible']+=1
            print('RETRY_ELIGIBLE',ident,'old_job='+first['slurm_job_id'])
        else:
            if len(entry['attempts'])<2:
                raise RuntimeError('Expected recorded retry attempt: '+ident)
            counts['already_retried']+=1
            print('RETRY_ALREADY_REGISTERED',ident,'latest_job='+entry['attempts'][-1]['slurm_job_id'])
    print('RETRY_INSPECT',batch_id,json.dumps(counts,sort_keys=True))


def _create_intent(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        json.dump(data,stream,sort_keys=True,indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def submit(repo,batch_id,gpu,count=None,experiment=None):
    if gpu!='a100':
        raise RuntimeError('This recovery workflow is A100-only')
    if (count is None)==(experiment is None):
        raise ValueError('Supply exactly one of --count or --experiment')
    if count is not None and count<1:
        raise ValueError('Positive --count required')
    with locked(repo/ORCHESTRATOR):
        batch,doc,records=_candidates(repo,batch_id)
        script=_script(repo,gpu)
        pending=[(i,e,f) for i,e,f in records if e['status']=='failed']
        if experiment:
            pending=[x for x in pending if x[0]==experiment]
            if not pending:
                raise RuntimeError('Requested experiment is not eligible: '+experiment)
        else:
            if count>len(pending):
                raise RuntimeError(f'Only {len(pending)} failures eligible for new attempts')
            pending=pending[:count]
        original=_sacct([f['slurm_job_id'] for _,_,f in pending])
        # Full preflight before first external side effect.
        targets=[]
        for ident,entry,first in pending:
            run=_retry_proof(repo,ident,entry,first,original[first['slurm_job_id']])
            journal=_journal(repo,ident,first['slurm_job_id'])
            if journal.exists():
                raise RuntimeError('Existing recovery intent; manual review required: '+str(journal))
            targets.append((ident,entry,first,run,journal))
        for ident,entry,first,run,journal in targets:
            base={'schema':SCHEMA,'phase':'before_sbatch','batch':batch_id,
                  'experiment':ident,'old_job_id':first['slurm_job_id'],
                  'gpu':gpu,'mode':'fresh','target_epochs':80,'run':str(run),
                  'job_id':None}
            _create_intent(journal,base)
            args=['sbatch','--parsable','--hold',
                  '--export=ALL,T04_06_RUN='+str(run)+',T04_06_MODE=fresh,T04_06_TARGET=80',
                  str(script)]
            try:
                r=subprocess.run(args,cwd=repo,text=True,capture_output=True,check=True,timeout=60)
                job=r.stdout.strip().split(';')[0]
                if not JOB.fullmatch(job):
                    raise RuntimeError('Unrecognized sbatch job ID: '+repr(r.stdout))
                base['job_id']=job
                atomic_replace(journal,dict(base,phase='held_submitted'))
                transition(repo/RUNTIME,ident,'resumed',job_id=job,
                    note='T04-06 fresh-retry-no-checkpoint gpu_pool=a100 original_job='+first['slurm_job_id']+' resource_profile=cpu1_workers0_v1')
                atomic_replace(journal,dict(base,phase='registered_held'))
                print('RETRY_HELD_REGISTERED',job,ident,'original='+first['slurm_job_id'],flush=True)
            except Exception:
                print('RETRY_SUBMISSION_UNCERTAIN',ident,'INTENT',journal,flush=True)
                raise
        print('RETRY_HELD_BATCH',batch_id,len(targets))


def _current_retry(repo,batch_id,ident,entry):
    first=entry['attempts'][0]
    journal=_journal(repo,ident,first['slurm_job_id'])
    if not journal.exists():
        return None
    data=json.loads(journal.read_text())
    if data.get('schema')!=SCHEMA or data.get('batch')!=batch_id or data.get('gpu')!='a100' or data.get('experiment')!=ident or data.get('old_job_id')!=first['slurm_job_id']:
        raise RuntimeError('Invalid recovery journal: '+str(journal))
    if data.get('phase') not in ('registered_held','released'):
        raise RuntimeError('Unresolved recovery submission: '+str(journal))
    if len(entry['attempts'])<2 or entry['attempts'][-1]['slurm_job_id']!=data['job_id']:
        raise RuntimeError('Recovery journal and registry do not match: '+ident)
    return journal,data


def verify_held(repo,batch_id,experiment=None):
    batch,doc,records=_candidates(repo,batch_id)
    selected=[]
    for ident,entry,first in records:
        state=_current_retry(repo,batch_id,ident,entry)
        if state is None or state[1]['phase']=='released':
            continue
        if experiment is not None and ident!=experiment:
            continue
        journal,data=state
        job=data['job_id']
        line=_scontrol(job)
        pool=POOLS['a100']
        req=['JobId='+job+' ', 'JobState=PENDING', 'Reason=JobHeldUser',
             'Partition='+pool['partition'], 'QOS='+pool['qos'],
             'NumCPUs=1 ', 'cpu=1,mem='+pool['mem'],
             'gres/gpu:'+pool['gres']+'=1',
             'Command='+str((repo/pool['script']).resolve())]
        if any(t not in line for t in req):
            raise RuntimeError('Slurm retry resource/hold mismatch for '+job+'\n'+line)
        if entry['status']!='resumed':
            raise RuntimeError('Unexpected registry retry status: '+ident+' '+entry['status'])
        selected.append((ident,job,journal,data))
    if experiment and len(selected)!=1:
        raise RuntimeError('No held retry for selected experiment')
    if not selected:
        raise RuntimeError('No unreleased held recovery jobs found')
    print('RETRY_VERIFIED_HELD',batch_id,len(selected))
    return selected


def release(repo,batch_id,experiment=None):
    with locked(repo/ORCHESTRATOR):
        records=verify_held(repo,batch_id,experiment)
        for ident,job,journal,data in records:
            subprocess.run(['scontrol','release',job],check=True,timeout=15)
            atomic_replace(journal,dict(data,phase='released'))
            print('RETRY_RELEASED',job,ident,flush=True)
        print('RETRY_RELEASED_BATCH',batch_id,len(records))


def status(repo,batch_id):
    _,_,records=_candidates(repo,batch_id)
    for ident,entry,first in records:
        state=_current_retry(repo,batch_id,ident,entry)
        if state:
            print(ident,entry['status'],'old='+first['slurm_job_id'],
                  'new='+state[1]['job_id'],'phase='+state[1]['phase'])
        else:
            print(ident,entry['status'],'old='+first['slurm_job_id'],'new=-')


def register_cli(sub):
    p=sub.add_parser('retry-inspect');p.add_argument('--batch',required=True)
    p=sub.add_parser('retry-held');p.add_argument('--batch',required=True);p.add_argument('--gpu',choices=['a100'],required=True)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--count',type=int)
    group.add_argument('--experiment')
    for command in ('retry-verify','retry-release','retry-status'):
        p=sub.add_parser(command);p.add_argument('--batch',required=True)
        if command!='retry-status':p.add_argument('--experiment')


def dispatch(args):
    repo=_repo()
    if args.command=='retry-inspect':inspect(repo,args.batch)
    elif args.command=='retry-held':submit(repo,args.batch,args.gpu,args.count,args.experiment)
    elif args.command=='retry-verify':verify_held(repo,args.batch,args.experiment)
    elif args.command=='retry-release':release(repo,args.batch,args.experiment)
    elif args.command=='retry-status':status(repo,args.batch)
    else:raise RuntimeError('Invalid retry command')
