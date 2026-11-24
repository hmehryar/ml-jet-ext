"""T04-06 Stage 3D.2: checkpoint-bound manual held resume of first experiment."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import subprocess
from tools.t04_06_registry import read, locked, atomic_replace, transition, sha
from tools.t04_06_reconcile import query_sacct, evidence
from train_utils.t04_preflight import verify
from train_utils.t04_sessions import latest_checkpoint
from train_utils.t04_resume_state import load_checkpoint

DEFAULT_RUNTIME=Path('training_output/T04/t04_06/registry.runtime.json')
IDENTITY='clip_s2_e5_seed42'

def fence_for(runtime,identity):
    return Path(runtime).resolve().parent/('resume3d2.'+identity+'.intent.json')

def inspect(runtime, identity=IDENTITY):
    doc=read(runtime)
    entry=doc['entries'][identity]
    if entry['status']!='ready_to_resume':
        raise ValueError('Expected ready_to_resume; run read-only reconcile first')
    attempt=entry['attempts'][-1]
    observation=query_sacct([attempt['slurm_job_id']]).get(attempt['slurm_job_id'])
    if observation != {'state':'COMPLETED','exit_code':'0:0'}:
        raise ValueError('Prior Slurm job is not independently verified COMPLETED 0:0')
    report=evidence(entry)
    if not report.get('resume_eligible'):
        raise ValueError('No verified partial segment: '+str(report))
    run=Path(entry['run'])
    verify(run)
    ptr=latest_checkpoint(run)
    if ptr is None or ptr['sha256']!=attempt.get('checkpoint_sha256') or ptr['session']!=attempt.get('session'):
        raise ValueError('Registry checkpoint/session do not match latest pointer')
    config=json.loads((run/'config.resolved.json').read_text())
    if ptr['epoch']>=config['epochs']:
        raise ValueError('Already at configured epoch horizon')
    state=load_checkpoint(ptr['absolute_path'],map_location='cpu',
                          manifest_sha256=sha(run/'run_manifest.json'),
                          config_sha256=sha(run/'config.resolved.json'))
    if state['epoch']!=ptr['epoch'] or state.get('stopped'):
        raise ValueError('Invalid epoch or terminal early-stopped checkpoint')
    if not state.get('rng_state') or state.get('optimizer_state_dict') is None:
        raise ValueError('Incomplete optimizer/RNG resume state')
    if config.get('scheduler') and state.get('scheduler_state_dict') is None:
        raise ValueError('Missing scheduler state')
    if config.get('precision')=='fp16' and state.get('scaler_state_dict') is None:
        raise ValueError('Missing FP16 scaler state')
    target=ptr['epoch']+1
    if fence_for(runtime,identity).exists():
        raise ValueError('Previous resume intent exists; refuse duplicate')
    return entry,ptr,target

def submit_held(runtime, identity, script):
    fence=fence_for(runtime,identity)
    with locked(fence):
        entry,ptr,target=inspect(runtime,identity)
        script=Path(script).resolve()
        if not script.is_file():
            raise FileNotFoundError(script)
        subprocess.run(['bash','-n',str(script)],check=True)
        atomic_replace(fence,dict(schema=1,phase='before_sbatch',identity=identity,
                                  parent_sha256=ptr['sha256'],target_epochs=target,job_id=None))
        cmd=['sbatch','--parsable','--hold',
             '--export=ALL,T04_06_RUN='+entry['run']+',T04_06_MODE=resume,T04_06_TARGET='+str(target),
             str(script)]
        try:
            proc=subprocess.run(cmd,check=True,text=True,capture_output=True)
            job=proc.stdout.strip().split(';')[0]
            if not re.fullmatch(r'\d+',job):
                raise ValueError('Invalid Slurm submission response')
            atomic_replace(fence,dict(schema=1,phase='held_submitted',identity=identity,
                                      parent_sha256=ptr['sha256'],target_epochs=target,job_id=job))
            transition(runtime,identity,'resumed',job_id=job,checkpoint_sha256=ptr['sha256'],
                       note='Stage3D.2 held single-epoch continuation; manual release only')
            atomic_replace(fence,dict(schema=1,phase='registered_held',identity=identity,
                                      parent_sha256=ptr['sha256'],target_epochs=target,job_id=job))
            print('HELD_RESUME_REGISTERED',job,'parent_epoch',ptr['epoch'],'target_epoch',target)
            print('DO_NOT_RELEASE before verifying Slurm owner, resources and registry')
        except Exception:
            print('Submission uncertain: durable intent retained; DO NOT retry automatically')
            raise

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['inspect','submit-held','intent'])
    p.add_argument('--runtime',type=Path,default=DEFAULT_RUNTIME)
    p.add_argument('--id',default=IDENTITY)
    p.add_argument('--script',type=Path,default=Path('slurm/t04_06_single.slurm'))
    a=p.parse_args(); runtime=a.runtime.resolve()
    if a.action=='intent':
        f=fence_for(runtime,a.id)
        print(f.read_text() if f.exists() else 'NO_INTENT')
    elif a.action=='inspect':
        e,ptr,target=inspect(runtime,a.id)
        print('RESUME_ELIGIBLE',e['id'],'epoch',ptr['epoch'],'sha256',ptr['sha256'],'next_target',target,'no mutations')
    else:
        submit_held(runtime,a.id,a.script)

if __name__=='__main__':
    main()
