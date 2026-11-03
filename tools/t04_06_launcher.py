"""T04-06 Stage 3: conservative fail-closed batch-array submission.

No automatic retry/requeue. A durable intent fences the sbatch/registry gap.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from tools.t04_06_registry import read, locked, atomic_replace, transition, sha
from train_utils.t04_preflight import preflight, prepare, verify
from train_utils.t04_sessions import latest_checkpoint

NAMES = ('clip', 'siglip')
STRATEGIES = ('S1', 'S2', 'S3', 'S4')


def selected(doc):
    entries = doc['entries']
    ids = ['%s_%s_e5_seed42' % (m, s.lower()) for m in NAMES for s in STRATEGIES]
    if any(i not in entries for i in ids):
        raise ValueError('Missing E5 seed42 experiments')
    return [entries[i] for i in ids]


def validate_batch(runtime, *, require_prepared=False):
    doc = read(runtime)
    batch = selected(doc)
    for e in batch:
        if e['status'] != 'planned' or e['attempts']:
            raise ValueError('Not planned/empty-attempts: '+e['id'])
        cfg = Path(doc['repo']) / e['config']
        if sha(cfg) != e['config_sha256']:
            raise ValueError('Config hash mismatch: '+e['id'])
        run = Path(e['run'])
        if run.exists():
            verify(run)
            if (run/'source/config.original.yml').read_bytes() != cfg.read_bytes():
                raise ValueError('Existing run config mismatch: '+str(run))
            if latest_checkpoint(run) is not None or (run/'sessions').exists():
                raise ValueError('Run is no longer fresh: '+str(run))
        elif require_prepared:
            raise ValueError('Run not prepared: '+str(run))
        # The test split must not be consumed by the launcher; the T04 preflight
        # binds all three split hashes but training workers load train + val only.
    return doc, batch


def intent_path(runtime):
    return Path(runtime).parent / 'stage3.e5.seed42.intent.json'


def write_intent(path, data):
    atomic_replace(path, data)


def check_scheduler(repo):
    required = ['sbatch', 'scontrol', 'squeue', 'sacct']
    import shutil
    missing = [x for x in required if not shutil.which(x)]
    if missing:
        raise RuntimeError('Slurm commands unavailable: '+','.join(missing))
    script = repo / 'slurm/t04_06_array.slurm'
    subprocess.run(['bash','-n',str(script)], check=True)
    # Validate partition/QoS through Slurm rather than assuming 16 MIG slots.
    probe = subprocess.run(['sinfo','-h','-p','ecscp','-o','%P'],capture_output=True,text=True)
    if probe.returncode or not probe.stdout.strip():
        raise RuntimeError('Cannot verify ecscp partition')
    return script


def prepare_batch(runtime):
    runtime = Path(runtime).resolve()
    with locked(intent_path(runtime).with_suffix('.prepare.lock')):
        if intent_path(runtime).exists():
            raise RuntimeError('Submission intent exists; do not prepare again')
        doc, batch = validate_batch(runtime)
        for e in batch:
            cfg = Path(doc['repo'])/e['config']
            run = Path(e['run'])
            if not run.exists():
                preflight(cfg)
                actual = Path(prepare(cfg,run.name)).resolve()
                if actual != run.resolve():
                    raise RuntimeError('Prepared run differs from plan: '+str(actual))
            verify(run)
            print('PREPARED',e['id'],run,flush=True)


def array_manifest(runtime, batch):
    return {'schema':1,'kind':'t04-06-e5-array', 'registry':str(Path(runtime).resolve()),
            'entries':[{'id':e['id'],'run':e['run'],'config_sha256':e['config_sha256']} for e in batch]}


def submit(runtime, *, concurrency, script=None):
    runtime = Path(runtime).resolve()
    if type(concurrency) is not int or not 1 <= concurrency <= 4:
        raise ValueError('Stage3 first gate permits concurrency 1..4')
    fence = intent_path(runtime)
    # Lock across sbatch / registry transaction / release, including across processes.
    with locked(fence):
        if fence.exists():
            raise RuntimeError('Submission intent already exists. Reconcile manually; NEVER auto-resubmit.')
        doc, batch = validate_batch(runtime,require_prepared=True)
        repo = Path(doc['repo'])
        script = Path(script) if script else check_scheduler(repo)
        if not script.is_file():
            raise FileNotFoundError(script)
        manifest = fence.parent / 'stage3.e5.seed42.array.json'
        if manifest.exists():
            raise RuntimeError('Array manifest already exists: '+str(manifest))
        # Preserve a fence BEFORE talking to Slurm. Crashes must require manual reconciliation.
        write_intent(fence, {'schema':1,'phase':'before_sbatch','job_id':None,
                             'entry_ids':[e['id'] for e in batch]})
        try:
            write_intent(manifest,array_manifest(runtime,batch))
            cmd=['sbatch','--parsable','--hold',f'--array=0-7%{concurrency}',
                 '--export=ALL,T04_06_ARRAY_MANIFEST='+str(manifest),str(script)]
            response=subprocess.run(cmd,cwd=repo,text=True,capture_output=True,check=True)
            job = response.stdout.strip().split(';')[0]
            if not re.fullmatch(r'\d+',job):
                raise RuntimeError('Unexpected sbatch response '+response.stdout)
            write_intent(fence, {'schema':1,'phase':'held_submitted','job_id':job,
                                 'entry_ids':[e['id'] for e in batch], 'manifest':str(manifest)})
            for i,e in enumerate(batch):
                transition(runtime,e['id'],'submitted',job_id=f'{job}_{i}',
                           note='T04-06 held E5 seed42 fresh array')
            write_intent(fence, {'schema':1,'phase':'registered_held','job_id':job,
                                 'entry_ids':[e['id'] for e in batch], 'manifest':str(manifest)})
            subprocess.run(['scontrol','release',job],check=True)
            write_intent(fence, {'schema':1,'phase':'released','job_id':job,
                                 'entry_ids':[e['id'] for e in batch], 'manifest':str(manifest)})
            print('SUBMITTED_ARRAY',job,'TASKS=8 CONCURRENCY='+str(concurrency))
        except Exception:
            # Retain fence and held jobs; a person must inspect Slurm and registry.
            print('ERROR: submission uncertainty; inspect intent, registry and Slurm. DO NOT retry.',file=sys.stderr)
            raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['dry-run','prepare','submit','intent'])
    p.add_argument('--runtime',type=Path,default=Path('training_output/T04/t04_06/registry.runtime.json'))
    p.add_argument('--concurrency',type=int,default=4)
    a=p.parse_args()
    a.runtime=a.runtime.resolve()
    if a.command=='intent':
        f=intent_path(a.runtime)
        print(f.read_text() if f.exists() else 'NO_INTENT')
        return
    if a.command=='dry-run':
        doc,batch=validate_batch(a.runtime)
        for i,e in enumerate(batch):
            print(i,e['id'],e['status'],e['run'], 'prepared='+str(Path(e['run']).exists()))
        print('PASS: eight unique planned E5 seed42 jobs; dry-run did not reserve or submit')
    elif a.command=='prepare':
        prepare_batch(a.runtime)
    else:
        submit(a.runtime,concurrency=a.concurrency)


if __name__=='__main__':main()
