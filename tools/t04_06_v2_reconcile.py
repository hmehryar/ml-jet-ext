"""T04-06 v2: fixed-budget-only reconciliation of Slurm and immutable checkpoint evidence.

Read-only by default. --apply performs registry transitions only after positive evidence.
Does not create sessions, prepare runs, submit, cancel, or requeue jobs.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import re
import subprocess

from tools.t04_06_registry import read, transition, sha
from train_utils.t04_training_termination import validate_protocol
from tools.t04_06_v2_matrix import PLAN, RUNTIME, KIND, OUTPUT
from train_utils.t04_preflight import verify
from train_utils.t04_sessions import latest_checkpoint

FINAL_OK = {'COMPLETED'}
FINAL_BAD = {'FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'}
ACTIVE = {'RUNNING','COMPLETING'}
WAITING = {'PENDING','CONFIGURING','SUSPENDED','RESIZING'}


def parse_sacct(text):
    """Parse sacct -P root task records, excluding .batch/.extern steps."""
    out = {}
    for line in text.splitlines():
        bits = line.strip().split('|')
        if len(bits) < 3 or not re.fullmatch(r'\d+(?:_\d+)?', bits[0]):
            continue
        job, state, exit_code = bits[:3]
        state = state.split()[0].split('+')[0].upper()
        if job in out:
            raise ValueError('Ambiguous duplicate Slurm task: '+job)
        out[job] = {'state': state, 'exit_code': exit_code}
    return out


def query_sacct(job_ids):
    """Failures are unknown scheduler evidence, never grounds for failure transitions."""
    if not job_ids:
        return {}
    command = ['sacct','-n','-P','-j',','.join(sorted(job_ids)),
               '--format=JobIDRaw,State,ExitCode']
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError('sacct failed: '+result.stderr.strip())
    return parse_sacct(result.stdout)


def evidence(entry):
    """Reject legacy termination; v2 completes only at configured epoch budget."""
    run = Path(entry['run'])
    if not run.is_dir():
        return {'ok': False, 'reason': 'prepared run missing'}
    try:
        verify(run)
        cfg = json.loads((run/'config.resolved.json').read_text())
        validate_protocol(cfg.get('training_protocol'), cfg.get('training_protocol_hash'), cfg['epochs'])
        if cfg['training_protocol'] != 'fixed_budget_v2' or Path(cfg['output_root']) != Path(OUTPUT):
            raise ValueError('Non-v2 run or output root')
        pointer = latest_checkpoint(run)
        if pointer is None:
            raise ValueError('checkpoint missing')
        epoch = pointer['epoch']
        if epoch < 1 or epoch > cfg['epochs']:
            raise ValueError('checkpoint epoch outside configured budget')
        with (run/'moment_epoch_metrics.csv').open(newline='') as f:
            rows = list(csv.DictReader(f))
        if [int(row['epoch']) for row in rows] != list(range(1, epoch + 1)):
            raise ValueError('nonconsecutive CSV epoch history')
        session_dir = run/'sessions'/pointer['session']
        session = json.loads((session_dir/'session_manifest.json').read_text())
        result = json.loads((session_dir/'result.json').read_text())
        if session.get('status') != 'completed' or result.get('epoch') != epoch:
            raise ValueError('session incomplete/epoch mismatch')
        if result.get('early_stopped') is not False:
            raise ValueError('fixed_budget_v2 forbids early-stopping terminal state')
        if epoch == cfg['epochs']:
            return {'ok':True,'reason':'validated configured budget reached',
                    'epoch':epoch,'checkpoint_sha256':pointer['sha256'],
                    'early_stopped':False,'resume_eligible':False}
        return {'ok':False,'reason':'verified partial fixed-budget segment',
                'epoch':epoch,'checkpoint_sha256':pointer['sha256'],'resume_eligible':True}
    except (OSError, ValueError, KeyError, AssertionError, RuntimeError, TypeError) as exc:
        return {'ok':False,'reason':str(exc)[:240]}


def _require_v2_registry(doc, runtime):
    repo = Path(doc['repo']).resolve()
    if Path(runtime).resolve() != (repo / RUNTIME).resolve() or Path(doc['plan_path']).resolve() != (repo / PLAN).resolve():
        raise RuntimeError('Refusing noncanonical fixed_budget_v2 registry')
    plan = json.loads(Path(doc['plan_path']).read_text())
    if plan.get('kind') != KIND or plan.get('training_protocol') != 'fixed_budget_v2':
        raise RuntimeError('Not a v2 plan')


def decide(entry, slurm):
    if entry['status'] not in ('submitted','running','resumed'):
        return {'action':'none','reason':'registry entry is not active'}
    if not entry['attempts']:
        return {'action':'none','reason':'active entry lacks attempt'}
    last = entry['attempts'][-1]
    observation = slurm.get(last['slurm_job_id'])
    if observation is None:
        return {'action':'none','reason':'Slurm task not observed (unknown)'}
    state = observation['state']
    if state in WAITING:
        return {'action':'none','reason':'waiting: '+state}
    if state in ACTIVE:
        return {'action':'running' if entry['status'] in ('submitted','resumed') else 'none',
                'reason':'Slurm '+state}
    if state == 'COMPLETED' and observation['exit_code'] == '0:0':
        artifact = evidence(entry)
        if artifact.get('ok'):
            return {'action':'completed','reason':artifact['reason'],
                    'checkpoint_sha256':artifact['checkpoint_sha256'], 'epoch':artifact['epoch'],
                    'early_stopped':artifact['early_stopped']}
        if artifact.get('resume_eligible') and artifact.get('checkpoint_sha256'):
            return {'action':'ready_to_resume', 'reason':'Slurm success; verified partial segment',
                    'checkpoint_sha256':artifact['checkpoint_sha256'], 'epoch':artifact['epoch']}
        return {'action':'none','reason':'Slurm complete but artifacts insufficient: '+artifact['reason']}
    if state in FINAL_BAD or (state == 'COMPLETED' and observation['exit_code'] != '0:0'):
        return {'action':'failed','reason':'Slurm '+state+' exit '+observation['exit_code']}
    return {'action':'none','reason':'unrecognized Slurm state '+state}


def reconcile(runtime, *, apply=False, scheduler=None):
    doc = read(runtime)
    _require_v2_registry(doc, runtime)
    entries = [e for e in doc['entries'].values() if e['status'] in ('submitted','running','resumed')]
    ids = {e['attempts'][-1]['slurm_job_id'] for e in entries}
    scheduler = query_sacct(ids) if scheduler is None else scheduler
    reports = []
    for entry in entries:
        decision = decide(entry, scheduler)
        record = {'id':entry['id'],'job_id':entry['attempts'][-1]['slurm_job_id'], **decision}
        if apply and decision['action'] != 'none':
            # The transactional transition API re-reads the registry under lock.
            # Safely promote submitted->running for already finished Slurm jobs.
            ident, attempt = entry['id'],entry['attempts'][-1]['id']
            if entry['status'] in ('submitted','resumed') and decision['action'] in ('completed','failed','ready_to_resume'):
                transition(runtime,ident,'running',attempt_id=attempt,
                           note='Observed terminal Slurm task; intermediate state')
            if decision['action'] == 'running':
                transition(runtime,ident,'running',attempt_id=attempt,note=decision['reason'])
            elif decision['action'] == 'ready_to_resume':
                transition(runtime,ident,'ready_to_resume',attempt_id=attempt,
                           checkpoint_sha256=decision['checkpoint_sha256'],
                           session=json.loads((Path(entry['run'])/'latest_checkpoint.json').read_text())['session'],
                           note=decision['reason'])
            elif decision['action'] == 'completed':
                transition(runtime,ident,'completed',attempt_id=attempt,
                           checkpoint_sha256=decision['checkpoint_sha256'],note=decision['reason'])
            elif decision['action'] == 'failed':
                transition(runtime,ident,'failed',attempt_id=attempt,note=decision['reason'])
            record['applied']=True
        reports.append(record)
    return reports


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('inspect','apply'))
    parser.add_argument('--runtime',type=Path,default=RUNTIME)
    parser.add_argument('--output',type=Path,help='Optional new JSON report path (never overwrites)')
    args=parser.parse_args()
    results=reconcile(args.runtime.resolve(),apply=args.command=='apply')
    for item in results:
        print(json.dumps(item,sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as f:
            json.dump({'schema':1,'mode':args.command,'entries':results},f,indent=2)
            f.write('\n')
    print('RECONCILED',len(results),'active entries; mode='+args.command)

if __name__=='__main__':
    main()
