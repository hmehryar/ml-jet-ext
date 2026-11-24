"""T04-07 21-run CE-loss pilot controller. Safe by default: commands are read-only
unless they explicitly include --apply. Slurm submissions are always held.

Run as: python -m tools.t04_07_production COMMAND [arguments]
Python 3.9 compatible; all writes are separate from T04-06 output roots.
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
import subprocess
import tempfile

import yaml

from train_utils.t04_preflight import parse_config, preflight, prepare, verify
from train_utils.t04_sessions import latest_checkpoint

REPO = Path(__file__).resolve().parents[1]
CONFIGS = Path('experiments/t04/config/t04_07_pilot')
OUTPUT = Path('/rs/rs_grp_mljet/ml-jet-ext/training_output/T04/t04_07_loss_pilot')
STATE = OUTPUT / '_operations'
REGISTRY = STATE / 'registry.runtime.json'
SCRIPT = Path('slurm/t04_07_production.slurm')
COMMIT = '70d2a7b3c239adad334251b27cf1d4b04e51e3dd'
SEEDS = (42, 123, 2026)
TREATMENTS = (('physical_mae', None), ('ce_physical_mae', 0.1),
              ('ce_physical_mae', 0.5), ('ce_physical_mae', 1.0),
              ('ce_ordinal', 0.1), ('ce_ordinal', 0.5), ('ce_ordinal', 1.0))
ACTIVE = {'submitted', 'running'}
SHA_RE = re.compile(r'^[0-9a-f]{64}$')
JOB_RE = re.compile(r'^\d+$')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.t04-07-', suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, sort_keys=True, indent=2)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
        fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


@contextmanager
def mutex():
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / 'registry.lock').open('a+b') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def assert_git():
    import subprocess as sp
    head = sp.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    # Permit a later commit that adds the controller/tests, provided the validated
    # 70d2a7b training implementation itself has not changed.
    if sp.run(['git','-C',str(REPO),'merge-base','--is-ancestor',COMMIT,head],
              check=False).returncode != 0:
        raise RuntimeError('Validated base commit is not an ancestor of current HEAD')
    core = ('train_utils','models','data','config.py','train_t04.ipynb')
    if sp.run(['git','-C',str(REPO),'diff','--quiet',COMMIT,head,'--',*core],
              check=False).returncode != 0:
        raise RuntimeError('Frozen training implementation differs from validated base')
    tracked = sp.check_output(['git', '-C', str(REPO), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    if tracked:
        raise RuntimeError('Tracked source has uncommitted modifications: '+tracked[:240])
    extras = sp.check_output(['git','-C',str(REPO),'ls-files','--others',
                              '--exclude-standard','--',*core],text=True).strip()
    if extras:
        raise RuntimeError('Untracked training code would enter frozen snapshot: '+extras[:240])
    return head


def expected_rows():
    rows = []
    for seed in SEEDS:
        for kind, weight in TREATMENTS:
            suffix = kind if weight is None else kind + '_lambda' + str(weight).replace('.', 'p')
            identity = 'clip_s4_e3_seed{}_{}'.format(seed, suffix)
            path = REPO / CONFIGS / (identity + '.yml')
            if not path.is_file():
                raise FileNotFoundError(path)
            cfg = parse_config(path)
            loss = {'type': kind}
            if kind == 'ce_physical_mae':
                loss['mae_weight'] = weight
            elif kind == 'ce_ordinal':
                loss['ordinal_weight'] = weight
            if (cfg.model != 'clip' or cfg.strategy != 'S4' or cfg.moments != [1,2,3]
                    or cfg.seed != seed or cfg.moment_loss != loss
                    or cfg.epochs != 80 or cfg.training_protocol != 'fixed_budget_v2'
                    or cfg.precision != 'bf16' or cfg.split_protocol != 'aggregate_disjoint'
                    or cfg.batch_size != 8 or cfg.gradient_accumulation != 1
                    or Path(cfg.output_root).resolve() != OUTPUT.resolve()):
                raise ValueError('Unexpected immutable configuration: '+str(path))
            runid = 't04-07-' + identity.replace('_', '-') + '-prod-v1'
            run = OUTPUT / 'clip' / 'S4' / 'E3' / ('seed-'+str(seed)) / runid
            rows.append({'id': identity, 'config': str(path.relative_to(REPO)),
                         'config_sha256': sha(path), 'run_id': runid,
                         'run': str(run.resolve()), 'status': 'planned',
                         'attempts': [], 'events': []})
    if len(rows) != 21 or len({r['run'] for r in rows}) != 21:
        raise AssertionError('Expected 21 distinct runs')
    plan_path = REPO / CONFIGS / 'pilot_plan.json'
    plan = json.loads(plan_path.read_text())
    if plan.get('new_config_count') != 21 or plan.get('baseline_reused') != 3:
        raise RuntimeError('Pilot plan mismatch')
    published = plan.get('rows', [])
    if len(published) != 21 or len({p.get('id') for p in published}) != 21:
        raise RuntimeError('Pilot plan missing 21 distinct identities')
    by_id = {p['id']: p for p in published}
    for row in rows:
        p = by_id.get(row['id'])
        cfg = yaml.safe_load((REPO / row['config']).read_text())
        if not p or p.get('sha256') != row['config_sha256'] or p.get('seed') != cfg['seed'] or p.get('loss') != cfg['moment_loss']:
            raise RuntimeError('Pilot plan entry/hash differs: '+row['id'])
    return rows, sha(plan_path)


def load():
    if not REGISTRY.is_file():
        raise FileNotFoundError('Registry not initialized; run init --apply only after dry-run')
    doc = json.loads(REGISTRY.read_text())
    if doc.get('kind') != 'T04-07-21-run-production-v1' or doc.get('schema') != 1 or doc.get('git_commit') != COMMIT:
        raise ValueError('Wrong registry identity/commit')
    rows, plan_hash = expected_rows()
    if doc.get('plan_sha256') != plan_hash or len(doc.get('entries', {})) != 21:
        raise RuntimeError('Plan changed or missing registry entries')
    for row in rows:
        e = doc['entries'].get(row['id'])
        if not e or any(e.get(k) != row[k] for k in ('id','run','run_id','config','config_sha256')):
            raise RuntimeError('Changed registry entry: '+row['id'])
        if e['status'] not in {'planned','prepared','submitting','submitted','running','ready_to_resume','completed','failed','needs_review'}:
            raise RuntimeError('Invalid registry state: '+row['id'])
        if len(e['attempts']) > 1 and len({a['job_id'] for a in e['attempts'] if a.get('job_id')}) != len([a for a in e['attempts'] if a.get('job_id')]):
            raise RuntimeError('Duplicated attempt job ID')
    return doc


def save(doc):
    atomic_json(REGISTRY, doc)


def do_full_dry_run():
    """Full 21-entry CPU-only preflight; no filesystem writes or Slurm calls."""
    head=assert_git()
    rows,plan_hash=expected_rows()
    if REGISTRY.exists():
        raise FileExistsError('Runtime registry already exists; use inspect/prepare preview')
    if not (REPO/SCRIPT).is_file():
        raise FileNotFoundError(REPO/SCRIPT)
    subprocess.run(['bash','-n',str(REPO/SCRIPT)],check=True)
    from train_utils.t04_training_termination import validate_protocol
    for row in rows:
        cfg, inputs=preflight(REPO/row['config'])
        if Path(row['run']).exists():
            raise FileExistsError('Run collision: '+row['run'])
        validate_protocol(cfg.training_protocol,cfg.training_protocol_hash,cfg.epochs)
        if cfg.training_protocol!='fixed_budget_v2' or cfg.epochs!=80:
            raise RuntimeError('Training protocol changed')
        if inputs.get('source_overlap_independence_claim'):
            raise RuntimeError('Unexpected source overlap representation; review provenance')
    print(json.dumps({'status':'PASS','mode':'CPU_ONLY_DRY_RUN',
        'commit':head,'plan_sha256':plan_hash,'config_count':21,
        'preflight_passed':21,'proposed_prepared_runs':21,
        'proposed_submission_target_epochs':80,'proposed_submission_mode':'fresh',
        'hold_required':True,'max_per_explicit_submission':1,
        'source_disjoint_claim':False,'runs_reserved':0,'registry_written':False,
        'scheduler_queried':False,'jobs_submitted':0},indent=2))


def do_init(apply=False):
    head = assert_git()
    rows, plan = expected_rows()
    for row in rows:
        cfg, _ = preflight(REPO / row['config'])
        if Path(cfg.output_root).resolve() != OUTPUT.resolve():
            raise AssertionError('Output root changed')
        if Path(row['run']).exists():
            raise FileExistsError('Reserved run already exists: '+row['run'])
    if REGISTRY.exists():
        raise FileExistsError('Registry already exists: '+str(REGISTRY))
    print(json.dumps({'action':'init','mode':'apply' if apply else 'dry-run',
                      'expected':len(rows),'git_commit':head,'plan_sha256':plan,
                      'registry':str(REGISTRY),'runs_reserved':0,'jobs_submitted':0}, indent=2))
    if apply:
        with mutex():
            if REGISTRY.exists():
                raise FileExistsError(REGISTRY)
            doc = {'schema':1,'kind':'T04-07-21-run-production-v1',
                   'git_commit':head,'plan_sha256':plan,'created_utc':utc(),
                   'entries':{r['id']:r for r in rows}}
            with REGISTRY.open('x') as f:
                json.dump(doc, f, sort_keys=True, indent=2)
                f.write('\n')
            print('INITIALIZED', REGISTRY)


def do_prepare(apply=False):
    assert_git()
    doc = load()
    wanted = [e for e in doc['entries'].values() if e['status']=='planned']
    # Fail closed if a planned run already exists; investigate rather than silently adopt it.
    for e in wanted:
        if Path(e['run']).exists():
            raise FileExistsError('Run exists without registry preparation: '+e['run'])
        preflight(REPO / e['config'])
    print('PREPARE', 'apply' if apply else 'dry-run', 'planned',len(wanted), 'already_prepared',21-len(wanted))
    if not apply:
        return
    for e in wanted:
        with mutex():
            doc = load()
            current = doc['entries'][e['id']]
            if current['status'] != 'planned' or Path(current['run']).exists():
                raise RuntimeError('Concurrent or changed preparation: '+e['id'])
            # Mark intent before allocating a run so an interruption is detectable.
            current['status'] = 'needs_review'
            current['events'].append({'at':utc(),'event':'preparation_intent'})
            save(doc)
        path = Path(prepare(REPO / e['config'], e['run_id'])).resolve()
        if path != Path(e['run']).resolve():
            raise RuntimeError('Prepared path mismatch: '+str(path))
        verify(path)
        original = path / 'source/config.original.yml'
        if sha(original) != e['config_sha256']:
            raise RuntimeError('Frozen YAML SHA mismatch: '+e['id'])
        with mutex():
            doc = load()
            current = doc['entries'][e['id']]
            if current['status'] != 'needs_review':
                raise RuntimeError('Unexpected preparation state')
            current['status'] = 'prepared'
            current['events'].append({'at':utc(),'event':'prepared'})
            save(doc)
        print('PREPARED', e['id'])


def verify_prepared(e):
    run = Path(e['run'])
    verify(run)
    if sha(run / 'source/config.original.yml') != e['config_sha256']:
        raise RuntimeError('Frozen configuration changed: '+e['id'])
    manifest = json.loads((run / 'run_manifest.json').read_text())
    provenance = manifest.get('git',{}).get('commit')
    if not provenance or subprocess.run(['git','-C',str(REPO),'merge-base','--is-ancestor',
                                         COMMIT,provenance],check=False).returncode != 0:
        raise RuntimeError('Frozen source commit does not descend from validated implementation: '+e['id'])
    return run


def observe(e):
    run = verify_prepared(e)
    ptr = latest_checkpoint(run)
    if ptr is None:
        sessions = run / 'sessions'
        if sessions.exists() and any(sessions.iterdir()):
            return {'epoch':0,'status':'uncommitted_session'}
        return {'epoch':0,'status':'prepared'}
    cfg = json.loads((run / 'config.resolved.json').read_text())
    epoch = ptr['epoch']
    if epoch < 1 or epoch > cfg['epochs']:
        raise RuntimeError('Checkpoint outside configured training budget')
    history = run / 'moment_epoch_metrics.csv'
    import csv
    with history.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if [int(x['epoch']) for x in rows] != list(range(1,epoch+1)):
        raise RuntimeError('Nonconsecutive epoch CSV')
    session = json.loads((run/'sessions'/ptr['session']/'session_manifest.json').read_text())
    if session.get('status') != 'completed':
        return {'epoch':epoch,'status':'incomplete_session','sha256':ptr['sha256']}
    return {'epoch':epoch,'status':'completed_session','sha256':ptr['sha256'], 'session':ptr['session']}


def inspect():
    doc=load()
    counts={}
    for e in doc['entries'].values():
        counts[e['status']]=counts.get(e['status'],0)+1
        if e['status'] != 'planned':
            verify_prepared(e)
    print(json.dumps({'registry':str(REGISTRY),'counts':counts,'runs':len(doc['entries'])},indent=2))


def submission_preview(max_count=1, mode="fresh", allow_active=False):
    if not 1 <= max_count <= 3:
        raise ValueError('Max submission count is 1..3 per explicit operation')
    assert_git()
    doc=load()
    if any(e['status'] == 'submitting' for e in doc['entries'].values()):
        raise RuntimeError('Unresolved submission intent: investigate before further sbatch calls')
    if not allow_active and any(e['status'] in ACTIVE for e in doc['entries'].values()):
        raise RuntimeError('There is an unresolved active/submitting attempt; reconcile first')
    if mode not in ('fresh', 'resume'):
        raise ValueError('Unsupported submission mode')
    desired = 'prepared' if mode == 'fresh' else 'ready_to_resume'
    prepared = [e for e in doc['entries'].values() if e['status']==desired]
    selected=prepared[:max_count]
    if not selected:
        raise RuntimeError('No prepared experiments available')
    for e in selected:
        obs=observe(e)
        if mode == 'fresh' and obs != {'epoch':0,'status':'prepared'}:
            raise RuntimeError('Fresh run has unexpected artifacts: '+e['id'])
        if mode == 'resume' and not (obs['status']=='completed_session' and 0<obs['epoch']<80 and SHA_RE.fullmatch(obs['sha256'])):
            raise RuntimeError('Resume lacks verified partial completed session: '+e['id'])
    return selected


def do_preview(count=1,mode="fresh"):
    selected=submission_preview(count,mode)
    print(json.dumps({'action':'submission-preview','jobs':len(selected),'held':True,
        'max_concurrent_array_tasks':1,'entries':[{'id':e['id'],'run':e['run'],
        'mode':mode,'target_epochs':80} for e in selected],
        'warning':'NO sbatch, no run reservation, no release; submit requires explicit --apply'},indent=2))


def do_submit(count=1,apply=False,mode="fresh",allow_active=False):
    # Intentionally no automatic submission by default and always use --hold.
    selected=(submission_preview(count,mode,allow_active=True) if allow_active
              else submission_preview(count,mode))
    if not apply:
        do_preview(count,mode)
        return
    if count != 1:
        raise RuntimeError('Initial submissions must be one at a time for transaction safety')
    script=REPO / SCRIPT
    if not script.is_file():
        raise FileNotFoundError(script)
    subprocess.run(['bash','-n',str(script)],check=True)
    e=selected[0]
    record={'at':utc(),'id':e['id'],'mode':mode,'target_epochs':80,
            'run':e['run'],'job_id':None,'held':True}
    intent = STATE / 'intents' / (e['id']+'.'+mode+'.json')
    intent.parent.mkdir(parents=True,exist_ok=True)
    with mutex():
        doc=load()
        if doc['entries'][e['id']]['status']!=('prepared' if mode=='fresh' else 'ready_to_resume'):
            raise RuntimeError('Submission state changed')
        with intent.open('x') as f:
            json.dump(record,f,indent=2)
            f.write('\n')
        doc['entries'][e['id']]['status']='submitting'
        doc['entries'][e['id']]['events'].append({'at':utc(),'event':'sbatch_intent','intent':str(intent)})
        save(doc)
    # Log path must already exist; no auto release. Failures leave submitting intent.
    logs=STATE/'slurm_logs'
    logs.mkdir(parents=True,exist_ok=True)
    cmd=['sbatch','--parsable','--hold',
         '--output='+str(logs/'%j.out'),'--error='+str(logs/'%j.err'),
         '--export=ALL,T04_07_PROD_RUN='+e['run']+',T04_07_PROD_MODE='+mode+',T04_07_PROD_TARGET=80',
         str(script)]
    result=subprocess.run(cmd,cwd=str(REPO),text=True,capture_output=True,check=True)
    jobid=result.stdout.strip().split(';')[0]
    if not JOB_RE.fullmatch(jobid):
        raise RuntimeError('Slurm returned unrecognized job ID; intent retained: '+result.stdout)
    record['job_id']=jobid
    with mutex():
        doc=load()
        entry=doc['entries'][e['id']]
        if entry['status']!='submitting':
            raise RuntimeError('Submission state changed after sbatch; investigate '+jobid)
        entry['attempts'].append(record)
        entry['status']='submitted'
        entry['events'].append({'at':utc(),'event':'submitted_held','job_id':jobid})
        save(doc)
    print('SUBMITTED HELD',jobid,e['id'])
    print('NOT RELEASED. Inspect with scontrol show job '+jobid)
    print('To explicitly release after approval: scontrol release '+jobid)




def do_submit_remaining(*, apply=False, limit=20):
    """Submit previously prepared, distinct runs as HELD individual Slurm jobs.

    Explicitly authorized extension for the established T04-07 production wave.
    Never releases jobs. Refuses ambiguity or partially submitted waves, so an
    operator must inspect/reconcile before retrying after an sbatch error.
    """
    if type(limit) is not int or not 1 <= limit <= 20:
        raise ValueError('limit must be an integer between 1 and 20')
    assert_git()
    doc = load()
    entries = list(doc['entries'].values())
    counts = {state: sum(e['status'] == state for e in entries)
              for state in set(e['status'] for e in entries)}
    if any(e['status'] in ('submitting', 'needs_review', 'failed', 'ready_to_resume')
           for e in entries):
        raise RuntimeError('Ambiguous or failed registry entry: investigate before submitting batch')
    active = [e for e in entries if e['status'] in ACTIVE]
    waiting = [e for e in entries if e['status'] == 'prepared']
    # The initial production wave has one active first run and twenty prepared.
    if len(active) != 1 or len(waiting) != 20 or len(entries) != 21:
        raise RuntimeError('Expected exactly one active first job and twenty prepared runs: ' + str(counts))
    selected = waiting[:limit]
    for e in selected:
        observed = observe(e)
        if observed != {'epoch':0, 'status':'prepared'}:
            raise RuntimeError('Prepared run has unexpected artifacts: ' + e['id'])
    print(json.dumps({'action':'submit-remaining', 'mode':'apply' if apply else 'dry-run',
        'submission_style':'individual sbatch --hold', 'already_active':len(active),
        'selected_count':len(selected), 'selected_ids':[e['id'] for e in selected],
        'will_release':False}, indent=2))
    if not apply:
        return
    for e in selected:
        # Re-read registry and revalidate selected candidate for each transaction.
        candidate = submission_preview(1, 'fresh', allow_active=True)[0]
        if candidate['id'] != e['id']:
            raise RuntimeError('Registry ordering/state changed before submission: ' + e['id'])
        do_submit(1, True, 'fresh', allow_active=True)
    print('BATCH SUBMITTED HELD:', len(selected), 'new individual jobs. NO jobs released.')


def do_release_preview():
    """Read-only list of T04-07 production job IDs for separate manual release."""
    doc = load()
    jobs = []
    for entry in doc['entries'].values():
        if entry['status'] in ACTIVE and entry['attempts']:
            attempt = entry['attempts'][-1]
            if attempt.get('held') and JOB_RE.fullmatch(str(attempt.get('job_id',''))):
                jobs.append({'id':entry['id'],'job_id':str(attempt['job_id'])})
    print(json.dumps({'action':'release-preview', 'note':'READ ONLY; verify held state with scontrol before release',
                       'jobs':jobs}, indent=2))


def do_reconcile(apply=False):
    doc=load()
    active=[e for e in doc['entries'].values() if e['status'] in ACTIVE]
    if not active:
        print('RECONCILE: no active jobs; read-only',not apply)
        return
    ids=[e['attempts'][-1]['job_id'] for e in active]
    from tools.t04_06_v2_reconcile import query_sacct
    sched=query_sacct(ids)
    changes=[]
    for e in active:
        job=e['attempts'][-1]['job_id']
        obs=sched.get(job)
        state=obs['state'] if obs else 'UNKNOWN'
        action='none'; detail=''
        if state == 'RUNNING':
            action='running' if e['status']=='submitted' else 'none'
        elif state == 'COMPLETED' and obs['exit_code']=='0:0':
            artifact=observe(e)
            if artifact['status']=='completed_session' and artifact['epoch']==80:
                action='completed'
            elif artifact['status']=='completed_session' and artifact['epoch']<80:
                action='ready_to_resume'
            else:
                detail='Terminal job without verified completed session: '+str(artifact)
        elif state in ('FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'):
            action='needs_review'; detail='Slurm '+state+'; inspect checkpoint before recovery'
        elif state in ('PENDING','CONFIGURING','SUSPENDED'):
            detail='queued or held'
        else:
            detail='unknown or ambiguous Slurm state; no transition'
        changes.append({'id':e['id'],'job_id':job,'scheduler_state':state,
                        'action':action,'detail':detail})
    print(json.dumps(changes,indent=2))
    if apply:
        with mutex():
            newer=load()
            for item in changes:
                if item['action']=='none':continue
                e=newer['entries'][item['id']]
                if e['attempts'][-1]['job_id']!=item['job_id'] or e['status'] not in ACTIVE:
                    raise RuntimeError('Stale reconciliation proposal')
                e['status']=item['action']
                e['attempts'][-1]['observed_state']=item['scheduler_state']
                e['events'].append({'at':utc(),'event':item['action'],'job_id':item['job_id']})
            save(newer)
        print('REGISTRY UPDATED; no jobs submitted')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    for name in ('init','prepare','reconcile'):
        s=sub.add_parser(name)
        s.add_argument('--apply',action='store_true')
    sub.add_parser('inspect')
    sub.add_parser('dry-run')
    s=sub.add_parser('preview')
    s.add_argument('--count',type=int,default=1)
    s.add_argument('--mode',choices=('fresh','resume'),default='fresh')
    s=sub.add_parser('submit-remaining')
    s.add_argument('--limit',type=int,default=20)
    s.add_argument('--apply',action='store_true')
    sub.add_parser('release-preview')
    s=sub.add_parser('submit')
    s.add_argument('--count',type=int,default=1)
    s.add_argument('--mode',choices=('fresh','resume'),default='fresh')
    s.add_argument('--apply',action='store_true')
    args=p.parse_args(argv)
    if args.command=='dry-run':do_full_dry_run()
    elif args.command=='init':do_init(args.apply)
    elif args.command=='prepare':do_prepare(args.apply)
    elif args.command=='inspect':inspect()
    elif args.command=='preview':do_preview(args.count,args.mode)
    elif args.command=='submit':do_submit(args.count,args.apply,args.mode)
    elif args.command=='submit-remaining':do_submit_remaining(apply=args.apply,limit=args.limit)
    elif args.command=='release-preview':do_release_preview()
    elif args.command=='reconcile':do_reconcile(args.apply)

if __name__=='__main__':main()
