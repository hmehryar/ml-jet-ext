"""T04-10 exact T04-07 seven-treatment loss matrix, 84 new single-head runs.

No writes for 'plan'/'preview'. 'prepare' writes immutable NEW runs; 'submit'
creates HELD Slurm jobs only. Release jobs explicitly after resource inspection.
Never touches 80/200-epoch CE runs or T04-07 output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess

from train_utils.t04_10_preflight import prepare, verify
from train_utils.t04_10_loss_runner import verify_ablation, sha
from train_utils.loss_config import resolve_moment_loss

ROOT = Path('/rs/rs_grp_mljet/ml-jet-ext/training_output/T04/t04_10_single_head')
REPO = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO/'experiments/t04/config/t04_10_stage4_frozen'
# EXACT treatment definition/order from tools/t04_07_production.py.
# 7 NEW loss runs per (architecture, task, seed). CE is REUSED, not rerun.
TREATMENTS = (('physical_mae', None),
              ('ce_physical_mae', 0.1), ('ce_physical_mae', 0.5),
              ('ce_physical_mae', 1.0),
              ('ce_ordinal', 0.1), ('ce_ordinal', 0.5),
              ('ce_ordinal', 1.0))
MATRIX = (('clip','alpha_only','E3'),('clip','q0_only','E4'),
          ('siglip','alpha_only','E2'),('siglip','q0_only','E4'))
SEEDS=(42,123,2026)


def treatment(kind, weight=None):
    if kind == 'physical_mae':
        if weight is not None:
            raise ValueError('Standalone physical MAE has NO lambda')
        spec = {'type': 'physical_mae'}
    elif kind in ('ce_ordinal', 'ce_physical_mae'):
        if type(weight) not in (int, float) or float(weight) not in (0.1, 0.5, 1.0):
            raise ValueError('Exact T04-07 lambda grid is 0.1, 0.5, 1.0')
        spec = {'type':kind, 'ordinal_weight' if kind == 'ce_ordinal' else 'mae_weight': float(weight)}
    else:
        raise ValueError('Unknown T04-07 loss treatment')
    resolve_moment_loss(spec)
    return spec


def matrix(root=ROOT):
    """Return the exact 7 x 4 x 3 = 84 loss matrix; no CE re-training."""
    for model, task, moment in MATRIX:
        for seed in SEEDS:
            config = BASE_CONFIG / f't04_10_{model}_{task}_seed{seed}.yml'
            for kind, weight in TREATMENTS:
                spec = treatment(kind, weight)
                suffix = kind if weight is None else kind + '_lambda' + str(weight).replace('.', 'p')
                run_id = f't04_10_{model}_{task}_seed{seed}_{suffix}_v1'
                yield dict(model=model, task=task, moment=moment, seed=seed,
                           loss=spec, treatment=suffix, config=config,
                           run_id=run_id, run=Path(root)/model/task/moment/f'seed-{seed}'/run_id)


def selected(selector=None, exclude=None, root=ROOT, treatments=None):
    rows=list(matrix(root))
    if treatments:
        selected_treatments=set(treatments)
        known={r['treatment'] for r in rows}
        if selected_treatments - known:
            raise ValueError('Unknown treatment(s): '+repr(sorted(selected_treatments-known)))
        rows=[r for r in rows if r['treatment'] in selected_treatments]
    if selector:
        rows=[r for r in rows if r['run_id']==selector]
    if exclude:
        rows=[r for r in rows if r['run_id']!=exclude]
    if not rows: raise ValueError('Empty selection; use exact run_id from plan')
    if len({r['run_id'] for r in rows}) != len(rows):
        raise AssertionError('Duplicate experiment identity')
    return rows


def make_row(row):
    run=row['run']
    if run.exists():
        verify(run)
        _,payload,_,_=verify_ablation(run,201)
        if payload['loss']!=row['loss']: raise ValueError(f'Loss identity mismatch: {run}')
        return run,'ALREADY_PREPARED'
    from train_utils.t04_10_preflight import load_settings
    cfg=load_settings(row['config'])
    assert cfg['model']==row['model'] and cfg['task']==row['task'] and cfg['seed']==row['seed']
    assert len(cfg['moments'])==int(row['moment'][1:]) and cfg['epochs']==80
    run=prepare(row['config'],run_id=row['run_id'])
    manifest=json.loads((run/'run_manifest.json').read_text())
    payload={'schema':1,'run_id':row['run_id'],'loss':row['loss'],
             'resolved_loss':resolve_moment_loss(row['loss']),
             'source_commit':manifest['git_commit'],
             'config_sha256':manifest['config_sha256'],
             'run_manifest_sha256':sha(run/'run_manifest.json'),
             'baseline_protocol':'same architecture/strategy/moments/seed/splits as T04-10 CE',
             'initialization':'fresh; NOT CE checkpoint fine-tuning',
             'selection_metric':'validation_macro_f1'}
    (run/'loss_ablation_manifest.json').write_text(json.dumps(payload,sort_keys=True,indent=2)+'\n')
    return run,'PREPARED'


def submit_held(row,target):
    # No duplicate sessions, checkpoint/resume within the SAME loss run when applicable.
    verify_ablation(row['run'],target)
    cmd=['sbatch','--parsable','--hold',
         f'--export=ALL,T04_10_LOSS_RUN={row["run"]},T04_10_LOSS_TARGET={target}',
         str(REPO/'slurm/t04_10_loss_ablation.slurm')]
    job=subprocess.check_output(cmd,text=True).strip().split(';')[0]
    if not re.fullmatch(r'\d+',job):
        raise RuntimeError('Unexpected Slurm job id: '+job)
    print('SUBMITTED_HELD',job,row['run_id'],flush=True)
    return job


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('plan','prepare','preview','submit'))
    p.add_argument('--treatment',action='append',help='Filter by treatment suffix (repeatable); default all seven')
    p.add_argument('--target',type=int,default=200)
    p.add_argument('--only',help='Exact run_id; use plan to inspect')
    p.add_argument('--exclude',help='Exclude an already submitted pilot run_id')
    p.add_argument('--apply',action='store_true',help='Required to submit held Slurm jobs')
    args=p.parse_args(argv)
    if type(args.target) is not int or args.target < 1: p.error('Target must be positive')
    rows=selected(args.only,args.exclude,treatments=args.treatment)
    if args.command=='plan':
        for r in rows: print(r['run_id'],r['config'],r['run'])
        print('NEW_RUN_COUNT',len(rows),'CE_BASELINES_REUSED',12,'TOTAL_WITH_CE',len(rows)+12)
        return
    if args.command=='prepare':
        for row in rows:
            run,kind=make_row(row)
            print(kind,run)
        return
    for row in rows:
        if not row['run'].exists():
            raise FileNotFoundError(f'Not prepared: {row["run"]}')
        cfg,payload,parent,loss_hash=verify_ablation(row['run'],args.target)
        if payload['loss'] != row['loss']:
            raise ValueError(f'Loss mismatch: {row["run"]}')
        print('GUARD_PASS',row['run_id'], 'current_epoch',parent['epoch'] if parent else 0)
    if args.command=='preview' or not args.apply:
        print('PREVIEW_ONLY count',len(rows),'no jobs submitted')
        return
    active=subprocess.check_output(['squeue','-h','-u',__import__('os').environ['USER'],
                                     '-n','T04_10_loss','-o','%i'],text=True).strip()
    if active:
        raise RuntimeError('Other T04_10_loss jobs active or held. Avoid duplicate submissions: '+active)
    for row in rows: submit_held(row,args.target)
    print('HELD_ONLY: inspect resources and release explicitly. COUNT',len(rows))

if __name__=='__main__': main()
