"""T04-10 Stage A: 10 ViT-Tiny scratch CE experiments (E1-E5 x 2 tasks).

Two epochs for smoke testing only. These exact runs may resume to 40/80/200
without restarting. Held submission only; never use pilot score to rank E.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import yaml
from train_utils.t04_10_preflight import prepare, verify, sha256
from train_utils.t04_10_sessions import latest_checkpoint
from train_utils.loss_config import resolve_moment_loss

REPO=Path(__file__).resolve().parents[1]
SEEDS=(42,)
TASKS=('alpha_only','q0_only')
E_VALUES=(1,2,3,4,5)
TREATMENTS=(('ce',None),)  # Stage A intentionally excludes loss ablations
CONFIG_DIR=REPO/'experiments/t04/config/t04_10_vit_e_screen'
OUTPUT_ROOT=Path('/rs/rs_grp_mljet/ml-jet-ext/training_output/T04/t04_10_single_head')
BASE_TEMPLATE=REPO/'experiments/t04/config/t04_10_stage4_frozen/t04_10_clip_alpha_only_seed42.yml'
SCRATCH_MODEL = 'timm/vit_tiny_patch16_224_scratch'
SCRATCH_REVISION = '0' * 40

def loss_spec(kind,lam):
    if kind in ('ce','physical_mae'):
        if lam is not None:raise ValueError('Unexpected lambda for unweighted loss')
        return {'type':'cross_entropy' if kind=='ce' else kind}
    if kind not in ('ce_physical_mae','ce_ordinal') or lam not in (0.1,0.5,1.0):
        raise ValueError('Unknown treatment/weight')
    return {'type':kind,'mae_weight' if kind=='ce_physical_mae' else 'ordinal_weight':float(lam)}

def rows():
    for task in TASKS:
        for e in E_VALUES:
            for seed in SEEDS:
                kind, weight = TREATMENTS[0]
                run_id=f't04_10_vit_{task}_e{e}_seed{seed}_scratch_s4_ce_v1'
                yield dict(task=task, e=e, seed=seed, kind=kind, weight=weight,
                    treatment='ce', loss=loss_spec(kind, weight), run_id=run_id,
                    config=CONFIG_DIR/f't04_10_vit_{task}_e{e}_seed{seed}.yml',
                    run=OUTPUT_ROOT/'vit'/task/f'E{e}'/f'seed-{seed}'/run_id)

def selected(only=None,exclude=None,treatments=None):
    if only is not None and not only.strip():raise ValueError('Empty --only forbidden')
    excludes = [] if exclude is None else ([exclude] if isinstance(exclude,str) else list(exclude))
    if any(not x or not x.strip() for x in excludes):raise ValueError('Empty --exclude forbidden')
    all_rows=list(rows())
    known={r['run_id'] for r in all_rows}
    if only is not None and only not in known:raise ValueError('Unknown --only run')
    if set(excludes) - known:raise ValueError('Unknown --exclude run')
    if treatments:
        possible={r['treatment'] for r in all_rows}
        if set(treatments)-possible:raise ValueError('Unknown --treatment')
    chosen=[r for r in all_rows if (only is None or only==r['run_id'])
        and r['run_id'] not in excludes
        and (not treatments or r['treatment'] in treatments)]
    if not chosen:raise ValueError('Empty run selection')
    return chosen

def generate():
    base=yaml.safe_load(BASE_TEMPLATE.read_text())
    if base['model']!='clip' or base['task']!='alpha_only':
        raise ValueError('Unexpected baseline template')
    CONFIG_DIR.mkdir(parents=True,exist_ok=True)
    for task in TASKS:
        for e in E_VALUES:
            for seed in SEEDS:
                cfg=dict(base)
                cfg.update(model='vit', task=task, seed=seed, moments=list(range(1,e+1)),
                    strategy='S4',checkpoint=SCRATCH_MODEL,
                    checkpoint_revision=SCRATCH_REVISION,
                    vit_initialization='random',num_workers=1,
                    learning_rates={'backbone':1e-4,'adapter':1e-4,'heads':1e-4})
                path=CONFIG_DIR/f't04_10_vit_{task}_e{e}_seed{seed}.yml'
                data=yaml.safe_dump(cfg,sort_keys=False)
                if path.exists():
                    if path.read_text()!=data:
                        raise FileExistsError(f'Frozen config differs: {path}')
                    print('UNCHANGED',path)
                else:
                    path.write_text(data)
                    print('GENERATED',path)
    print('STAGE_A_CONFIGS',len(TASKS)*len(E_VALUES)*len(SEEDS),
          'INITIALIZATION=random PRETRAINED=false LOSS=CE')
    print('Commit source and YAMLs BEFORE preparing immutable runs.')

def loss_manifest(row,run):
    manifest=json.loads((run/'run_manifest.json').read_text())
    spec=row['loss']
    return dict(schema=1,run_id=row['run_id'],loss=spec,
        resolved_loss=resolve_moment_loss(spec),source_commit=manifest['git_commit'],
        config_sha256=manifest['config_sha256'],run_manifest_sha256=sha256(run/'run_manifest.json'),
        initialization='random ViT-Tiny; no pretrained or transfer checkpoint',
        stage='A_E1_E5_two_epoch_smoke_not_model_selection',
        selection_metric='validation_macro_f1')

def verify_row(row,target,require_manifest=True):
    from train_utils.t04_10_vit_study_run import validate
    run=Path(row['run'])
    cfg,lm,pointer,loss_hash=validate(run,target)
    if (cfg['task']!=row['task'] or cfg['seed']!=row['seed']
        or len(cfg['moments'])!=row['e'] or lm['loss']!=row['loss']):
        raise ValueError('Matrix identity mismatch')
    return pointer['epoch'] if pointer else 0

def prepare_row(row):
    run=Path(row['run'])
    if run.exists():
        verify_row(row,201)
        return 'ALREADY_PREPARED',run
    if not row['config'].is_file():raise FileNotFoundError(f'First generate YAML: {row["config"]}')
    run=prepare(row['config'],run_id=row['run_id'])
    if run.resolve()!=row['run'].resolve():raise RuntimeError('Prepared run directory mismatch')
    p=run/'vit_loss_manifest.json'
    p.write_text(json.dumps(loss_manifest(row,run),sort_keys=True,indent=2)+'\n')
    return 'PREPARED',run

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('generate','plan','prepare','preview','submit'))
    p.add_argument('--only')
    p.add_argument('--exclude',action='append',help='Exact run_id to exclude; repeatable')
    p.add_argument('--treatment',action='append')
    p.add_argument('--target',type=int,default=2)
    p.add_argument('--apply',action='store_true')
    a=p.parse_args(argv)
    if a.command=='generate':
        if a.only is not None or a.exclude is not None or a.treatment:p.error('Generate always freezes all six YAML files')
        generate()
        return
    if a.target<1:p.error('Target must be positive')
    chosen=selected(a.only,a.exclude,a.treatment)
    if a.command=='plan':
        for r in chosen:print(r['run_id'],r['config'],r['run'])
        print('STAGE_A_TOTAL',len(chosen),'E_VALUES',E_VALUES,'SEEDS',SEEDS,'LOSS=CE')
        return
    if a.command=='prepare':
        for r in chosen:
            status,run=prepare_row(r)
            print(status,run,flush=True)
        return
    for r in chosen:
        epoch=verify_row(r,a.target)
        print('GUARD_PASS',r['run_id'],'current_epoch',epoch,flush=True)
    if a.command=='preview' or not a.apply:
        print('PREVIEW_ONLY',len(chosen),'no Slurm submission')
        return
    active=subprocess.check_output(['squeue','-h','-u',os.environ['USER'],
        '-n','T04_10_vitA','-o','%i'],text=True).strip()
    if active:raise RuntimeError('ViT Stage A jobs already queued; never double-submit: '+active)
    for row in chosen:
        command=['sbatch','--parsable','--hold',
            f'--export=ALL,T04_10_VIT_RUN={row["run"]},T04_10_VIT_TARGET={a.target}',
            str(REPO/'slurm/t04_10_vit_single.slurm')]
        job=subprocess.check_output(command,text=True).strip().split(';')[0]
        if not re.fullmatch(r'\d+',job):raise RuntimeError('Unexpected Slurm ID: '+job)
        print('SUBMITTED_HELD',job,row['run_id'],flush=True)
    print('HELD_ONLY',len(chosen),'review resources and release deliberately')

if __name__=='__main__':main()
