"""Generate 84 T04-07 top-four loss ablations from frozen T04-06 CE YAMLs.

No datasets, run reservations, Slurm calls, or GPU operations. Dry-run by default.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import yaml

REPO = Path(__file__).resolve().parents[1]
BASE = Path('experiments/t04/config/t04_06_v2')
TARGET = Path('experiments/t04/config/t04_07_top4')
OUTPUT = '/rs/rs_grp_mljet/ml-jet-ext/training_output/T04/t04_07_top4_production'
MODELS = (('clip','S4',4),('siglip','S4',4),('siglip','S4',5),('clip','S4',2))
SEEDS = (42, 123, 2026)
TREATMENTS = (('physical_mae',None), ('ce_physical_mae',0.1),
              ('ce_physical_mae',0.5), ('ce_physical_mae',1.0),
              ('ce_ordinal',0.1), ('ce_ordinal',0.5), ('ce_ordinal',1.0))
RANKED = (
    {'rank':2,'model':'clip','strategy':'S4','moments':'E4','mean_best_joint_accuracy':0.1914285714,'sample_sd':0.0108091043},
    {'rank':3,'model':'siglip','strategy':'S4','moments':'E4','mean_best_joint_accuracy':0.175,'sample_sd':0.0173205081},
    {'rank':4,'model':'siglip','strategy':'S4','moments':'E5','mean_best_joint_accuracy':0.1733333333,'sample_sd':0.0136713116},
    {'rank':5,'model':'clip','strategy':'S4','moments':'E2','mean_best_joint_accuracy':0.1659523810,'sample_sd':0.0505295091},
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def build(repo=REPO):
    repo=Path(repo).resolve()
    configs=[]; rows=[]
    for model,strategy,k in MODELS:
        for seed in SEEDS:
            template = BASE / f'{model}_{strategy.lower()}_e{k}_seed{seed}.yml'
            src=repo/template
            original=yaml.safe_load(src.read_text())
            expected={'model':model,'strategy':strategy,'moments':list(range(1,k+1)),
                      'seed':seed,'epochs':80,'training_protocol':'fixed_budget_v2',
                      'precision':'bf16','split_protocol':'aggregate_disjoint',
                      'batch_size':8,'gradient_accumulation':1,'moment_loss':{'type':'cross_entropy'}}
            for key, value in expected.items():
                if original.get(key)!=value:
                    raise ValueError(f'{src}: changed baseline {key}: {original.get(key)}')
            for kind,weight in TREATMENTS:
                suffix=kind if weight is None else kind+'_lambda'+str(weight).replace('.','p')
                identity=f'{model}_{strategy.lower()}_e{k}_seed{seed}_{suffix}'
                cfg=dict(original)
                loss={'type':kind}
                if kind=='ce_physical_mae': loss['mae_weight']=weight
                if kind=='ce_ordinal': loss['ordinal_weight']=weight
                cfg['moment_loss']=loss
                cfg['output_root']=OUTPUT
                relative=TARGET/(identity+'.yml')
                raw=yaml.safe_dump(cfg,sort_keys=False).encode()
                # Validate full schema after writing; first check invariants.
                assert set(cfg) == set(original) and cfg['moment_loss']==loss
                configs.append((relative,raw))
                rows.append({'id':identity,'config':str(relative),'config_sha256':sha(raw),
                             'source_baseline':str(template),'source_baseline_sha256':sha(src.read_bytes()),
                             'model':model,'strategy':strategy,'moments':f'E{k}',
                             'seed':seed,'loss':loss})
    if len(rows)!=84 or len({r['id'] for r in rows})!=84:
        raise AssertionError('Expected 84 unique configurations')
    plan={'schema':1,'kind':'T04-07-top4-loss-production-v1','baseline_reused':12,
          'original_top1_already_complete':'clip_s4_e3','expected_new_runs':84,
          'original_ce_ranked':list(RANKED),'output_root':OUTPUT,
          'seeds':list(SEEDS),'treatments':7,'rows':rows}
    return configs,plan


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--apply',action='store_true')
    args=ap.parse_args(argv)
    configs,plan=build()
    dest=REPO/TARGET
    print(json.dumps({'mode':'apply' if args.apply else 'dry-run','new_runs':len(configs),
        'groups':len(MODELS),'seeds':list(SEEDS),'loss_treatments':7,
        'output_root':OUTPUT,'config_dir':str(dest),
        'already_completed_top1':'clip_s4_e3','jobs_submitted':0},indent=2))
    if not args.apply:
        print('DRY RUN ONLY: no files created')
        return
    if dest.exists(): raise FileExistsError('Never overwrite top4 configs: '+str(dest))
    dest.mkdir(parents=True,exist_ok=False)
    from train_utils.t04_preflight import parse_config
    for rel,raw in configs:
        path=REPO/rel
        with path.open('xb') as f:f.write(raw)
        parse_config(path)
    with (dest/'top4_plan.json').open('x') as f:
        json.dump(plan,f,indent=2)
        f.write('\n')
    print('CREATED 84 YAMLs and immutable top4 plan; no run reservations or jobs')

if __name__=='__main__':main()
