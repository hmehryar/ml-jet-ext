"""Generate exactly 21 non-CE CLIP S4/E3 YAMLs; no training or preparation."""
from pathlib import Path
import argparse, json, hashlib
import yaml

SEEDS=(42,123,2026)
TREATMENTS=(('physical_mae',None),('ce_physical_mae',0.1),('ce_physical_mae',0.5),('ce_physical_mae',1.0),('ce_ordinal',0.1),('ce_ordinal',0.5),('ce_ordinal',1.0))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path.cwd())
    p.add_argument('--output-root',default='/rs/rs_grp_mljet/ml-jet-ext/training_output/T04/t04_07_loss_pilot')
    p.add_argument('--apply',action='store_true',help='Create configs; dry-run otherwise')
    args=p.parse_args(); repo=args.repo.resolve()
    base_dir=repo/'experiments/t04/config/t04_06_v2'
    out=repo/'experiments/t04/config/t04_07_pilot'
    if out.exists(): raise FileExistsError(f'Pilot directory exists; never overwrite: {out}')
    if 't04_06' in args.output_root.lower() or 't04_07' not in args.output_root.lower():
        raise ValueError('Output root must be dedicated to t04_07 and not t04_06')
    rows=[]; configs=[]
    for seed in SEEDS:
        source=base_dir/f'clip_s4_e3_seed{seed}.yml'
        base=yaml.safe_load(source.read_text())
        expected={'model':'clip','strategy':'S4','moments':[1,2,3], 'seed':seed,
                  'epochs':80,'training_protocol':'fixed_budget_v2'}
        for key,value in expected.items():
            if base.get(key)!=value: raise ValueError(f'Source config {source} unexpected {key}: {base.get(key)}')
        if base.get('moment_loss',{}).get('type')!='cross_entropy': raise ValueError('Source is not CE baseline')
        if base.get('split_protocol')!='aggregate_disjoint': raise ValueError('Source split protocol changed')
        for kind,weight in TREATMENTS:
            suffix=kind if weight is None else kind+'_lambda'+str(weight).replace('.','p')
            identity=f'clip_s4_e3_seed{seed}_{suffix}'
            cfg=dict(base)
            cfg['moment_loss']={'type':kind}
            if kind=='ce_physical_mae': cfg['moment_loss']['mae_weight']=weight
            if kind=='ce_ordinal': cfg['moment_loss']['ordinal_weight']=weight
            cfg['output_root']=args.output_root
            from train_utils.t04_preflight import parse_config
            # Strict schema validation without accessing Ceph data or reserving a run.
            name=identity+'.yml'
            payload=yaml.safe_dump(cfg,sort_keys=False)
            rows.append({'id':identity,'path':str(out/name),'source':str(source),'seed':seed,
                        'loss':cfg['moment_loss'],'sha256':hashlib.sha256(payload.encode()).hexdigest()})
            configs.append((name,payload))
    if len(configs)!=21 or len({x[0] for x in configs})!=21: raise AssertionError('Invalid pilot matrix')
    print(json.dumps({'config_count':len(configs),'output':str(out),'treatments':len(TREATMENTS),'seeds':SEEDS,'mode':'apply' if args.apply else 'dry-run'},indent=2))
    if args.apply:
        out.mkdir(parents=True,exist_ok=False)
        for name,payload in configs: (out/name).open('x').write(payload)
        (out/'pilot_plan.json').open('x').write(json.dumps({'schema':1,'baseline':'clip_s4_e3','baseline_seeds':list(SEEDS),
            'baseline_reused':3,'new_config_count':21,'rows':rows,'no_test_selection':True},indent=2)+'\n')
        for row in rows: parse_config(Path(row['path']))
        print('Created 21 schema-validated configs, no training runs and no reserved experiments')
    else: print('DRY RUN ONLY: No files created')

if __name__=='__main__': main()
