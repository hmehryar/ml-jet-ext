"""T04-06 Stage 1: deterministic, CPU-only, fail-closed factorial planner.

No run reservation, checkpoint access, GPU initialization, or Slurm submission.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import yaml

MODELS = ('clip', 'siglip')
STRATEGIES = ('S1', 'S2', 'S3', 'S4')
SEEDS = (42, 123, 2026)
SCHEMA = 1


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def stable_yaml(config):
    return yaml.safe_dump(config, sort_keys=False).encode()


def plan(repo, *, seeds=SEEDS, epochs=80, min_epochs=12, patience=16,
         output_root='training_output/T04/t04_06_production'):
    repo = Path(repo).resolve()
    if len(seeds) != 3 or len(set(seeds)) != 3 or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError('Exactly three distinct nonnegative integer seeds required')
    if not (1 <= min_epochs <= epochs and patience >= 1):
        raise ValueError('Invalid training horizon')
    rows = []
    for model in MODELS:
        source = repo / 'experiments/t04/config/t04_05_smoke' / f'{model}_s2_e5_seed42.yml'
        if not source.is_file():
            raise FileNotFoundError(source)
        base = yaml.safe_load(source.read_text())
        if (base.get('model'), base.get('precision'), base.get('moments'),
                base.get('batch_size'), base.get('gradient_accumulation'),
                base.get('split_protocol')) != (model, 'bf16', [1,2,3,4,5],8,1,'aggregate_disjoint'):
            raise ValueError(f'Unverified T04-05 template assumptions: {source}')
        for strategy in STRATEGIES:
            for k in range(1, 6):
                for seed in seeds:
                    config = dict(base)
                    config.update(strategy=strategy, moments=list(range(1,k+1)), seed=seed,
                                  epochs=epochs, min_epochs=min_epochs, patience=patience,
                                  output_root=output_root)
                    identity = f'{model}_{strategy.lower()}_e{k}_seed{seed}'
                    run_id = f't04-06-{identity.replace("_", "-")}-v1'
                    path = f'experiments/t04/config/t04_06/{identity}.yml'
                    run = str(((repo / output_root).expanduser().resolve() / model / strategy /
                               f'E{k}' / f'seed-{seed}' / run_id))
                    data = stable_yaml(config)
                    rows.append(dict(id=identity, model=model, strategy=strategy,
                        moments=f'E{k}', seed=seed, config=path, config_sha256=digest(data),
                        run_id=run_id, run=run, status='planned', slurm_job_id=None,
                        source_template=str(source.relative_to(repo)),
                        source_template_sha256=digest(source.read_bytes())))
    if len(rows) != 120 or len({r['id'] for r in rows}) != 120 or len({r['run'] for r in rows}) != 120:
        raise RuntimeError('Factorial count or identity collision')
    return rows


def _atomic_new(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(payload)


def generate(repo, *, config_dir=None, registry=None, seeds=SEEDS,
             epochs=80, min_epochs=12, patience=16):
    repo = Path(repo).resolve()
    config_dir = Path(config_dir).resolve() if config_dir else repo/'experiments/t04/config/t04_06'
    registry = Path(registry).resolve() if registry else repo/'training_output/T04/t04_06/registry.json'
    rows = plan(repo,seeds=seeds,epochs=epochs,min_epochs=min_epochs,patience=patience)
    if registry.exists():
        raise FileExistsError(f'Refusing existing registry: {registry}')
    for r in rows:
        p = config_dir / (r['id']+'.yml')
        if p.exists():
            raise FileExistsError(f'Refusing existing config: {p}')
        if Path(r['run']).exists():
            raise FileExistsError(f'Refusing reserved output path: {r["run"]}')
    # Precompute full configs before first write; fail on unexpected template or collision.
    all_files=[]
    for row in rows:
        base=yaml.safe_load((repo/row['source_template']).read_text())
        base.update(strategy=row['strategy'], moments=list(range(1,int(row['moments'][1:])+1)),
                    seed=row['seed'], epochs=epochs,min_epochs=min_epochs,patience=patience,
                    output_root='training_output/T04/t04_06_production')
        payload=stable_yaml(base)
        if digest(payload)!=row['config_sha256']:
            raise RuntimeError('Unexpected generator hash mismatch')
        all_files.append((config_dir/(row['id']+'.yml'),payload))
    for path,payload in all_files:
        _atomic_new(path,payload)
    payload={'schema':SCHEMA,'kind':'T04-06-plan','seeds':list(seeds),
             'epochs':epochs,'min_epochs':min_epochs,'patience':patience,
             'entries':rows}
    payload['plan_sha256']=digest(canonical(rows))
    _atomic_new(registry,(json.dumps(payload,indent=2)+'\n').encode())
    return registry


def verify(repo, registry):
    repo=Path(repo).resolve()
    doc=json.loads(Path(registry).read_text())
    rows=doc['entries']
    if doc.get('schema')!=SCHEMA or len(rows)!=120 or digest(canonical(rows))!=doc['plan_sha256']:
        raise ValueError('Invalid/tampered T04-06 plan registry')
    expected=plan(repo,seeds=tuple(doc['seeds']),epochs=doc['epochs'],
                  min_epochs=doc['min_epochs'],patience=doc['patience'])
    if rows != expected:
        raise ValueError('Registry differs from deterministic matrix or templates')
    for row in rows:
        config=repo/row['config']
        if not config.is_file() or digest(config.read_bytes())!=row['config_sha256']:
            raise ValueError(f'Config missing or hash mismatch: {config}')
        if Path(row['run']).exists():
            raise ValueError(f'Output path already reserved (plan-only registry): {row["run"]}')
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('dry-run','generate','verify'))
    p.add_argument('--repo',type=Path,default=Path.cwd())
    p.add_argument('--registry',type=Path)
    p.add_argument('--epochs',type=int,default=80)
    p.add_argument('--min-epochs',type=int,default=12)
    p.add_argument('--patience',type=int,default=16)
    args=p.parse_args()
    if args.command=='dry-run':
        rows=plan(args.repo,epochs=args.epochs,min_epochs=args.min_epochs,patience=args.patience)
        for i,row in enumerate(rows):
            print(f'{i:03d} {row["id"]:26} status=planned run={row["run"]}')
        print(f'PASS: {len(rows)} unique configurations. No files written, no jobs reserved or submitted.')
    elif args.command=='generate':
        print('REGISTRY',generate(args.repo,registry=args.registry,epochs=args.epochs,
             min_epochs=args.min_epochs,patience=args.patience))
    else:
        registry=args.registry or args.repo/'training_output/T04/t04_06/registry.json'
        print(f'PASS: {len(verify(args.repo,registry))} configs and paths are collision-free and hash-verified')

if __name__=='__main__':
    main()
