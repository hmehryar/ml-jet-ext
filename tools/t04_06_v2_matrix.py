"""Independent fixed_budget_v2 120-run planner; NO Slurm calls or run reservations.

Generates a fresh plan and config tree, never touching t04_06 legacy outputs.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import yaml

from tools.t04_06_matrix import MODELS, STRATEGIES, SEEDS, canonical, digest, stable_yaml
from train_utils.t04_training_policy import policy, PROTOCOL_ID

SCHEMA = 1
KIND = 'T04-06-v2-plan'
CONFIG_DIR = Path('experiments/t04/config/t04_06_v2')
PLAN = Path('training_output/T04/t04_06_v2/registry.json')
RUNTIME = Path('training_output/T04/t04_06_v2/registry.runtime.json')
OUTPUT = 'training_output/T04/t04_06_v2_production'
LEGACY_ROOT = Path('training_output/T04/t04_06_production')


def build(repo, *, epochs=80, seeds=SEEDS):
    repo = Path(repo).resolve()
    protocol = policy(epochs)
    if tuple(seeds) != SEEDS:
        raise ValueError('T04-06 v2 seeds are pinned to 42,123,2026')
    rows, files = [], {}
    for model in MODELS:
        template = Path('experiments/t04/config/t04_05_smoke') / f'{model}_s2_e5_seed42.yml'
        path = repo / template
        if not path.is_file():
            raise FileNotFoundError(path)
        base = yaml.safe_load(path.read_text())
        if not isinstance(base, dict) or (base.get('model'), base.get('precision'),
             base.get('moments'), base.get('batch_size'), base.get('gradient_accumulation'),
             base.get('split_protocol')) != (model, 'bf16', [1,2,3,4,5],8,1,'aggregate_disjoint'):
            raise ValueError(f'Unverified T04-05 source template: {path}')
        for strategy in STRATEGIES:
            for k in range(1, 6):
                for seed in seeds:
                    identity = f'{model}_{strategy.lower()}_e{k}_seed{seed}'
                    run_id = f't04-06-{identity.replace("_", "-")}-v2'
                    conf = dict(base)
                    conf.update(strategy=strategy, moments=list(range(1,k+1)), seed=seed,
                                epochs=epochs, min_epochs=min(12, epochs), patience=16, output_root=OUTPUT,
                                training_protocol=PROTOCOL_ID,
                                training_protocol_hash=protocol.digest())
                    # Required legacy schema fields remain present but do not activate early stop.
                    if not (isinstance(conf.get('min_epochs'), int) and
                            isinstance(conf.get('patience'), int)):
                        raise ValueError('Missing required baseline training fields')
                    if conf['min_epochs'] > epochs:
                        raise ValueError('Configured epoch budget smaller than min_epochs')
                    config = CONFIG_DIR / (identity + '.yml')
                    payload = stable_yaml(conf)
                    run = (repo / OUTPUT / model / strategy / f'E{k}' /
                           f'seed-{seed}' / run_id).resolve()
                    if str(run).startswith(str((repo / LEGACY_ROOT).resolve()) + '/'):
                        raise RuntimeError('v2 run collides with legacy prefix')
                    row = dict(id=identity, model=model, strategy=strategy, moments=f'E{k}',
                               seed=seed, config=str(config), config_sha256=digest(payload),
                               run_id=run_id, run=str(run), status='planned',slurm_job_id=None,
                               source_template=str(template),
                               source_template_sha256=digest(path.read_bytes()))
                    rows.append(row)
                    files[repo / config] = payload
    if len(rows)!=120 or len(files)!=120 or len({r['run'] for r in rows})!=120:
        raise RuntimeError('Invalid or duplicate v2 factorial matrix')
    doc = dict(schema=SCHEMA, kind=KIND, seeds=list(seeds), epochs=epochs,
               training_protocol=PROTOCOL_ID, training_protocol_hash=protocol.digest(),
               output_root=OUTPUT, entries=rows, plan_sha256=digest(canonical(rows)))
    return doc, files


def verify(repo, registry):
    repo = Path(repo).resolve()
    doc = json.loads(Path(registry).read_text())
    if doc.get('kind') != KIND or doc.get('schema') != SCHEMA:
        raise ValueError('Invalid v2 registry type/schema')
    expected, files = build(repo,epochs=doc.get('epochs'),seeds=tuple(doc.get('seeds',[])))
    if doc != expected:
        raise ValueError('v2 plan differs from immutable template-derived matrix')
    for path, payload in files.items():
        if not path.is_file() or path.read_bytes()!=payload:
            raise ValueError(f'v2 config changed/missing: {path}')
    # Existing prepared runs are valid during production.
    # Operational state and run provenance are checked separately.
    return doc['entries']


def generate(repo, *, epochs=80, registry=None):
    repo = Path(repo).resolve()
    dest = Path(registry).resolve() if registry else repo / PLAN
    # Require the standard canonical plan location; no arbitrary path aliases.
    if dest != (repo / PLAN).resolve():
        raise ValueError('v2 plan must use its dedicated canonical location')
    doc, files = build(repo,epochs=epochs)
    collisions = [str(p) for p in [dest,repo / RUNTIME,*files]
                  if p.exists()]
    collisions += [r['run'] for r in doc['entries'] if Path(r['run']).exists()]
    if collisions:
        raise FileExistsError('Refusing existing v2 output: '+str(collisions[:4]))
    # Exclusive file creation. No rewriting existing artifacts.
    # Directory creation may remain after a failure; rerunning is deliberately blocked.
    for path,payload in files.items():
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:
            stream.write(payload)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('xb') as stream:
        stream.write((json.dumps(doc,indent=2)+'\n').encode())
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['dry-run','generate','verify'])
    parser.add_argument('--repo',type=Path,default=Path.cwd())
    parser.add_argument('--epochs',type=int,default=80)
    args = parser.parse_args()
    if args.command=='dry-run':
        doc,_=build(args.repo,epochs=args.epochs)
        print('V2_MATRIX',len(doc['entries']),'EPOCHS',doc['epochs'],
              'PROTOCOL_SHA256',doc['training_protocol_hash'])
        print('PLAN',args.repo.resolve()/PLAN)
        print('OUTPUT_ROOT',args.repo.resolve()/OUTPUT)
        print('FIRST',doc['entries'][0]['id'],'LAST',doc['entries'][-1]['id'])
        print('DRY_RUN only; no files or jobs created')
    elif args.command=='generate':
        print('GENERATED',generate(args.repo,epochs=args.epochs))
    else:
        print('VERIFIED',len(verify(args.repo,args.repo.resolve()/PLAN)),'v2 configs')

if __name__=='__main__': main()
