"""T04-07 three-treatment GPU smoke: prepare, probe, verify. No implicit launches.

Requires invocation from repository root: python -m tools.t04_07_gpu_smoke ...
Does not alter existing T04-06 or T04-07 production run artifacts.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import yaml

SEED = 42
TREATMENTS = (
    ('physical_mae', None, 'physical_mae'),
    ('ce_physical_mae', 0.5, 'ce_physical_mae_lambda0p5'),
    ('ce_ordinal', 0.5, 'ce_ordinal_lambda0p5'),
)
SOURCE_CONFIG_DIR = Path('experiments/t04/config/t04_07_pilot')
SMOKE_CONFIG_DIR = Path('experiments/t04/config/t04_07_smoke')
SMOKE_ROOT = '/rs/rs_grp_mljet/ml-jet-ext/training_output/T04/t04_07_gpu_smoke'
RUN_IDS = tuple('t04-07-gpu-smoke-clip-s4-e3-seed42-' + suffix for _, _, suffix in TREATMENTS)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def entries(repo, smoke_root):
    from train_utils.t04_preflight import parse_config
    configs = []
    for index, ((kind, weight, suffix), run_id) in enumerate(zip(TREATMENTS, RUN_IDS)):
        src = repo/SOURCE_CONFIG_DIR/f'clip_s4_e3_seed42_{suffix}.yml'
        if not src.is_file():
            raise FileNotFoundError(src)
        data = yaml.safe_load(src.read_text())
        if (data.get('model'), data.get('strategy'), data.get('moments'),
            data.get('seed'), data.get('epochs'), data.get('precision'),
            data.get('training_protocol'), data.get('split_protocol')) != (
            'clip', 'S4', [1,2,3], SEED, 80, 'bf16', 'fixed_budget_v2', 'aggregate_disjoint'):
            raise ValueError('Pilot configuration mismatch: '+str(src))
        loss = {'type': kind}
        if kind == 'ce_physical_mae': loss['mae_weight'] = weight
        if kind == 'ce_ordinal': loss['ordinal_weight'] = weight
        if data.get('moment_loss') != loss:
            raise ValueError('Pilot loss mismatch: '+str(src))
        data['output_root'] = str(smoke_root)
        data['num_workers'] = 0  # smoke runtime: no extra dataloader processes
        # Keep 80-epoch protocol/hash. Execute only 1 epoch, then resume to 2.
        dst = repo/SMOKE_CONFIG_DIR/(suffix+'.yml')
        path = Path(smoke_root).resolve()/'clip'/'S4'/'E3'/'seed-42'/run_id
        configs.append(dict(index=index, loss=loss, source_config=str(src.resolve()),
                            source_sha256=sha(src), config=str(dst.resolve()),
                            run_id=run_id, run=str(path), data=data))
    if len({e['run'] for e in configs}) != 3:
        raise RuntimeError('Nonunique smoke run paths')
    return configs


def create(args):
    from train_utils.t04_preflight import preflight, prepare, verify
    repo = args.repo.resolve()
    root = Path(args.smoke_root).resolve()
    if 't04_07_gpu_smoke' not in str(root) or 't04_06' in str(root):
        raise ValueError('Dedicated t04_07_gpu_smoke output root required')
    obj = entries(repo, root)
    manifest = args.manifest.resolve()
    if manifest.exists() or (repo/SMOKE_CONFIG_DIR).exists():
        raise FileExistsError('Smoke manifest or config directory already exists; refusing overwrite')
    for e in obj:
        if Path(e['run']).exists() or Path(e['config']).exists():
            raise FileExistsError('Smoke run/config already exists: '+e['run'])
    print(json.dumps({'mode':'apply' if args.apply else 'dry-run',
                      'configs': [dict(index=e['index'], loss=e['loss'], run=e['run']) for e in obj],
                      'manifest':str(manifest)}, indent=2))
    if not args.apply:
        print('DRY RUN: no config or run prepared')
        return
    # Configs are independent from committed production pilot YAML.
    (repo/SMOKE_CONFIG_DIR).mkdir(parents=True, exist_ok=False)
    for e in obj:
        config = Path(e['config'])
        with config.open('x') as f:
            yaml.safe_dump(e['data'], f, sort_keys=False)
        preflight(config)
    for e in obj:
        path = Path(prepare(e['config'], e['run_id'])).resolve()
        if path != Path(e['run']):
            raise RuntimeError('Unexpected prepared run path: '+str(path))
        verify(path)
        if (path/'source/config.original.yml').read_bytes() != Path(e['config']).read_bytes():
            raise RuntimeError('Frozen original config mismatch')
        print('PREPARED', path, flush=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    serializable = [{k:v for k,v in e.items() if k!='data'} for e in obj]
    with manifest.open('x') as f:
        json.dump({'schema':1,'scope':'T04-07 three-run smoke only',
                   'configured_epochs':80,'fresh_target':1,'resume_target':2,
                   'seed':SEED,'entries':serializable},f,indent=2)
        f.write('\n')
    print('MANIFEST',manifest)


def load(args):
    from train_utils.t04_preflight import verify
    data = json.loads(args.manifest.resolve().read_text())
    if data.get('schema') != 1 or data.get('scope') != 'T04-07 three-run smoke only':
        raise ValueError('Unexpected smoke manifest')
    e = data['entries']
    if len(e) != 3 or [row['run_id'] for row in e] != list(RUN_IDS):
        raise ValueError('Unexpected smoke manifest entries')
    for row in e:
        verify(row['run'])
        source = Path(row['source_config'])
        if sha(source) != row['source_sha256']:
            raise ValueError('Original production pilot YAML changed: '+str(source))
        if (Path(row['run'])/'source/config.original.yml').read_bytes() != Path(row['config']).read_bytes():
            raise ValueError('Frozen configuration differs: '+row['run'])
    return e


def get_run(args):
    e=load(args)
    if not (0 <= args.index < 3): raise ValueError('Index must be 0, 1, or 2')
    print(e[args.index]['run'])


def probe(args):
    import torch
    from train_utils.ordinal_loss import build_moment_criterion
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA allocation required for BF16 probe')
    print('CUDA_DEVICE', torch.cuda.get_device_name(0))
    for kind, weight, suffix in TREATMENTS:
        spec={'type':kind}
        if kind == 'ce_physical_mae': spec['mae_weight']=weight
        if kind == 'ce_ordinal': spec['ordinal_weight']=weight
        losses=build_moment_criterion(spec,device='cuda')
        for head,n_classes in [('alpha_output',7),('q0_output',5)]:
            logits=torch.randn(8,n_classes,device='cuda',dtype=torch.float32,requires_grad=True)
            labels=torch.arange(8,device='cuda')%n_classes
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                total=losses[head](logits, labels)
            total.backward()
            if not (torch.isfinite(total) and torch.isfinite(logits.grad).all() and
                    logits.grad.abs().sum()>0):
                raise RuntimeError(f'Nonfinite/zero gradients: {suffix}, {head}')
            print('PASS_PROBE', suffix, head, 'loss=',float(total.detach()))
    print('PASS: BF16 CUDA loss forward/backward for 3 losses x 2 heads')


def check(args):
    from train_utils.t04_sessions import latest_checkpoint
    e = load(args)
    expected = args.expected_epoch
    if expected not in (0,1,2): raise ValueError('Expected epoch must be 0, 1, or 2')
    for row in e:
        run=Path(row['run'])
        ptr=latest_checkpoint(run)
        if expected == 0:
            if ptr is not None or (run/'sessions').exists():
                raise RuntimeError('Unexpected execution before fresh smoke: '+str(run))
            print('PASS_PREPARED',row['run_id'])
            continue
        if ptr is None or ptr['epoch']!=expected:
            raise RuntimeError(f'Expected checkpoint epoch {expected}: {run}')
        session_dir=run/'sessions'
        sessions=sorted(session_dir.glob('session-*'))
        if len(sessions)!=expected:
            raise RuntimeError(f'Expected {expected} completed sessions: {run}')
        for s in sessions:
            m=json.loads((s/'session_manifest.json').read_text())
            if m.get('status')!='completed':
                raise RuntimeError('Incomplete session: '+str(s))
            if not (s/'result.json').is_file():
                raise RuntimeError('Missing result.json: '+str(s))
            if not (s/'runtime_environment.json').is_file():
                raise RuntimeError('Missing runtime environment: '+str(s))
            env=json.loads((s/'runtime_environment.json').read_text())
            if env.get('moment_loss') != row['loss']:
                raise RuntimeError('Runtime loss differs from selected YAML: '+str(s))
        with (run/'moment_epoch_metrics.csv').open(newline='') as f:
            rows=list(csv.DictReader(f))
        if len(rows)!=expected or [int(x['epoch']) for x in rows] != list(range(1,expected+1)):
            raise RuntimeError('Missing/repeated epoch rows: '+str(run))
        required=('train_loss','val_loss','val_joint_accuracy','val_alpha_mae','val_q0_mae',
                  'val_alpha_accuracy','val_q0_accuracy')
        for item in rows:
            for key in required:
                value=float(item[key])
                if not math.isfinite(value):
                    raise RuntimeError(f'Invalid {key}: {run}')
            for tag in ('alpha','q0'):
                if row['loss']['type']=='physical_mae':
                    components=['physical_mae']
                elif row['loss']['type']=='ce_physical_mae':
                    components=['ce','physical_mae']
                else:
                    components=['ce','ordinal']
                for split in ('train','val'):
                    for component in components:
                        key=f'{split}_{tag}_{component}_loss'
                        if key not in item or not item[key] or not math.isfinite(float(item[key])):
                            raise RuntimeError(f'Missing/nonfinite loss component {key}: {run}')
        if expected==2:
            parent=json.loads((sessions[1]/'session_manifest.json').read_text())['parent_checkpoint']
            if parent is None or parent['epoch']!=1:
                raise RuntimeError('Resume provenance not anchored to epoch 1: '+str(run))
        print('PASS',row['run_id'], 'epoch',expected,'sessions',len(sessions),
              'best_val_joint',max(float(x['val_joint_accuracy']) for x in rows))
    print(f'PASS: 3/3 immutable smoke runs at epoch {expected}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path.cwd())
    parser.add_argument('--smoke-root',type=Path,default=Path(SMOKE_ROOT))
    parser.add_argument('--manifest',type=Path,default=Path(SMOKE_ROOT)/'smoke_manifest.json')
    sub=parser.add_subparsers(dest='cmd',required=True)
    sub.add_parser('prepare').add_argument('--apply',action='store_true')
    sub.add_parser('probe')
    sub.add_parser('get-run').add_argument('--index',type=int,required=True)
    sub.add_parser('verify').add_argument('--expected-epoch',type=int,required=True)
    args=parser.parse_args()
    {'prepare':create,'probe':probe,'get-run':get_run,'verify':check}[args.cmd](args)

if __name__=='__main__': main()
