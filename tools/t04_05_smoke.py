"""T04-05: independently prepare and verify eight immutable E5 BF16 smoke runs."""
import argparse
import json
from pathlib import Path

import yaml

from train_utils.t04_preflight import preflight, prepare, verify
from train_utils.t04_sessions import latest_checkpoint

MODELS = ('clip', 'siglip')
STRATEGIES = ('s1', 's2', 's3', 's4')


def make_config(base, output):
    data = yaml.safe_load(base.read_text())
    assert data['model'] in MODELS
    assert data['strategy'].lower() in STRATEGIES
    assert data['moments'] == [1, 2, 3, 4, 5]
    assert data['seed'] == 42
    assert data['precision'] == 'bf16'
    assert data['optimizer']['type'] == 'adamw'
    assert data['split_protocol'] == 'aggregate_disjoint'
    data.update(batch_size=8, gradient_accumulation=1, epochs=2, min_epochs=1, patience=5,
                gpu_profile='a100_mig_1g_10gb')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
    return data


def create(repo, manifest):
    entries = []
    config_dir = repo / 'experiments/t04/config/t04_05_smoke'
    for model in MODELS:
        for strategy in STRATEGIES:
            name = '%s_%s_e5_seed42' % (model, strategy)
            source = repo / 'experiments/t04/config/t04_05' / (name + '.yml')
            dest = config_dir / (name + '.yml')
            if not dest.exists():
                make_config(source, dest)
            cfg, _ = preflight(dest)
            if (cfg.model, cfg.strategy.lower(), cfg.batch_size, cfg.gradient_accumulation,
                    cfg.epochs, cfg.precision) != (model, strategy, 8, 1, 2, 'bf16'):
                raise ValueError('Smoke config mismatch: ' + str(dest))
            run_id = 't04-05-%s-%s-e5-seed42-smoke-01' % (model, strategy)
            expected = (Path(cfg.output_root).expanduser().resolve() / model /
                        strategy.upper() / 'E5' / 'seed-42' / run_id)
            if expected.exists():
                verify(expected)
                # Never adopt an existing experiment based on path alone.
                original = expected / 'source/config.original.yml'
                if not original.is_file() or original.read_bytes() != dest.read_bytes():
                    raise ValueError('Existing run config mismatch: ' + str(expected))
                print('VERIFIED_EXISTING', expected, flush=True)
            else:
                actual = Path(prepare(dest, run_id)).resolve()
                if actual != expected:
                    raise RuntimeError('Unexpected output path')
                verify(actual)
                print('PREPARED', actual, flush=True)
            entries.append({'model': model, 'strategy': strategy.upper(),
                            'config': str(dest.resolve()), 'run': str(expected)})
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open('x') as f:
        json.dump({'schema': 1, 'entries': entries, 'smoke': {'precision': 'bf16',
            'microbatch': 8, 'accumulation': 1, 'epochs': 2, 'seed': 42}}, f, indent=2)
        f.write('\n')
    print('MANIFEST', manifest, flush=True)


def load(manifest):
    obj = json.loads(manifest.read_text())
    entries = obj['entries']
    if len(entries) != 8 or [(e['model'], e['strategy'].lower()) for e in entries] != [
            (model, strategy) for model in MODELS for strategy in STRATEGIES]:
        raise ValueError('Expected ordered 8-run matrix')
    for entry in entries:
        verify(entry['run'])
    return entries


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--repo', type=Path, default=Path.cwd())
    p.add_argument('--manifest', type=Path, required=True)
    p = sub.add_parser('get-run')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--index', type=int, required=True)
    p = sub.add_parser('verify')
    p.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        create(args.repo.resolve(), args.manifest.resolve())
    elif args.command == 'get-run':
        entries = load(args.manifest.resolve())
        if args.index not in range(8):
            raise ValueError('index must be 0..7')
        print(entries[args.index]['run'])
    else:
        entries = load(args.manifest.resolve())
        for idx, item in enumerate(entries):
            pointer = latest_checkpoint(item['run'])
            print(idx, item['model'], item['strategy'], 'epoch=',
                  pointer['epoch'] if pointer else 'not_started')
        print('PASS: eight frozen run manifests')


if __name__ == '__main__':
    main()
