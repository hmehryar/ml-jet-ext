"""Generate eight T04-05 E5 strategy configurations from the verified T04-04 pilots.

CPU-only; generation does not prepare/reserve immutable experiments.
"""
import argparse
from pathlib import Path
import yaml

MODELS = ('clip', 'siglip')
STRATEGIES = ('S1', 'S2', 'S3', 'S4')


def generate(repo, *, output=None):
    repo = Path(repo).resolve()
    base = repo / 'experiments/t04/config'
    output = Path(output).resolve() if output else base / 't04_05'
    output.mkdir(parents=True, exist_ok=True)
    created = []
    for model in MODELS:
        src = base / ('t04_04_%s_s2_bf16_pilot.yml' % model)
        if not src.is_file():
            raise FileNotFoundError('Missing verified T04-04 template: %s' % src)
        template = yaml.safe_load(src.read_text())
        if not isinstance(template, dict) or template.get('model') != model:
            raise ValueError('Invalid pilot template: %s' % src)
        for key, expected in [('moments', [1, 2, 3, 4, 5]), ('precision', 'bf16'),
                              ('split_protocol', 'aggregate_disjoint')]:
            if template.get(key) != expected:
                raise ValueError('Unexpected %s in %s' % (key, src))
        if template.get('optimizer', {}).get('type') != 'adamw':
            raise ValueError('Requires AdamW template')
        for strategy in STRATEGIES:
            config = dict(template)
            config['strategy'] = strategy
            config['seed'] = 42
            config['epochs'] = 2
            config['min_epochs'] = 1
            config['patience'] = max(3, int(config.get('patience', 3)))
            # Conservative starting settings; the feasibility gate chooses final microbatch.
            config['batch_size'] = 1
            config['gradient_accumulation'] = 4
            config['output_root'] = 'training_output/T04/t04_05_smoke'
            target = output / ('%s_%s_e5_seed42.yml' % (model, strategy.lower()))
            if target.exists():
                raise FileExistsError('Refusing to overwrite config: %s' % target)
            target.write_text(yaml.safe_dump(config, sort_keys=False))
            created.append(target)
    return created


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', default='.')
    p.add_argument('--output')
    a = p.parse_args()
    created = generate(a.repo, output=a.output)
    for item in created:
        print('CREATED', item)
    print('PASS: generated %d independent E5 configurations; no run reserved' % len(created))


if __name__ == '__main__':
    main()
