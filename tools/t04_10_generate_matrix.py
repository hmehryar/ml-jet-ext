"""Generate 12 CLIP/SigLIP T04-10 configs from user-selected validated CE baselines.

The six ViT pilot entries are explicitly blocked pending the ViT parity task.
Never selects baselines from test metrics or submits jobs.
"""
import argparse
import json
from pathlib import Path
import yaml

SEEDS = (42, 123, 2026)
TASKS = ('alpha_only', 'q0_only')


def generate(clip_config, siglip_config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    plan = {'issue':30, 'ready_for_preflight':12, 'blocked_vit':6,
            'selection_status':'USER_VALIDATED_BASELINE_REQUIRED', 'entries':[]}
    for architecture, baseline in (('clip', clip_config), ('siglip', siglip_config)):
        source = Path(baseline).resolve()
        cfg = yaml.safe_load(source.read_text())
        if cfg.get('model') != architecture or cfg.get('moment_loss') != {'type':'cross_entropy'}:
            raise ValueError(f'{source}: expected {architecture} CE dual-head source')
        if cfg.get('training_protocol') != 'fixed_budget_v2' or cfg.get('epochs') != 80:
            raise ValueError(f'{source}: expected 80-epoch fixed-budget source')
        if cfg.get('strategy') not in ('S1','S2','S3','S4'):
            raise ValueError(f'{source}: invalid strategy')
        for task in TASKS:
            for seed in SEEDS:
                translated = {k:cfg[k] for k in (
                    'model', 'checkpoint', 'checkpoint_revision', 'strategy', 'moments',
                    'precision', 'optimizer', 'learning_rates', 'batch_size',
                    'num_workers', 'dataset_root', 'source_audit_report',
                    'split_protocol', 'gpu_profile', 'epochs', 'training_protocol')}
                translated['task'] = task
                translated['seed'] = seed
                translated['output_root'] = 'training_output/T04/t04_10_single_head'
                translated['scheduler'] = {k:cfg['scheduler'][k] for k in
                    ('type','factor','patience','min_lr')}
                ident = f't04_10_{architecture}_{task}_seed{seed}'
                path = output / f'{ident}.yml'
                path.write_text(yaml.safe_dump(translated, sort_keys=False))
                plan['entries'].append({'id':ident, 'config':str(path),
                    'status':'NOT_PREPARED', 'source_baseline':str(source)})
    for task in TASKS:
        for seed in SEEDS:
            plan['entries'].append({'id':f't04_10_vit_{task}_seed{seed}',
                'status':'BLOCKED_VIT_PARITY', 'source_baseline':None})
    (output / 'pilot_plan.json').write_text(json.dumps(plan, indent=2))
    return plan


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--clip-baseline', required=True)
    p.add_argument('--siglip-baseline', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    result = generate(args.clip_baseline, args.siglip_baseline, args.output)
    print(f"T04-10 matrix: {len(result['entries'])} entries; 12 candidate CLIP/SigLIP, 6 ViT blocked")
    print('No training scheduled. Baseline selections must be validation-audited before preparation.')

if __name__ == '__main__':
    main()
