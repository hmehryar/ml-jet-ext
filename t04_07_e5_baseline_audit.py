#!/usr/bin/env python3
"""Read-only analysis of completed T04-06 v2 E5 validation histories on Ceph.

No model loading, job submission, config mutation, or test-set evaluation.
Outputs are new CSV/JSON files in a user-specified destination.
"""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

MODELS = ('clip', 'siglip')
STRATEGIES = ('S1', 'S2', 'S3', 'S4')
SEEDS = (42, 123, 2026)
COLUMNS = ('val_joint_accuracy', 'val_alpha_accuracy', 'val_q0_accuracy',
           'val_alpha_macro_f1', 'val_q0_macro_f1', 'val_alpha_mae',
           'val_q0_mae', 'val_alpha_rmse', 'val_q0_rmse')
CEPH = Path('/rs/rs_grp_mljet/ml-jet-ext/training_output/T04/t04_06_v2_production')


def number(row, key):
    value = row.get(key, '')
    try:
        x = float(value)
    except (ValueError, TypeError):
        return None
    return x if math.isfinite(x) else None


def get_run(root, model, strategy, seed):
    ident = f'{model}_{strategy.lower()}_e5_seed{seed}'
    run_id = f't04-06-{ident.replace("_", "-")}-v2'
    run = root / model / strategy / 'E5' / f'seed-{seed}' / run_id
    return ident, run


def read_run(root, model, strategy, seed, *, require_complete=True):
    ident, run = get_run(root, model, strategy, seed)
    history = run / 'moment_epoch_metrics.csv'
    if not history.is_file():
        return {'id': ident, 'model': model, 'strategy': strategy, 'seed': seed,
                'status': 'missing_csv', 'run': str(run)}
    with history.open(newline='') as f:
        reader = csv.DictReader(f)
        missing = {'epoch', *COLUMNS} - set(reader.fieldnames or ())
        if missing:
            return {'id': ident, 'model': model, 'strategy': strategy, 'seed': seed,
                    'status': 'missing_columns:' + ','.join(sorted(missing)), 'run': str(run)}
        rows = list(reader)
    config_path = run / 'config.resolved.json'
    if not config_path.exists():
        return {'id': ident, 'model': model, 'strategy': strategy, 'seed': seed,
                'status': 'missing_config', 'run': str(run)}
    config = json.loads(config_path.read_text())
    expected_epochs = config.get('epochs')
    if (config.get('model') != model or config.get('strategy') != strategy
            or config.get('seed') != seed or config.get('moments') != [1,2,3,4,5]
            or config.get('moment_loss', {}).get('type') != 'cross_entropy'
            or config.get('training_protocol') != 'fixed_budget_v2'):
        return {'id': ident, 'model': model, 'strategy': strategy, 'seed': seed,
                'status': 'config_mismatch', 'run': str(run)}
    try:
        epochs = [int(x['epoch']) for x in rows]
    except (ValueError, TypeError):
        epochs = []
    if epochs != list(range(1, len(rows) + 1)):
        return {'id': ident, 'model': model, 'strategy': strategy, 'seed': seed,
                'status': 'invalid_epoch_history', 'run': str(run)}
    if require_complete and (not isinstance(expected_epochs, int) or len(rows) != expected_epochs):
        return {'id': ident, 'model': model, 'strategy': strategy, 'seed': seed,
                'status': 'incomplete_epoch_history', 'run': str(run),
                'epochs_seen': len(rows), 'epochs_expected': expected_epochs}
    scored = [(i, number(row, 'val_joint_accuracy')) for i, row in enumerate(rows)]
    scored = [(i, v) for i, v in scored if v is not None]
    if not scored:
        return {'id': ident, 'model': model, 'strategy': strategy, 'seed': seed,
                'status': 'no_finite_joint_accuracy', 'run': str(run)}
    # Match T04 training: strictly greater improvement => earliest epoch on ties.
    i, _ = max(scored, key=lambda z: z[1])
    best = rows[i]
    out = {'id': ident, 'model': model, 'strategy': strategy, 'seed': seed,
           'status': 'ok', 'run': str(run), 'selected_epoch': int(best['epoch']),
           'epochs_seen': len(rows), 'epochs_expected': expected_epochs,
           'selection': 'earliest maximal val_joint_accuracy'}
    for col in COLUMNS:
        out[col] = number(best, col)
    return out


def summarize(records):
    summary = []
    for model in MODELS:
        for strategy in STRATEGIES:
            group = [r for r in records if r['model'] == model and r['strategy'] == strategy]
            ok = [r for r in group if r['status'] == 'ok']
            result = {'model': model, 'strategy': strategy, 'n_seeds': len(ok),
                      'complete': len(ok) == len(SEEDS)}
            for col in COLUMNS:
                vals = [r[col] for r in ok if r.get(col) is not None]
                result[col + '_mean'] = statistics.mean(vals) if len(vals) == len(SEEDS) else None
                result[col + '_std'] = statistics.stdev(vals) if len(vals) == len(SEEDS) else None
            summary.append(result)
    return summary


def write_csv(path, records):
    fields = list(dict.fromkeys(k for r in records for k in r))
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ceph-root', type=Path, default=CEPH)
    parser.add_argument('--output', type=Path, default=Path.cwd() / 't04_07_e5_audit')
    parser.add_argument('--allow-incomplete', action='store_true', help='Inspect partial histories but do not use for final strategy selection')
    args = parser.parse_args()
    root = args.ceph_root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f'Ceph baseline root missing: {root}')
    records = [read_run(root, m, s, seed, require_complete=not args.allow_incomplete)
               for m in MODELS for s in STRATEGIES for seed in SEEDS]
    summary = summarize(records)
    winners = {}
    for model in MODELS:
        choices = [r for r in summary if r['model'] == model and r['complete']
                   and r['val_joint_accuracy_mean'] is not None]
        if len(choices) == 4 and not args.allow_incomplete:
            # Predeclared rank: joint accuracy, then average sum of physical MAEs,
            # then order S1..S4. Physical values reported in different units;
            # secondary criterion only breaks ties and is not a cross-head average.
            winners[model] = sorted(choices, key=lambda x: (
                -x['val_joint_accuracy_mean'],
                (x['val_alpha_mae_mean'] if x['val_alpha_mae_mean'] is not None else math.inf),
                (x['val_q0_mae_mean'] if x['val_q0_mae_mean'] is not None else math.inf),
                STRATEGIES.index(x['strategy'])))[0]['strategy']
        else:
            winners[model] = None
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / 'e5_seed_metrics.csv', records)
    write_csv(args.output / 'e5_strategy_summary.csv', summary)
    report = {'protocol': 'read_only_T04-06_v2_CE_E5_validation',
              'ceph_root': str(root), 'seeds': list(SEEDS),
              'candidate_count': 24, 'valid_count': sum(r['status']=='ok' for r in records),
              'winners': winners, 'warnings': [r for r in records if r['status']!='ok'],
              'selection_rule': 'per run earliest maximum val_joint_accuracy; per strategy mean across three seeds; tie broken by alpha MAE then Q0 MAE',
              'test_metrics_used': False, 'note': 'Aggregate-disjoint does not establish source-disjointness.'}
    with (args.output / 'audit.json').open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print(json.dumps({'valid': report['valid_count'], 'winners': winners,
                      'output': str(args.output), 'warnings': len(report['warnings'])}, indent=2))
    if report['valid_count'] != 24:
        raise SystemExit('Incomplete E5 data; inspect e5_seed_metrics.csv before selecting strategies')

if __name__ == '__main__':
    main()
