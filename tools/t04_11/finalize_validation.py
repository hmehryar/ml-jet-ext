"""T04-11 validation-only final audit and manuscript-ready summary.
Run: python -m tools.t04_11.finalize_validation
Requires existing results/t04_11/evaluation/val and predictions/val.
No test access, GPU usage or inference.
"""
from __future__ import annotations
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path('results/t04_11')
EVAL = ROOT / 'evaluation' / 'val'
CACHE = ROOT / 'predictions' / 'val'
DEST = ROOT / 'final' / 'validation_only'
SEEDS = (42, 123, 2026)
ROLES = ('clip_joint', 'siglip_joint', 'clip_alpha', 'siglip_alpha', 'clip_q0', 'siglip_q0')
SYSTEMS = (
    'CLIP joint', 'SigLIP joint', 'CLIP α + CLIP Q0',
    'CLIP α + SigLIP Q0', 'SigLIP α + CLIP Q0',
    'SigLIP α + SigLIP Q0',
)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read_rows(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))

def assert_close(x, y, msg, tol=1e-10):
    if not np.isclose(x, y, atol=tol, rtol=0):
        raise AssertionError(f'{msg}: {x} != {y}')

def make_table(headers, rows):
    out = ['| ' + ' | '.join(headers) + ' |',
           '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    out.extend('| ' + ' | '.join(map(str, row)) + ' |' for row in rows)
    return '\n'.join(out)

def main():
    manifest = json.loads((EVAL / 'evaluation_manifest.json').read_text())
    assert manifest['split'] == 'val'
    assert tuple(manifest['seeds']) == SEEDS
    assert manifest['samples_per_seed'] == 1400
    assert set(manifest['systems']) == set(SYSTEMS)
    split_sha = manifest['split_sha256']
    assert len(manifest['provenance']) == 18
    assert all(p['split_sha256'] == split_sha for p in manifest['provenance'])
    assert len({(p['role'], int(p['seed'])) for p in manifest['provenance']}) == 18

    # Independently verify cache bytes, precision, and sample/label alignment.
    ref = None
    for seed in SEEDS:
        for role in ROLES:
            path = CACHE / f'seed-{seed}' / f'{role}.npz'
            meta = json.loads(path.with_suffix('.json').read_text())
            assert meta['prediction_npz_sha256'] == digest(path)
            assert meta['split_sha256'] == split_sha
            assert meta['inference_precision'] == 'bf16'
            assert int(meta['seed']) == seed and meta['role'] == role
            with np.load(path, allow_pickle=False) as x:
                current = tuple(x[k] for k in ('sample_ids', 'true_alpha', 'true_q0'))
                assert len(current[0]) == 1400
                assert len(np.unique(current[0])) == 1400
                if ref is None:
                    ref = tuple(v.copy() for v in current)
                else:
                    assert all(np.array_equal(a, b) for a, b in zip(ref, current))
                assert all(np.isfinite(x[k]).all() for k in x.files if k.endswith('_logits'))

    metrics = read_rows(EVAL / 'per_seed_metrics.csv')
    summaries = read_rows(EVAL / 'summary_mean_sd.csv')
    predictions = read_rows(EVAL / 'per_sample_predictions.csv')
    assert len(metrics) == 18 and len(summaries) == 6
    assert len(predictions) == 6 * 3 * 1400
    by_sys_seed = {}
    sample_lookup = defaultdict(list)
    for r in predictions:
        key = (r['system'], int(r['seed']))
        assert key[0] in SYSTEMS and key[1] in SEEDS
        sample_lookup[key].append(r)
    for r in metrics:
        key = (r['system'], int(r['seed']))
        assert key not in by_sys_seed
        by_sys_seed[key] = r
        group = sample_lookup[key]
        assert len(group) == 1400
        ids = [x['sample_id'] for x in group]
        assert len(set(ids)) == 1400
        truth_a = np.array([int(x['true_alpha']) for x in group])
        truth_q = np.array([int(x['true_q0']) for x in group])
        pred_a = np.array([int(x['pred_alpha']) for x in group])
        pred_q = np.array([int(x['pred_q0']) for x in group])
        a = pred_a == truth_a
        q = pred_q == truth_q
        assert_close(a.mean(), float(r['alpha_accuracy']), f'{key} alpha')
        assert_close(q.mean(), float(r['q0_accuracy']), f'{key} q0')
        assert_close((a & q).mean(), float(r['joint_accuracy']), f'{key} joint')
        for metric, value in [
            ('joint_top1', (a & q).mean()),
            ('joint_within0', (a & q).mean()),
            ('alpha_within1', (np.abs(pred_a - truth_a) <= 1).mean()),
            ('q0_within1', (np.abs(pred_q - truth_q) <= 1).mean()),
            ('joint_within1', ((np.abs(pred_a - truth_a) <= 1) & (np.abs(pred_q - truth_q) <= 1)).mean()),
            ('alpha_mae_class_steps', np.abs(pred_a - truth_a).mean()),
            ('q0_mae_class_steps', np.abs(pred_q - truth_q).mean()),
        ]:
            assert_close(value, float(r[metric]), f'{key} {metric}')
        for axis, truth, pred, n in [('alpha', truth_a, pred_a, 7), ('q0', truth_q, pred_q, 5),
                                      ('joint', truth_a * 5 + truth_q, pred_a * 5 + pred_q, 35)]:
            assert np.all((truth >= 0) & (truth < n))
            assert np.all((pred >= 0) & (pred < n))
    assert set(by_sys_seed) == {(s, seed) for s in SYSTEMS for seed in SEEDS}
    for system in SYSTEMS:
        for axis, n in [('alpha', 7), ('q0', 5), ('joint', 35)]:
            name = system.replace(' ', '_').replace('α', 'alpha').replace('+', 'plus')
            matrix = np.load(EVAL / f'{name}_{axis}_confusion.npy', allow_pickle=False)
            assert matrix.shape == (n, n)
            assert int(matrix.sum()) == 4200
            observed = np.zeros((n,n), dtype=np.int64)
            for seed in SEEDS:
                for p in sample_lookup[(system, seed)]:
                    if axis == 'alpha':
                        i,j = int(p['true_alpha']),int(p['pred_alpha'])
                    elif axis == 'q0':
                        i,j = int(p['true_q0']),int(p['pred_q0'])
                    else:
                        i = int(p['true_alpha']) * 5 + int(p['true_q0'])
                        j = int(p['pred_alpha']) * 5 + int(p['pred_q0'])
                    observed[i,j] += 1
            assert np.array_equal(matrix, observed), f'{system} {axis} confusion mismatch'
    for row in summaries:
        system = row['system']
        for field in ('joint_accuracy', 'alpha_accuracy', 'q0_accuracy', 'joint_top3',
                      'joint_top5', 'joint_within1', 'alpha_mae_class_steps', 'q0_mae_class_steps'):
            vals = np.array([float(by_sys_seed[(system, seed)][field]) for seed in SEEDS])
            assert_close(vals.mean(), float(row[f'{field}_mean']), f'{system} {field} mean')
            assert_close(vals.std(ddof=1), float(row[f'{field}_sd']), f'{system} {field} sd')

    DEST.mkdir(parents=True, exist_ok=True)
    lines = [
        '# T04-11 — Validation-only final evaluation', '',
        '**Status:** complete for the predefined aggregate-disjoint validation split (n=1,400).',
        '**Seeds:** 42, 123, 2026 (three paired runs). **Inference:** BF16.',
        '**No held-out test evaluation was conducted for this report.**', '',
        '## Joint accuracy and error tolerance', '',
    ]
    summap = {r['system']: r for r in summaries}
    def pct(system, field):
        r = summap[system]
        return f"{100*float(r[field+'_mean']):.2f} ± {100*float(r[field+'_sd']):.2f}"
    lines.append(make_table(['System', 'Joint top-1 (%)', 'Joint top-3 (%)',
                             'Joint top-5 (%)', 'Joint within-1 (%)'], [
        [s, pct(s,'joint_accuracy'), pct(s,'joint_top3'),
         pct(s,'joint_top5'), pct(s,'joint_within1')] for s in SYSTEMS
    ]))
    lines.extend(['', '## Marginal accuracy and ordinal error', ''])
    lines.append(make_table(['System', 'Alpha accuracy (%)', 'Q0 accuracy (%)',
                             'Alpha MAE (steps)', 'Q0 MAE (steps)'], [
        [s, pct(s,'alpha_accuracy'), pct(s,'q0_accuracy'),
         f"{float(summap[s]['alpha_mae_class_steps_mean']):.4f} ± {float(summap[s]['alpha_mae_class_steps_sd']):.4f}",
         f"{float(summap[s]['q0_mae_class_steps_mean']):.4f} ± {float(summap[s]['q0_mae_class_steps_sd']):.4f}"]
        for s in SYSTEMS
    ]))
    lines.extend(['', '## Paired-seed differences against CLIP joint', '',
                  'Differences in **percentage points** for joint top-1 accuracy;'
                  ' mean ± sample SD across three paired seed-wise differences.', ''])
    delta_rows = []
    baseline = np.array([float(by_sys_seed[('CLIP joint', seed)]['joint_accuracy']) for seed in SEEDS])
    for system in SYSTEMS[1:]:
        values = np.array([float(by_sys_seed[(system, seed)]['joint_accuracy']) for seed in SEEDS])
        diff = 100 * (values - baseline)
        delta_rows.append([system, f'{diff.mean():+.2f} ± {diff.std(ddof=1):.2f}',
                           ', '.join(f'{x:+.2f}' for x in diff)])
    lines.append(make_table(['System', 'Mean paired delta (pp)', 'Seed 42, 123, 2026 deltas (pp)'], delta_rows))
    lines.extend(['', '## Scope and limitations', '',
                  '- Model and loss selections were made using validation results; this is a validation-only characterization, not a held-out generalization estimate.',
                  '- Dataset splits are aggregate-disjoint, **not fully source-jet-disjoint**; source overlap was disclosed by the training manifests.',
                  '- Across-seed SD (n=3) is descriptive; it should not be interpreted as a formal significance test.',
                  '- Joint top-k ranks all 35 class pairs by the sum of their alpha and Q0 head logits; within-k uses ordinal class-index distance.',
                  '- Confusion matrices pool three seeds (4,200 predictions) for visualization; repeated validation samples across seeds are not 4,200 independent observations.',
                  '', '## Reproducibility', '',
                  '- Evaluation summary: `results/t04_11/evaluation/val/summary_mean_sd.csv`',
                  '- Per-seed metrics: `results/t04_11/evaluation/val/per_seed_metrics.csv`',
                  '- Per-sample predictions: `results/t04_11/evaluation/val/per_sample_predictions.csv`',
                  '- Provenance: `results/t04_11/evaluation/val/evaluation_manifest.json`',
                  '- Confusion matrices and PDF figures: `results/t04_11/evaluation/val/`', ''])
    (DEST / 'validation_results.md').write_text('\n'.join(lines) + '\n')
    files = [EVAL / f for f in ('per_seed_metrics.csv', 'summary_mean_sd.csv',
                                 'per_sample_predictions.csv', 'evaluation_manifest.json')]
    files.extend(sorted(EVAL.glob('*_confusion.npy')))
    files.extend(sorted((EVAL / 'figures').glob('*.pdf')))
    files.append(DEST / 'validation_results.md')
    records = [{'path': str(p), 'sha256': digest(p), 'size_bytes': p.stat().st_size} for p in files]
    audit = {
        'schema': 't04_11_validation_final_audit_v1',
        'scope': 'validation_only',
        'test_evaluated': False,
        'split_sha256': split_sha,
        'seeds': list(SEEDS),
        'systems': list(SYSTEMS),
        'n_samples_per_seed': 1400,
        'n_cache_files_verified': 18,
        'n_prediction_rows_verified': 25200,
        'n_confusion_matrices_verified': 18,
        'files': records,
    }
    (DEST / 'final_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print('PASS: 18 cache checksums, 25,200 prediction records, 18 confusion matrices')
    print('PASS: top-1, within-1, MAE, mean/SD, sample alignment and split SHA-256')
    print('REPORT:', DEST / 'validation_results.md')
    print('AUDIT: ', DEST / 'final_audit.json')
    print('No test data accessed.')

if __name__ == '__main__':
    main()
