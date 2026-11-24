"""T04-11: cache-only evaluation of joint models and expert compositions.

Usage: python -m tools.t04_11.evaluate_cached --split val
No GPU or checkpoint loading required.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SEEDS = (42, 123, 2026)
ROLES = ('clip_joint', 'siglip_joint', 'clip_alpha', 'siglip_alpha', 'clip_q0', 'siglip_q0')
SYSTEMS = {
    'CLIP joint': ('clip_joint', 'clip_joint'),
    'SigLIP joint': ('siglip_joint', 'siglip_joint'),
    'CLIP α + CLIP Q0': ('clip_alpha', 'clip_q0'),
    'CLIP α + SigLIP Q0': ('clip_alpha', 'siglip_q0'),
    'SigLIP α + CLIP Q0': ('siglip_alpha', 'clip_q0'),
    'SigLIP α + SigLIP Q0': ('siglip_alpha', 'siglip_q0'),
}
ALPHA_VALUES = np.array([.20, .25, .30, .35, .40, .45, .50])
Q0_VALUES = np.array([1.0, 1.5, 2.0, 2.5, 3.0])


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path, records, columns):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)


def confusion(truth, predicted, n):
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (truth, predicted), 1)
    assert int(cm.sum()) == len(truth)
    return cm


def rank_of_truth(logits, truth):
    # Stable sort makes exact ties deterministic; top-k for classes is well-defined.
    order = np.argsort(-logits, axis=1, kind='stable')
    return np.argmax(order == truth[:, None], axis=1) + 1


def compute_metrics(a, q, y_a, y_q):
    pa, pq = a.argmax(1), q.argmax(1)
    n = len(y_a)
    arank, qrank = rank_of_truth(a, y_a), rank_of_truth(q, y_q)
    # Independent-head joint ranking: logit(alpha)+logit(q0), equivalent
    # to ranking products of independent softmax probabilities.
    joint_scores = (a[:, :, None] + q[:, None, :]).reshape(n, 35)
    y_joint = y_a * 5 + y_q
    joint_rank = rank_of_truth(joint_scores, y_joint)
    results = {
        'alpha_accuracy': float(np.mean(pa == y_a)),
        'q0_accuracy': float(np.mean(pq == y_q)),
        'joint_accuracy': float(np.mean((pa == y_a) & (pq == y_q))),
        'alpha_mae_class_steps': float(np.mean(np.abs(pa - y_a))),
        'q0_mae_class_steps': float(np.mean(np.abs(pq - y_q))),
        'alpha_mae_physical': float(np.mean(np.abs(ALPHA_VALUES[pa] - ALPHA_VALUES[y_a]))),
        'q0_mae_physical': float(np.mean(np.abs(Q0_VALUES[pq] - Q0_VALUES[y_q]))),
    }
    for k in (1, 2, 3, 5):
        results[f'alpha_top{k}'] = float(np.mean(arank <= k))
        results[f'q0_top{k}'] = float(np.mean(qrank <= k))
        results[f'joint_top{k}'] = float(np.mean(joint_rank <= k))
    for k in (0, 1, 2):
        aa = np.abs(pa - y_a) <= k
        qq = np.abs(pq - y_q) <= k
        results[f'alpha_within{k}'] = float(np.mean(aa))
        results[f'q0_within{k}'] = float(np.mean(qq))
        results[f'joint_within{k}'] = float(np.mean(aa & qq))
    assert abs(results['joint_top1'] - results['joint_accuracy']) < 1e-12
    assert abs(results['joint_within0'] - results['joint_accuracy']) < 1e-12
    return results, (pa, pq, y_joint, pa * 5 + pq)


def plot_cm(cm, labels, title, path, normalize=True):
    cm_show = cm.astype(float)
    if normalize:
        den = cm_show.sum(axis=1, keepdims=True)
        cm_show = np.divide(cm_show, den, out=np.zeros_like(cm_show), where=den != 0)
    n = len(labels)
    fig_size = (11, 9.5) if n == 35 else (6.0, 5.2)
    fig, ax = plt.subplots(figsize=fig_size)
    im = ax.imshow(cm_show, cmap='Blues', vmin=0, vmax=1 if normalize else None)
    ax.set(title=title, xlabel='Predicted', ylabel='True')
    ax.set_xticks(np.arange(n), labels, rotation=90 if n == 35 else 0)
    ax.set_yticks(np.arange(n), labels)
    ax.tick_params(labelsize=5 if n == 35 else 10)
    if n <= 7:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f'{cm_show[i,j]:.2f}' if normalize else str(cm[i,j]),
                        ha='center', va='center', fontsize=8,
                        color='white' if cm_show[i, j] > 0.5 else 'black')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def load_cache(root, seed, role, split):
    path = root / f'seed-{seed}' / f'{role}.npz'
    meta_path = path.with_suffix('.json')
    metadata = json.loads(meta_path.read_text())
    assert metadata['prediction_npz_sha256'] == sha256(path), path
    assert metadata['seed'] == seed and metadata['role'] == role
    assert metadata['split'] == split
    assert metadata['inference_precision'] == 'bf16'
    with np.load(path, allow_pickle=False) as raw:
        arrays = {key: raw[key] for key in raw.files}
    ids = arrays['sample_ids']
    n = len(ids)
    assert len(set(ids.tolist())) == n
    assert metadata['sample_count'] == n
    assert np.all((arrays['true_alpha'] >= 0) & (arrays['true_alpha'] < 7))
    assert np.all((arrays['true_q0'] >= 0) & (arrays['true_q0'] < 5))
    for key in ('alpha_logits', 'q0_logits'):
        if key in arrays:
            nclass = 7 if key == 'alpha_logits' else 5
            assert arrays[key].shape == (n, nclass)
            assert np.isfinite(arrays[key]).all()
    return arrays, metadata


def evaluate(split, cache_parent, output_parent):
    root = Path(cache_parent) / split
    dest = Path(output_parent) / split
    dest.mkdir(parents=True, exist_ok=True)
    metrics_rows = []
    sample_rows = []
    provenance = []
    reference = None
    hashes = set()
    pooled = defaultdict(lambda: {'alpha': np.zeros((7, 7), dtype=np.int64),
                                  'q0': np.zeros((5, 5), dtype=np.int64),
                                  'joint': np.zeros((35, 35), dtype=np.int64)})
    for seed in SEEDS:
        caches = {}
        for role in ROLES:
            arrays, meta = load_cache(root, seed, role, split)
            triple = (arrays['sample_ids'], arrays['true_alpha'], arrays['true_q0'])
            if reference is None:
                reference = tuple(x.copy() for x in triple)
            else:
                assert all(np.array_equal(a, b) for a, b in zip(reference, triple)), (seed, role)
            hashes.add(meta['split_sha256'])
            provenance.append({
                'seed': seed, 'role': role, 'run_id': meta['run_id'],
                'moments': meta['moments'], 'training_strategy': meta['training_strategy'],
                'inference_precision': meta['inference_precision'],
                'checkpoint_sha256': meta['checkpoint_sha256'],
                'config_sha256': meta['config_sha256'],
                'split_sha256': meta['split_sha256'],
                'prediction_sha256': meta['prediction_npz_sha256'],
            })
            caches[role] = arrays
        ids, y_a, y_q = reference
        for name, (a_role, q_role) in SYSTEMS.items():
            a = caches[a_role]['alpha_logits']
            q = caches[q_role]['q0_logits']
            results, (pa, pq, y_joint, p_joint) = compute_metrics(a, q, y_a, y_q)
            metrics_rows.append({'system': name, 'seed': seed, 'n': len(ids), **results})
            pooled[name]['alpha'] += confusion(y_a, pa, 7)
            pooled[name]['q0'] += confusion(y_q, pq, 5)
            pooled[name]['joint'] += confusion(y_joint, p_joint, 35)
            for i, sample_id in enumerate(ids):
                sample_rows.append({
                    'system': name, 'seed': seed, 'sample_id': str(sample_id),
                    'true_alpha': int(y_a[i]), 'true_q0': int(y_q[i]),
                    'pred_alpha': int(pa[i]), 'pred_q0': int(pq[i]),
                    'alpha_correct': int(pa[i] == y_a[i]),
                    'q0_correct': int(pq[i] == y_q[i]),
                    'joint_correct': int(pa[i] == y_a[i] and pq[i] == y_q[i]),
                })
    assert len(hashes) == 1, 'Mixed split hashes'
    assert len(provenance) == 18 and len(metrics_rows) == 18
    fieldnames = list(metrics_rows[0])
    write_csv(dest / 'per_seed_metrics.csv', metrics_rows, fieldnames)
    summary = []
    for name in SYSTEMS:
        subset = [r for r in metrics_rows if r['system'] == name]
        row = {'system': name, 'seeds': '42;123;2026', 'n_per_seed': subset[0]['n']}
        for key in fieldnames:
            if key in ('system', 'seed', 'n'):
                continue
            values = np.array([r[key] for r in subset], dtype=float)
            row[f'{key}_mean'] = float(values.mean())
            row[f'{key}_sd'] = float(values.std(ddof=1))
        summary.append(row)
    write_csv(dest / 'summary_mean_sd.csv', summary, list(summary[0]))
    write_csv(dest / 'per_sample_predictions.csv', sample_rows, list(sample_rows[0]))
    fig_dir = dest / 'figures'
    fig_dir.mkdir(exist_ok=True)
    a_labels = [f'{x:.2f}' for x in ALPHA_VALUES]
    q_labels = [f'{x:.1f}' for x in Q0_VALUES]
    j_labels = [f'{a:.2f}/{q:.1f}' for a in ALPHA_VALUES for q in Q0_VALUES]
    for name, matrices in pooled.items():
        safe_name = (name.replace(' ', '_').replace('α', 'alpha').replace('+', 'plus'))
        for head, labels in [('alpha', a_labels), ('q0', q_labels), ('joint', j_labels)]:
            cm = matrices[head]
            np.save(dest / f'{safe_name}_{head}_confusion.npy', cm)
            plot_cm(cm, labels, f'{name} — {head} (3 seeds, row-normalized)',
                    fig_dir / f'{safe_name}_{head}_confusion.pdf')
    report = {
        'schema': 't04_11_evaluation_v1', 'split': split,
        'split_sha256': next(iter(hashes)), 'seeds': list(SEEDS),
        'samples_per_seed': len(reference[0]),
        'systems': {name: {'alpha_source': a, 'q0_source': q} for name, (a,q) in SYSTEMS.items()},
        'class_values': {'alpha': ALPHA_VALUES.tolist(), 'q0': Q0_VALUES.tolist()},
        'joint_topk_definition': 'rank of true (alpha, q0) pair by sum of independent-head logits',
        'within_k_definition': 'absolute class-index error <= k (joint requires both heads)',
        'confusion_definition': 'pooled counts over three seeds; figures row-normalized',
        'seed_statistics': 'mean and sample SD (ddof=1) across three paired seeds',
        'source_jet_overlap_disclosed': True,
        'provenance': provenance,
    }
    (dest / 'evaluation_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS: 18 aligned caches, 6 systems × 3 paired seeds')
    print('Results:', dest)
    print('\nJoint validation accuracy (%), mean ± sample SD:')
    for row in summary:
        print(f"  {row['system']:<26} {100*row['joint_accuracy_mean']:.2f} ± {100*row['joint_accuracy_sd']:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--cache-root', default='results/t04_11/predictions')
    parser.add_argument('--output-root', default='results/t04_11/evaluation')
    args = parser.parse_args()
    evaluate(args.split, args.cache_root, args.output_root)


if __name__ == '__main__':
    main()
