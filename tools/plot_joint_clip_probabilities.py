#!/usr/bin/env python3
"""Validation-only mean softmax probability by true parameter bin for ML-JET-Ext.

Input: a T04-11 aligned, sample-level validation prediction export; either
  .npz containing alpha_logits, q0_logits, alpha_targets, q0_targets;
  .csv containing alpha_logits, q0_logits, alpha_target, q0_target,
       where logit cells are JSON arrays.

May alternatively supply probabilities with alpha_probs / q0_probs columns or
NPZ arrays, but these are used as probabilities (never softmaxed again).

NEVER invoke test split here; there is no training, checkpoint loading, or inference.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ALPHA = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
Q0 = [1.0, 1.5, 2.0, 2.5, 3.0]


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Logits must be a finite [N, C] array')
    z = x - x.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def _probability_check(x, n, label):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != n or not np.isfinite(x).all():
        raise ValueError(f'{label} probabilities must be finite [N,{n}]')
    if (x < -1e-9).any() or (x > 1 + 1e-9).any() or not np.allclose(x.sum(axis=1), 1, atol=1e-4):
        raise ValueError(f'{label}: invalid probability rows')
    return x


def _labels(x, values, name):
    x = np.asarray(x)
    if x.dtype.kind in 'USO':
        try:
            as_float = x.astype(float)
            if np.all(np.isfinite(as_float)) and np.all(as_float == np.floor(as_float)) and np.all((as_float >= 0) & (as_float < len(values))):
                x = as_float.astype(np.int64)
        except ValueError:
            pass
    if x.ndim != 1:
        raise ValueError(f'{name} labels must be 1D')
    # Class indices are standard. Physical float labels are also accepted.
    if np.issubdtype(x.dtype, np.integer) and np.all((x >= 0) & (x < len(values))):
        return x.astype(np.int64)
    try:
        floats = x.astype(float)
        mapped = np.array([int(np.argmin(np.abs(np.array(values) - v))) for v in floats])
        if np.allclose(floats, np.array(values)[mapped], atol=1e-5):
            return mapped
    except (ValueError, TypeError):
        pass
    raise ValueError(f'{name} labels must be class indices or configured physical values')


def load_prediction_export(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == '.npz':
        with np.load(path, allow_pickle=False) as f:
            keys = set(f.files)
            def get(*candidates):
                for k in candidates:
                    if k in keys:
                        return f[k]
                raise KeyError(f'Expected one of {candidates}; available keys: {sorted(keys)}')
            ya = get('true_alpha', 'alpha_targets', 'alpha_target', 'alpha_labels', 'y_alpha')
            yq = get('true_q0', 'q0_targets', 'q0_target', 'q0_labels', 'y_q0')
            if 'alpha_logits' in keys and 'q0_logits' in keys:
                pa = _softmax(f['alpha_logits'])
                pq = _softmax(f['q0_logits'])
            elif 'alpha_probs' in keys and 'q0_probs' in keys:
                pa, pq = f['alpha_probs'], f['q0_probs']
            else:
                raise KeyError(f'Missing both-head logits/probs; NPZ keys: {sorted(keys)}')
            ids = get('sample_ids', 'sample_id', 'ids') if keys.intersection({'sample_ids','sample_id','ids'}) else None
    elif suffix == '.csv':
        with path.open(newline='') as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            raise ValueError('Empty CSV')
        cols = set(rows[0])
        def col(*keys):
            for key in keys:
                if key in cols:
                    return [row[key] for row in rows]
            raise KeyError(f'Expected one of {keys}; CSV columns: {sorted(cols)}')
        ya = col('alpha_target', 'alpha_targets', 'alpha_label', 'y_alpha')
        yq = col('q0_target', 'q0_targets', 'q0_label', 'y_q0')
        if 'alpha_logits' in cols and 'q0_logits' in cols:
            pa = _softmax([json.loads(x) for x in col('alpha_logits')])
            pq = _softmax([json.loads(x) for x in col('q0_logits')])
        elif 'alpha_probs' in cols and 'q0_probs' in cols:
            pa = np.asarray([json.loads(x) for x in col('alpha_probs')], dtype=float)
            pq = np.asarray([json.loads(x) for x in col('q0_probs')], dtype=float)
        else:
            raise KeyError(f'Missing both-head logits/probs; CSV columns: {sorted(cols)}')
        ids = col('sample_id', 'sample_ids', 'id') if cols.intersection({'sample_id','sample_ids','id'}) else None
    else:
        raise ValueError('Use a .npz or .csv per-sample prediction export')
    ya = _labels(ya, ALPHA, 'alpha')
    yq = _labels(yq, Q0, 'q0')
    pa = _probability_check(pa, 7, 'alpha')
    pq = _probability_check(pq, 5, 'q0')
    if not (len(ya) == len(yq) == len(pa) == len(pq)):
        raise ValueError('Alpha and Q0 predictions/labels are misaligned')
    if ids is not None and len(set(map(str, ids))) != len(ya):
        raise ValueError('Duplicate sample IDs')
    return ya, yq, pa, pq


def summarize(labels, probs, values):
    nclasses = len(values)
    count = np.bincount(labels, minlength=nclasses)
    if (count == 0).any():
        raise ValueError(f'Missing true bins: {np.where(count == 0)[0].tolist()}')
    means = np.stack([probs[labels == t].mean(axis=0) for t in range(nclasses)])
    # Average confidence assigned to the true bin (not accuracy or calibration).
    diagonal = np.diag(means)
    nearby = np.array([sum(means[t, p] for p in range(nclasses) if abs(t-p) == 1)
                       for t in range(nclasses)])
    far = np.array([sum(means[t, p] for p in range(nclasses) if abs(t-p) > 1)
                    for t in range(nclasses)])
    predictions = probs.argmax(axis=1)
    return {'count': count, 'means': means, 'diagonal': diagonal,
            'nearby': nearby, 'far': far, 'top1': float((predictions == labels).mean()),
            'brier': float(np.mean(np.sum((probs - np.eye(nclasses)[labels]) ** 2, axis=1)))}


def plot_heatmaps(a, q, out, *, seed_label=None):
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.45), constrained_layout=True)
    for ax, data, vals, label in zip(axes, (a, q), (ALPHA, Q0), (r'$\alpha_s$', r'$Q_0$')):
        m = data['means']
        im = ax.imshow(m, vmin=0, vmax=1, cmap='Blues', interpolation='nearest', aspect='equal')
        ax.set_xticks(range(len(vals)), [f'{v:g}' for v in vals], rotation=45, ha='right')
        ax.set_yticks(range(len(vals)), [f'{v:g}' for v in vals])
        ax.set_xlabel(f'Predicted {label} bin')
        ax.set_ylabel(f'True {label} bin')
        panel = "a" if len(vals) == 7 else "b"
        ax.set_title(f"({panel}) {label}")
        for t in range(len(vals)):
            for p in range(len(vals)):
                v = m[t,p]
                ax.text(p, t, f'{v:.2f}', ha='center', va='center', fontsize=7.3,
                        color='white' if v > .55 else 'black')
    fig.colorbar(im, ax=axes, shrink=.74, label='Mean predicted probability')
    fig.savefig(str(out) + '.pdf', bbox_inches='tight')
    fig.savefig(str(out) + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_facets(a, q, out):
    """JetBench-style separate probability bars for each true class."""
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(15.4, 6.4), constrained_layout=True)
    gs = GridSpec(2, 7, figure=fig, height_ratios=[1, 1])
    for row, (data, vals, symbol) in enumerate(((a, ALPHA, r'$\alpha_s$'),
                                                 (q, Q0, r'$Q_0$'))):
        for t in range(len(vals)):
            if row == 0:
                ax = fig.add_subplot(gs[row, t])
            else:
                # 5 panels across a 7-column grid, with a little extra width for labels
                left = (7 - len(vals)) / 2
                ax = fig.add_axes([0.08 + (t + left)*0.125, 0.085, 0.106, 0.31])
            colors = ['#477ba8' if p != t else '#e3a343' for p in range(len(vals))]
            bars = ax.bar(range(len(vals)), data['means'][t], color=colors, width=.73)
            ax.set_ylim(0, 1.05)
            ax.set_title(f'True {symbol}={vals[t]:g} (N={data["count"][t]})', fontsize=8.2)
            ax.set_xticks(range(len(vals)), [f'{v:g}' for v in vals], rotation=55, ha='right', fontsize=7)
            ax.tick_params(axis='y', labelsize=7)
            if t == 0:
                ax.set_ylabel('Mean probability', fontsize=9)
            ax.grid(axis='y', alpha=.14)
    fig.savefig(str(out) + '.pdf', bbox_inches='tight')
    fig.savefig(str(out) + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True, type=Path,
                        help='Aligned validation per-sample NPZ or CSV with both heads')
    parser.add_argument('--out-dir', default='results/t04_11/probability_profiles/validation', type=Path)
    parser.add_argument('--split', choices=('val',), default='val',
                        help='Validation-only by design; do not touch held-out test')
    parser.add_argument('--seed', type=int, choices=(42, 123, 2026),
                        help='Optional selected CLIP joint seed, saved as provenance')
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ya, yq, pa, pq = load_prediction_export(args.predictions)
    a, q = summarize(ya, pa, ALPHA), summarize(yq, pq, Q0)
    plot_heatmaps(a, q, args.out_dir / 'clip_joint_mean_probability_heatmaps')
    plot_facets(a, q, args.out_dir / 'clip_joint_mean_probability_bars')
    with (args.out_dir / 'clip_joint_mean_probabilities.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(('head', 'true_class', 'true_value', 'predicted_class',
                    'predicted_value', 'n_true', 'mean_probability'))
        for tag, data, vals in (('alpha',a,ALPHA), ('q0',q,Q0)):
            for i, true_value in enumerate(vals):
                for j, predicted_value in enumerate(vals):
                    w.writerow((tag, i, true_value, j, predicted_value,
                                int(data['count'][i]), float(data['means'][i,j])))
    summary = {
        'model': 'clip_joint', 'split':'val', 'seed': args.seed,
        'source': str(args.predictions.resolve()), 'n_samples': len(ya),
        'alpha': {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                  for k,v in a.items()},
        'q0': {k: (v.tolist() if isinstance(v, np.ndarray) else v)
               for k,v in q.items()},
        'caution': 'Mean per-class probabilities measure error structure, not calibration. '
                   'A reliability diagram/ECE is required to assess calibration.',
    }
    (args.out_dir/'clip_joint_probability_summary.json').write_text(json.dumps(summary, indent=2))
    print(f'Wrote validation probability profiles for N={len(ya)}')
    print(f'Alpha top1={a["top1"]:.4f}; Q0 top1={q["top1"]:.4f}')
    print(f'Outputs in {args.out_dir}')

if __name__ == '__main__':
    main()
