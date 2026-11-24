#!/usr/bin/env python3
"""Publication figures for the existing, pre-normalized ML-JET-Ext moment dataset.

No simulation, aggregation, or preprocessing is performed.
Usage:
  python plot_mljet_moment_figures.py --dataset-root /path/to/..._m1-5_normalized --out-dir figures
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
import numpy as np

ALPHAS = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
Q0S = (1.0, 1.5, 2.0, 2.5, 3.0)
MOMENT_LABELS = ('M1: Mean', 'M2: Variance', 'M3: Third central moment',
                 'M4: Fourth central moment', 'M5: Fifth central moment')


def config_name(alpha, q0):
    return f"c0005_alpha{alpha:.2f}_q_{q0:.1f}".replace('.', 'p')


def load_samples(root, sample_index):
    samples, paths = {}, {}
    for q in Q0S:
        for a in ALPHAS:
            cfg = config_name(a, q)
            candidates = sorted((root / cfg).glob('agg500_event_*.npy'))
            if len(candidates) <= sample_index:
                raise FileNotFoundError(f'{cfg}: expected sample index {sample_index}; found {len(candidates)}')
            path = candidates[sample_index]
            x = np.load(path, allow_pickle=False)
            if x.shape != (5, 32, 32) or not np.isfinite(x).all():
                raise ValueError(f'Invalid tensor {path} with shape {x.shape}')
            samples[(a, q)] = x
            paths[(a, q)] = path
    return samples, paths


def scales(samples):
    # Globally consistent scale for each moment across all configuration figures.
    # Odd moments use a symmetric scale, preserving meaningful zero and negative values.
    out = []
    for k in range(5):
        vals = np.array([samples[(a, q)][k] for q in Q0S for a in ALPHAS])
        if k in (2, 4):
            lim = max(float(np.max(np.abs(vals))), 1e-12)
            out.append((TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim), 'coolwarm'))
        else:
            lim = max(float(vals.max()), 1e-12)
            out.append((Normalize(vmin=0, vmax=lim), 'turbo'))
    return out


def save_formats(fig, stem, out):
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'{stem}.{ext}', bbox_inches='tight', dpi=300, facecolor='white')
    plt.close(fig)


def grid_figure(samples, k, scale, out):
    # Match reference: rows=Q0, columns=alpha_s, a single bottom colorbar.
    norm, cmap = scale
    fig, axes = plt.subplots(5, 7, figsize=(15.5, 12.0), sharex=True, sharey=True,
                             gridspec_kw={'wspace': 0.16, 'hspace': 0.15})
    for ri, q in enumerate(Q0S):
        for ci, a in enumerate(ALPHAS):
            ax = axes[ri, ci]
            # Original code transposed before imshow; horizontal phi and vertical eta.
            ax.imshow(samples[(a, q)][k].T, origin='lower', interpolation='nearest',
                      extent=(-0.5, 0.5, -0.5, 0.5), aspect='equal', norm=norm, cmap=cmap)
            if ri == 0:
                ax.set_title(rf'$\alpha_s={a:.2f}$', fontsize=13, pad=9)
            if ci == 0:
                ax.set_ylabel(rf'$Q_0={q:.1f}\,\mathrm{{GeV}}$', fontsize=12, labelpad=10)
            ax.set_xticks([-0.5, 0, 0.5] if ri == 4 else [])
            ax.set_yticks([-0.5, 0, 0.5] if ci == 0 else [])
            ax.tick_params(labelsize=8, length=2)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.955, bottom=0.115)
    fig.supxlabel(r'$\phi$ (relative azimuth)', x=0.55, y=0.065, fontsize=13)
    fig.supylabel(r'$\eta$ (relative pseudorapidity)', x=0.018, y=0.54, fontsize=13)
    cb_ax = fig.add_axes([0.26, 0.025, 0.55, 0.018])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cb_ax, orientation='horizontal')
    cb.set_label(rf'Normalized M{k+1} (fixed-reference log / signed-log)', fontsize=11)
    cb.ax.tick_params(labelsize=9)
    save_formats(fig, f'moment_M{k+1}_all_35_configs', out)


def strip_figure(samples, a, q, all_scales, out):
    fig, axes = plt.subplots(1, 5, figsize=(17.4, 4.5), gridspec_kw={'wspace': 0.33})
    for k, ax in enumerate(axes):
        norm, cmap = all_scales[k]
        im = ax.imshow(samples[(a, q)][k].T, origin='lower', interpolation='nearest',
                       extent=(-0.5, 0.5, -0.5, 0.5), aspect='equal', norm=norm, cmap=cmap)
        ax.set_title(MOMENT_LABELS[k], fontsize=12)
        ax.set_xlabel(r'$\phi$', fontsize=11)
        if k == 0:
            ax.set_ylabel(r'$\eta$', fontsize=11)
        else:
            ax.set_yticklabels([])
        ax.set_xticks([-0.5, 0, 0.5]); ax.set_yticks([-0.5, 0, 0.5])
        ax.tick_params(labelsize=8)
        cb = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.06, pad=0.19)
        cb.ax.tick_params(labelsize=8)
    fig.suptitle(rf'Same 500-jet aggregate: $\alpha_s={a:.2f}$, $Q_0={q:.1f}$ GeV',
                 y=0.99, fontsize=14)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.84, bottom=0.17)
    save_formats(fig, 'same_aggregate_M1_to_M5', out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, default=Path('paper_figures'))
    parser.add_argument('--sample-index', type=int, default=0,
                        help='Zero-based index among lexically sorted agg500_event_*.npy files')
    parser.add_argument('--example-alpha', type=float, choices=ALPHAS, default=0.20)
    parser.add_argument('--example-q0', type=float, choices=Q0S, default=1.0)
    args = parser.parse_args()
    if args.sample_index < 0:
        parser.error('--sample-index must be nonnegative')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    samples, paths = load_samples(args.dataset_root, args.sample_index)
    scale = scales(samples)
    for k in range(5):
        grid_figure(samples, k, scale[k], args.out_dir)
    strip_figure(samples, args.example_alpha, args.example_q0, scale, args.out_dir)
    with (args.out_dir / 'figure_sample_manifest.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['alpha_s', 'q0_GeV', 'sample_index', 'tensor_path'])
        for q in Q0S:
            for a in ALPHAS:
                w.writerow([a, q, args.sample_index, str(paths[(a, q)].resolve())])
    print(f'Created 6 figures in PDF and PNG plus sample manifest in {args.out_dir}')


if __name__ == '__main__':
    main()
