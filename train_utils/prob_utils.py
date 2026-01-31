# train_utils/prob_utils.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, Tuple, Union

import numpy as np
import torch
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
# --- Global plot style ---
mpl.rcParams.update({
    "font.size": 15,           # global font size
    "axes.titlesize": 15,      # title font size
    "axes.labelsize": 15,      # x and y labels
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.titlesize": 16,
    "image.cmap": "Oranges",   # set global colormap
    "axes.prop_cycle": plt.cycler(color=["orange"]),  # <- all bars/lines orange
    # --- default edge color for patches (bars, hist bars, etc.) ---
    "patch.edgecolor": "black",

    # layout / suptitle spacing
    "figure.autolayout": False,   # keep manual control
    "figure.subplot.top": 0.88    # controls suptitle y position (0–1)
})
# when calling bar/hist, force alpha separately
default_alpha = 0.7

ModelOut = Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]]
@torch.no_grad()
def collect_head_probs(loader, model, device,
                        *,
                        alpha_key: str = "alpha_output",
                        q0_key: str = "q0_output",
                        energy_key: str | None = None,  # not required for this function
                    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Collect softmax probabilities for α_s and Q₀ heads across a loader.

    Returns:
        (alpha_probs, alpha_true), (q0_probs, q0_true)
        where alpha_probs.shape = [N, 3], q0_probs.shape = [N, 4]
    """
    model.eval()
    alpha_probs, q0_probs = [], []
    alpha_true,  q0_true  = [], []
    
    for x, labels in loader:
        x = x.to(device, non_blocking=True).float()
        for key in labels:
            labels[key] = labels[key].to(device, non_blocking=True).long()

        gt_alpha = labels[alpha_key].squeeze()
        gt_q0 = labels[q0_key].squeeze()

        # forward: expect model to return tuple/dict with heads
        out = model(x)
        # Handle either tuple or dict style outputs
        if isinstance(out, (list, tuple)):
            energy_out, alpha_out, q0_out = out
        elif isinstance(out, dict):
            alpha_out, q0_out = out[alpha_key], out[q0_key]
        else:
            raise ValueError("Unexpected model output structure")

        alpha_p = torch.softmax(alpha_out, dim=1)
        q0_p    = torch.softmax(q0_out,    dim=1)
        # alpha_p = alpha_out
        # q0_p    = q0_out

        alpha_probs.append(alpha_p.cpu().numpy())
        q0_probs.append(q0_p.cpu().numpy())
        alpha_true.append(gt_alpha.cpu().numpy())
        q0_true.append(gt_q0.cpu().numpy())

    alpha_probs = np.concatenate(alpha_probs, axis=0)  # [N, 3]
    q0_probs    = np.concatenate(q0_probs,    axis=0)  # [N, 4]
    alpha_true  = np.concatenate(alpha_true,  axis=0)  # [N]
    q0_true     = np.concatenate(q0_true,     axis=0)  # [N]
    return (alpha_probs, alpha_true), (q0_probs, q0_true)

def save_probs_csv(
    path_prefix: Union[str, Path],
    probs: np.ndarray,
    y_true: np.ndarray,
    class_names: Iterable[str],
) -> Path:
    """
    Save probabilities + y_true to CSV for later analysis.

    Produces: <path_prefix>_probs.csv
    """
    import pandas as pd
    path_prefix = Path(path_prefix)
    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(probs, columns=[f"p_{c}" for c in class_names])
    df["y_true"] = y_true
    csv_path = path_prefix.with_suffix("").as_posix() + "_probs.csv"
    df.to_csv(csv_path, index=False)
    return Path(csv_path)

# ---------------------------
# α_s helpers (probabilities)
# ---------------------------

def plot_alpha_histograms(
                        y_true_alpha,
                        y_pred_alpha,
                        class_names=[r"$\alpha_s=0.2$", r"$\alpha_s=0.3$", r"$\alpha_s=0.4$"],
                        figsize=(17, 5),
                        suptitle="Predicted αₛ distribution per true αₛ bin",
                        save_path=None,
                        show=False,
                      ):
    """
    Plot normalized histograms of predicted alpha_s values for each true alpha_s class.
    Bars are normalized (sum=1) but annotated with raw count and probability.
    Args:
        y_true_alpha (array-like, shape [N]): ground-truth αₛ class indices {0,1,2}
        y_pred_alpha (array-like, shape [N]): predicted αₛ class indices {0,1,2}
        class_names (tuple[str]): display names for classes in index order
        figsize (tuple): figure size
        suptitle (str): figure title
        save_path (str|None): if set, saves PNG to this path (creates dirs)
        show (bool): if True, plt.show() (useful in notebooks)
    Returns:
        fig (matplotlib.figure.Figure): the created figure (also saved if save_path)
    """

    y_true_alpha = np.array(y_true_alpha)
    y_pred_alpha = np.array(y_pred_alpha)
    num_classes = len(class_names)

    fig, axes = plt.subplots(1, num_classes, figsize=figsize, sharey=True,constrained_layout=True)

    for i, ax in enumerate(axes):
        # Select predictions where ground truth == i
        mask = (y_true_alpha == i)
        preds = y_pred_alpha[mask]

        # Count raw occurrences
        counts = np.bincount(preds, minlength=num_classes)
        total = counts.sum()

        # Normalize counts to probabilities
        probs = counts / total if total > 0 else np.zeros_like(counts)

        # Plot normalized histogram
        ax.bar(range(num_classes), probs, tick_label=class_names,
               alpha=default_alpha)

        # Add annotations with raw counts + probability
        for j, (c, p) in enumerate(zip(counts, probs)):
            ax.text(j, p + 0.02, f"{c} ({p:.2f})",
                    ha='center', va='bottom', clip_on=False)

        ax.grid(axis="y", linestyle="--", alpha=0.3)

        ax.set_title(f"True {class_names[i]} (N={total})")
        ax.set_ylabel("Normalized Frequency")

    fig.suptitle(suptitle)

    if save_path is not None:
        base = os.path.splitext(save_path)[0]  # allow ".png" or path without ext
        os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
        fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(base + ".pdf", bbox_inches="tight")
    if show:
        plt.tight_layout()
        plt.show()

    return fig
def plot_alpha_avgprob_histograms(
    y_true_alpha,
    y_alpha_proba,                        # shape (N, C), softmax probs per sample
    class_names=(r"$\alpha_s=0.2$", r"$\alpha_s=0.3$", r"$\alpha_s=0.4$"),
    figsize=(17, 6),
    suptitle=r"Average predicted $\alpha_s$ probability per true $\alpha_s$",
    save_path=None,
    show=False,
):
    """
    For each TRUE α_s class t, compute the mean predicted probability vector over all its samples:
        avg_probs[t] = mean( y_alpha_proba[ y_true_alpha == t ] , axis=0 )
    Then plot 3 bar charts (one per true α_s), bars = average probs for predicted classes.
    """
    y_true_alpha = np.asarray(y_true_alpha, dtype=int)
    y_alpha_proba = np.asarray(y_alpha_proba, dtype=float)
    C = len(class_names)

    # compute avg probs per true class
    avg_probs = np.zeros((C, C), dtype=float)
    Ns = np.zeros(C, dtype=int)
    for t in range(C):
        mask = (y_true_alpha == t)
        Ns[t] = int(mask.sum())
        if Ns[t] > 0:
            avg_probs[t] = y_alpha_proba[mask].mean(axis=0)

    # figure
    fig, axes = plt.subplots(1, C, figsize=figsize, sharey=True, constrained_layout=True)
    if C == 1:
        axes = [axes]
    x = np.arange(C)

    # global headroom to avoid label clipping
    ymax = float(np.max(avg_probs)) if avg_probs.size else 1.0
    ylim_top = max(0.05, min(1.05, ymax + 0.12))

    for t, ax in enumerate(axes):
        bars = ax.bar(x, avg_probs[t], alpha=default_alpha, tick_label=class_names)
        ax.set_ylim(0, ylim_top)
        ax.set_title(f"True {class_names[t]} (N={Ns[t]})")
        if t == 0:
            ax.set_ylabel("Average probability")

        # annotate value on each bar
        for j, b in enumerate(bars):
            h = b.get_height()
            ax.text(b.get_x()+b.get_width()/2, h + 0.02, f"{h:.3f}",
                    ha="center", va="bottom", clip_on=False)

        ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle(suptitle)

    if save_path:
        base = os.path.splitext(save_path)[0]  # allow ".png" or path without ext
        os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
        fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(base + ".pdf", bbox_inches="tight")

    if show:
        plt.show()
    return fig


# Plot Q0 average probability histograms
def plot_q0_avgprob_histograms(
    y_true_q0,
    y_q0_proba,                        # shape (N, 4), softmax probs per sample
    class_names=(r"$Q_0=1.0$", r"$Q_0=1.5$", r"$Q_0=2.0$", r"$Q_0=2.5$"),
    figsize=(21, 6),
    suptitle=r"Average predicted $Q_0$ probability per true $Q_0$",
    save_path=None,
    show=False,
):
    """
    For each TRUE Q0 class t, compute the mean predicted probability vector over all its samples:
        avg_probs[t] = mean( y_q0_proba[ y_true_q0 == t ] , axis=0 )
    Then plot 4 bar charts (one per true Q0), bars = average probs for predicted classes.
    """
    y_true_q0 = np.asarray(y_true_q0, dtype=int)
    y_q0_proba = np.asarray(y_q0_proba, dtype=float)
    C = len(class_names)

    # compute avg probs per true class
    avg_probs = np.zeros((C, C), dtype=float)
    Ns = np.zeros(C, dtype=int)
    for t in range(C):
        mask = (y_true_q0 == t)
        Ns[t] = int(mask.sum())
        if Ns[t] > 0:
            avg_probs[t] = y_q0_proba[mask].mean(axis=0)

    # figure
    fig, axes = plt.subplots(1, C, figsize=figsize, sharey=True, constrained_layout=True)
    if C == 1:
        axes = [axes]
    x = np.arange(C)

    ymax = float(np.max(avg_probs)) if avg_probs.size else 1.0
    ylim_top = max(0.05, min(1.05, ymax + 0.12))

    for t, ax in enumerate(axes):
        bars = ax.bar(x, avg_probs[t], alpha=default_alpha, tick_label=class_names)
        ax.set_ylim(0, ylim_top)
        ax.set_title(f"True {class_names[t]} (N={Ns[t]})")
        if t == 0:
            ax.set_ylabel("Average probability")

        for j, b in enumerate(bars):
            h = b.get_height()
            ax.text(b.get_x()+b.get_width()/2, h + 0.02, f"{h:.3f}",
                    ha="center", va="bottom", clip_on=False)

        ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle(suptitle)

    if save_path:
        base = os.path.splitext(save_path)[0]
        os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
        fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(base + ".pdf", bbox_inches="tight")

    if show:
        plt.show()
    return fig

def plot_prob_heatmap(probs, y_true, class_names, 
                    title=r"mean predicted distribution per true class",
                    save_path=None,
                    show=False):
    """Heatmap of mean predicted distribution per true class."""
    C = probs.shape[1]
    means = np.zeros((C, C), dtype=np.float32)
    for c in range(C):
        mask = (y_true == c)
        means[c] = probs[mask].mean(axis=0) if np.any(mask) else np.nan

    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(means, vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(C)); ax.set_xticklabels(class_names)
    ax.set_yticks(range(C)); ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted class'); ax.set_ylabel('True class')
    ax.set_title(title)
    # annotate
    for i in range(C):
        for j in range(C):
            txt = '' if np.isnan(means[i,j]) else f'{means[i,j]:.2f}'
            ax.text(j, i, txt, ha='center', va='center')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    if save_path:
        base = os.path.splitext(save_path)[0]
        os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
        fig.savefig(save_path+".png", dpi=300, bbox_inches='tight')
        fig.savefig(save_path + ".pdf", bbox_inches="tight")
    if show:
        plt.show()
    return fig