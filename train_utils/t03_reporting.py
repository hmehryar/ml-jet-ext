"""T03 validation-only reporting for cumulative M1-M5 runs.

Works with saved T02/T03 checkpoints and metric CSVs. The held-out test
partition must never be supplied to this module during model development.
"""
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from data.moment_loader import ALPHA_VALUES, Q0_VALUES
from train_utils.ordinal_loss import physical_class_metrics

HEADS = {"alpha_output": ("alpha", ["0.20", "0.25", "0.30", "0.35", "0.40", "0.45", "0.50"]),
         "q0_output": ("q0", ["1.0", "1.5", "2.0", "2.5", "3.0"])}


def _save_figure(fig, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _metric_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("No epoch metrics were recorded")
    return rows


def _curve(rows, columns, title, ylabel, path):
    xs = [int(row["epoch"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted = 0
    for key, label in columns:
        if key not in rows[0]:
            continue
        yy = []
        for row in rows:
            try:
                yy.append(float(row[key]))
            except (TypeError, ValueError):
                yy.append(float("nan"))
        if all(math.isnan(y) for y in yy):
            continue
        ax.plot(xs, yy, marker="o", markersize=2.5, linewidth=1.5, label=label)
        plotted += 1
    if not plotted:
        plt.close(fig)
        return False
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, path)
    return True


def plot_epoch_metrics(metrics_path, plot_dir):
    rows = _metric_rows(metrics_path)
    dest = Path(plot_dir)
    made = []
    defs = [
        ("train_val_loss", [("train_loss", "Train"), ("val_loss", "Validation")],
         "Total training objective", "Loss"),
        ("train_val_accuracy", [("train_joint_accuracy", "Train joint"), ("val_joint_accuracy", "Validation joint"),
                                ("train_alpha_accuracy", "Train alpha"), ("val_alpha_accuracy", "Validation alpha"),
                                ("train_q0_accuracy", "Train Q0"), ("val_q0_accuracy", "Validation Q0")],
         "Train vs. validation accuracy", "Accuracy"),
        ("alpha_loss_accuracy", [("train_alpha_loss", "Train alpha loss"), ("val_alpha_loss", "Validation alpha loss")],
         "Alpha classification loss", "Cross-entropy"),
        ("q0_loss_accuracy", [("train_q0_loss", "Train Q0 loss"), ("val_q0_loss", "Validation Q0 loss")],
         "Q0 classification loss", "Cross-entropy"),
        ("alpha_accuracy", [("train_alpha_accuracy", "Train alpha"), ("val_alpha_accuracy", "Validation alpha")],
         "Alpha classification accuracy", "Accuracy"),
        ("q0_accuracy", [("train_q0_accuracy", "Train Q0"), ("val_q0_accuracy", "Validation Q0")],
         "Q0 classification accuracy", "Accuracy"),
        ("macro_f1", [("val_alpha_macro_f1", "Alpha F1"), ("val_q0_macro_f1", "Q0 F1")],
         "Validation macro-F1", "Macro-F1"),
        ("learning_rate", [("lr", "Learning rate")], "Learning-rate schedule", "Learning rate"),
    ]
    for tag, unit in (('alpha', 'alpha'), ('q0', 'GeV')):
        defs.append((f'{tag}_physical_mae',
                     [(f'train_{tag}_mae', 'Train'), (f'val_{tag}_mae', 'Validation')],
                     f'{tag} physical class error', f'MAE ({unit})'))
        defs.append((f'{tag}_loss_components',
                     [(f'train_{tag}_ce_loss', 'Train CE'), (f'val_{tag}_ce_loss', 'Validation CE'),
                      (f'train_{tag}_ordinal_loss', 'Train ordinal'),
                      (f'val_{tag}_ordinal_loss', 'Validation ordinal')],
                     f'{tag} loss components (ordinal before lambda weighting)', 'Loss'))
    for stem, cols, title, ylabel in defs:
        if _curve(rows, cols, title, ylabel, dest / stem):
            made.append(stem)
    return made


def _load_best(model, checkpoint_path, device):
    # This is the T03 model checkpoint authored by our own training loop.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return checkpoint


@torch.inference_mode()
def _collect_validation(model, val_loader, device):
    y_true = {key: [] for key in HEADS}
    y_prob = {key: [] for key in HEADS}
    for images, labels in val_loader:
        logits = model(images.to(device, non_blocking=True))
        for key in HEADS:
            probabilities = torch.softmax(logits[key], dim=1)
            y_true[key].append(labels[key].cpu().numpy())
            y_prob[key].append(probabilities.cpu().numpy())
    return {key: (np.concatenate(y_true[key]), np.concatenate(y_prob[key])) for key in HEADS}


def _plot_confusion(truth, predictions, classes, path, title):
    mat = confusion_matrix(truth, predictions, labels=list(range(len(classes))))
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(mat, cmap="Oranges", interpolation="nearest")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    _save_figure(fig, path)
    return mat.tolist()


def _plot_histograms(truth, probabilities, labels, path_prefix, title):
    predictions = probabilities.argmax(axis=1)
    loc = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    width = 0.35
    ax.bar(loc - width / 2, np.bincount(truth, minlength=len(labels)), width, label="Ground truth")
    ax.bar(loc + width / 2, np.bincount(predictions, minlength=len(labels)), width, label="Prediction")
    ax.set_xticks(loc, labels)
    ax.set_xlabel("Class")
    ax.set_ylabel("Sample count")
    ax.set_title(title + " — predicted class distribution")
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, str(path_prefix) + "_prediction_histogram")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(probabilities.max(axis=1), bins=np.linspace(0, 1, 26), edgecolor="white")
    ax.set(xlabel="Maximum predicted class probability", ylabel="Validation samples",
           title=title + " — prediction confidence", xlim=(0, 1))
    fig.tight_layout()
    _save_figure(fig, str(path_prefix) + "_probability_distribution")


def generate_t03_report(model, validation_loader, device, output_dir, expected_samples=1400):
    """Restore *best* checkpoint and report exclusively on validation samples."""
    out = Path(output_dir)
    plots = out / "plots"
    checkpoint_file = out / "best_model.pth"
    metrics_file = out / "moment_epoch_metrics.csv"
    if not checkpoint_file.is_file() or not metrics_file.is_file():
        raise FileNotFoundError("Missing best_model.pth or moment_epoch_metrics.csv")
    if len(validation_loader.dataset) != expected_samples:
        raise ValueError("Reporting must use exactly the held-out validation set, not test")
    created = plot_epoch_metrics(metrics_file, plots)
    checkpoint = _load_best(model, checkpoint_file, device)
    data = _collect_validation(model, validation_loader, device)
    summary = {"best_epoch": int(checkpoint["epoch"]), "split": "validation",
               "num_validation_samples": int(expected_samples), "moments": checkpoint.get("moments"),
               "metric_plots": created}
    summary['loss_config'] = checkpoint.get('loss_config')
    summary['criterion_settings'] = checkpoint.get('criterion_settings')
    predictions = {}
    for key, (label, class_names) in HEADS.items():
        truth, probs = data[key]
        if len(truth) != expected_samples:
            raise ValueError("Validation collection size mismatch")
        pred = probs.argmax(axis=1)
        predictions[key] = pred
        mat = _plot_confusion(truth, pred, class_names, plots / f"{label}_confusion_matrix",
                              f"{label} validation confusion matrix")
        _plot_histograms(truth, probs, class_names, plots / label, label)
        summary[label] = {"accuracy": float(accuracy_score(truth, pred)),
                          "macro_f1": float(f1_score(truth, pred, average="macro", labels=list(range(len(class_names))), zero_division=0)),
                          "confusion_matrix": mat}
        values = ALPHA_VALUES if key == 'alpha_output' else Q0_VALUES
        summary[label].update(physical_class_metrics(truth.tolist(), pred.tolist(), values))
    ta = data["alpha_output"][0]
    tq = data["q0_output"][0]
    summary["joint_accuracy"] = float(np.mean((predictions["alpha_output"] == ta) &
                                              (predictions["q0_output"] == tq)))
    summary["best_checkpoint"] = str(checkpoint_file)
    (out / "best_model_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== T03 BEST MODEL SUMMARY (VALIDATION ONLY) ===")
    print(f"Best epoch: {summary['best_epoch']}")
    print(f"Alpha accuracy: {summary['alpha']['accuracy']:.4f} | macro-F1: {summary['alpha']['macro_f1']:.4f}")
    print(f"Q0 accuracy: {summary['q0']['accuracy']:.4f} | macro-F1: {summary['q0']['macro_f1']:.4f}")
    print(f"Joint exact-match accuracy: {summary['joint_accuracy']:.4f}")
    print(f"Saved plots to: {plots}")
    return summary
