"""T02.4: moment-only training/evaluation; legacy three-head code is untouched."""
import csv
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm

HEADS = ("alpha_output", "q0_output")


def _check_batch(outputs, labels):
    for key, nclasses in zip(HEADS, (7, 5)):
        if key not in outputs or key not in labels:
            raise KeyError(f"Missing moment head/target: {key}")
        if outputs[key].ndim != 2 or outputs[key].shape[1] != nclasses:
            raise ValueError(f"Expected {key} logits [B,{nclasses}], got {tuple(outputs[key].shape)}")
        if labels[key].ndim != 1 or labels[key].shape[0] != outputs[key].shape[0]:
            raise ValueError(f"Expected {key} labels [B]")
        if labels[key].dtype != torch.long:
            raise TypeError(f"Expected long class labels for {key}")


def _run(loader, model, criterion, device, loss_weights=None, optimizer=None):
    if len(loader) == 0:
        raise ValueError("Cannot process empty DataLoader")
    training = optimizer is not None
    model.train(training)
    weights = loss_weights or {}
    if "energy_loss_output" in weights:
        raise ValueError("Moment dataset does not support energy-loss weights")
    totals = {"loss": 0.0, "loss_alpha": 0.0, "loss_q0": 0.0}
    truth, guesses = {k: [] for k in HEADS}, {k: [] for k in HEADS}
    correct_both = 0
    total = 0
    iterator = tqdm(loader, desc="Moment train" if training else "Moment eval", leave=False, disable=not training)
    with torch.set_grad_enabled(training):
        for images, label_batch in iterator:
            images = images.to(device, non_blocking=True)
            labels = {key: label_batch[key].to(device, non_blocking=True) for key in HEADS}
            outputs = model(images)
            _check_batch(outputs, labels)
            losses = {key: criterion[key](outputs[key], labels[key]) for key in HEADS}
            batch_loss = sum(float(weights.get(key, 1.0)) * losses[key] for key in HEADS)
            if training:
                optimizer.zero_grad(set_to_none=True)
                batch_loss.backward()
                optimizer.step()
            b = images.size(0)
            total += b
            totals["loss"] += batch_loss.item() * b
            totals["loss_alpha"] += losses["alpha_output"].item() * b
            totals["loss_q0"] += losses["q0_output"].item() * b
            pred = {key: outputs[key].argmax(dim=1) for key in HEADS}
            correct_both += ((pred["alpha_output"] == labels["alpha_output"]) &
                             (pred["q0_output"] == labels["q0_output"])).sum().item()
            for key in HEADS:
                truth[key].extend(labels[key].detach().cpu().tolist())
                guesses[key].extend(pred[key].detach().cpu().tolist())
    if not total:
        raise ValueError("Empty moment dataset")
    result = {key: v / total for key, v in totals.items()}
    result["accuracy"] = correct_both / total  # both heads correct, same sample
    for key, tag, count in (("alpha_output", "alpha", 7), ("q0_output", "q0", 5)):
        y, p = truth[key], guesses[key]
        result[tag] = {
            "accuracy": float(accuracy_score(y, p)),
            "precision": float(precision_score(y, p, average="macro", labels=list(range(count)), zero_division=0)),
            "recall": float(recall_score(y, p, average="macro", labels=list(range(count)), zero_division=0)),
            "f1": float(f1_score(y, p, average="macro", labels=list(range(count)), zero_division=0)),
            "confusion_matrix": confusion_matrix(y, p, labels=list(range(count))).tolist(),
        }
        result[f"accuracy_{tag}"] = result[tag]["accuracy"]
    return result


def train_moment_epoch(loader, model, criterion, optimizer, device, loss_weights=None):
    return _run(loader, model, criterion, device, loss_weights, optimizer)


def evaluate_moment(loader, model, criterion, device, loss_weights=None):
    return _run(loader, model, criterion, device, loss_weights)


def run_moment_training_loop(cfg, train_loader, val_loader, device, model, criterion,
                             optimizer, scheduler=None, start_epoch=0):
    """Run moment-only training, record CSV and save best/last checkpoints."""
    if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
        raise ValueError("OneCycleLR requires per-batch scheduling; not supported by moment epoch loop")
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    weights = getattr(cfg, "loss_weights", None)
    best_acc, best_epoch, best_metrics, stagnant = -1.0, -1, None, 0
    csv_file = out / "moment_epoch_metrics.csv"
    cols = ["epoch", "train_loss", "train_alpha_accuracy", "train_q0_accuracy",
            "train_joint_accuracy", "val_loss", "val_alpha_accuracy",
            "val_q0_accuracy", "val_joint_accuracy", "val_alpha_macro_f1", "val_q0_macro_f1", "lr"]
    for epoch in range(start_epoch, cfg.epochs):
        train = train_moment_epoch(train_loader, model, criterion, optimizer, device, weights)
        val = evaluate_moment(val_loader, model, criterion, device, weights)
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val["accuracy"])
            else:
                scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        record = dict(epoch=epoch + 1, train_loss=train["loss"],
                      train_alpha_accuracy=train["alpha"]["accuracy"],
                      train_q0_accuracy=train["q0"]["accuracy"],
                      train_joint_accuracy=train["accuracy"],
                      val_loss=val["loss"], val_alpha_accuracy=val["alpha"]["accuracy"],
                      val_q0_accuracy=val["q0"]["accuracy"],
                      val_joint_accuracy=val["accuracy"],
                      val_alpha_macro_f1=val["alpha"]["f1"],
                      val_q0_macro_f1=val["q0"]["f1"], lr=lr)
        with csv_file.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(record)
        state = {"epoch": epoch + 1, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(), "metrics": val,
                 "dataset_type": "moment", "moments": getattr(cfg, "moments", None)}
        torch.save(state, out / "last_model.pth")
        if val["accuracy"] > best_acc:
            best_acc, best_epoch, best_metrics, stagnant = val["accuracy"], epoch + 1, val, 0
            torch.save(state, out / "best_model.pth")
        else:
            stagnant += 1
        print(f"[Moment] Epoch {epoch + 1}: val alpha={val['alpha']['accuracy']:.4f}, "
              f"q0={val['q0']['accuracy']:.4f}, joint={val['accuracy']:.4f}, "
              f"loss={val['loss']:.4f}")
        if stagnant >= cfg.patience:
            break
    return best_epoch, best_acc, best_metrics
