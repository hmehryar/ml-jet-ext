"""T04-10 task-isolated single-head epoch training and validation.

Does not change train_utils.moment_training or train_utils.t04_precision.
"""
from __future__ import annotations

from contextlib import nullcontext
import math

import torch
from torch import nn

from models.t04_10_single_head import TASK_CLASSES, TASK_OUTPUT, validate_task

ALPHA_VALUES = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
Q0_VALUES = (1.0, 1.5, 2.0, 2.5, 3.0)


def _extract(batch, task, device):
    if isinstance(batch, dict):
        x = batch.get("image", batch.get("images"))
        if x is None:
            # Actual moment loader uses `moments`, checked separately.
            x = batch.get("moments")
        y = batch[TASK_OUTPUT[task]]
    elif isinstance(batch, (tuple, list)) and len(batch) == 2:
        x, labels = batch
        y = labels[TASK_OUTPUT[task]] if isinstance(labels, dict) else labels
    else:
        raise TypeError("Batch must contain images and selected target")
    if x is None:
        raise KeyError("Expected image/images/moments in the batch")
    if not isinstance(y, torch.Tensor):
        raise TypeError("Selected target must be a tensor")
    return x.to(device, non_blocking=True), y.to(device, dtype=torch.long, non_blocking=True)


def run_single_head_epoch(loader, model, *, task: str, device,
                          optimizer=None, precision="fp32", criterion=None):
    """Return selected-head metrics; no inactive-head or joint metrics.

    Metrics accumulated globally, not mean of per-batch percentages.
    For BF16 CUDA autocast, gradient scaling is not needed.
    """
    validate_task(task)
    if precision not in ("fp32", "bf16"):
        raise ValueError("Single-head checkpoint 1 supports fp32/bf16 only")
    if getattr(model, "task", None) != task:
        raise ValueError("Model task and requested task disagree")
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    training = optimizer is not None
    model.train(training)
    device = torch.device(device)
    context = (torch.autocast(device_type=device.type, dtype=torch.bfloat16)
               if precision == "bf16" else nullcontext())
    n = 0
    loss_sum = 0.0
    confusion = torch.zeros((TASK_CLASSES[task], TASK_CLASSES[task]), dtype=torch.int64)
    abs_sum = sq_sum = 0.0
    values = torch.tensor(ALPHA_VALUES if task == "alpha_only" else Q0_VALUES,
                          dtype=torch.float64)
    grad_context = torch.enable_grad() if training else torch.no_grad()
    with grad_context:
        for batch in loader:
            x, y = _extract(batch, task, device)
            if y.ndim != 1 or y.numel() != x.shape[0]:
                raise ValueError("Selected target must be [B]")
            if torch.any(y < 0) or torch.any(y >= TASK_CLASSES[task]):
                raise ValueError("Selected class index out of range")
            if training:
                optimizer.zero_grad(set_to_none=True)
            with context:
                outputs = model(x)
                if set(outputs) != {TASK_OUTPUT[task]}:
                    raise RuntimeError("Model must emit exactly its selected head")
                logits = outputs[TASK_OUTPUT[task]]
                if logits.shape != (y.shape[0], TASK_CLASSES[task]):
                    raise RuntimeError("Wrong single-head output dimensions")
                loss = criterion(logits.float(), y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite selected-task loss")
            if training:
                loss.backward()
                optimizer.step()
            pred = logits.detach().argmax(1).cpu()
            truth = y.detach().cpu()
            count = int(y.numel())
            n += count
            loss_sum += float(loss.detach()) * count
            confusion += torch.bincount(truth * TASK_CLASSES[task] + pred,
                                        minlength=TASK_CLASSES[task] ** 2).reshape(confusion.shape)
            difference = values[pred] - values[truth]
            abs_sum += float(difference.abs().sum())
            sq_sum += float((difference ** 2).sum())
    if not n:
        raise ValueError("Empty loader")
    tp = confusion.diag().to(torch.float64)
    predicted = confusion.sum(0).to(torch.float64)
    actual = confusion.sum(1).to(torch.float64)
    denom = predicted + actual
    macro_f1 = torch.where(denom > 0, 2 * tp / denom.clamp_min(1), torch.zeros_like(tp)).mean()
    return {"task": task, "n": n, "loss": loss_sum / n,
            "accuracy": float(tp.sum() / n), "macro_f1": float(macro_f1),
            "physical_mae": abs_sum / n, "physical_rmse": math.sqrt(sq_sum / n),
            "confusion_matrix": confusion.tolist()}
