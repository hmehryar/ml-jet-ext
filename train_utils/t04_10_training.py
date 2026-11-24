"""T04-10 task-isolated single-head epoch training and validation.

Does not change train_utils.moment_training or train_utils.t04_precision.
"""
from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn

from models.t04_10_single_head import TASK_CLASSES, TASK_OUTPUT, validate_task
from train_utils.t04_10_metrics import ClassificationMetrics


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
                          optimizer=None, precision="fp32", criterion=None,
                          prediction_export_path=None, sample_ids=None,
                          export_metadata=None, split="val", telemetry_counts=None):
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
    if telemetry_counts is not None:
        telemetry_counts.update(optimizer_steps=0, optimizer_steps_completed=0,
                                optimizer_steps_skipped=0)
    if prediction_export_path is not None:
        if training or split != "val":
            raise ValueError("Prediction export is validation-only and evaluation-only")
        if sample_ids is None:
            raise ValueError("Stable sample_ids are required for export")
        required = ("run_id", "architecture", "strategy", "moments", "checkpoint_sha256")
        if not isinstance(export_metadata, dict) or any(k not in export_metadata for k in required):
            raise ValueError(f"Export metadata requires {required}")
        if export_metadata.get("split", "val") != "val":
            raise ValueError("Export metadata must identify validation split")
    model.train(training)
    device = torch.device(device)
    context = (torch.autocast(device_type=device.type, dtype=torch.bfloat16)
               if precision == "bf16" else nullcontext())
    n = 0
    loss_sum = 0.0
    confusion = torch.zeros((TASK_CLASSES[task], TASK_CLASSES[task]), dtype=torch.int64)
    ranking_metrics = ClassificationMetrics(task, collect_records=prediction_export_path is not None)
    sample_offset = 0
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
                if telemetry_counts is not None:
                    telemetry_counts['optimizer_steps'] += 1
                    telemetry_counts['optimizer_steps_completed'] += 1
            pred = logits.detach().argmax(1).cpu()
            truth = y.detach().cpu()
            count = int(y.numel())
            n += count
            loss_sum += float(loss.detach()) * count
            confusion += torch.bincount(truth * TASK_CLASSES[task] + pred,
                                        minlength=TASK_CLASSES[task] ** 2).reshape(confusion.shape)
            batch_ids = (sample_ids[sample_offset:sample_offset + count]
                         if sample_ids is not None else None)
            ranking_metrics.update(logits.detach(), truth, sample_ids=batch_ids)
            sample_offset += count
    if not n:
        raise ValueError("Empty loader")
    tp = confusion.diag().to(torch.float64)
    predicted = confusion.sum(0).to(torch.float64)
    actual = confusion.sum(1).to(torch.float64)
    denom = predicted + actual
    macro_f1 = torch.where(denom > 0, 2 * tp / denom.clamp_min(1), torch.zeros_like(tp)).mean()
    metrics = {"task": task, "n": n, "loss": loss_sum / n,
               "accuracy": float(tp.sum() / n), "macro_f1": float(macro_f1),
               "confusion_matrix": confusion.tolist()}
    metrics.update(ranking_metrics.compute())
    if abs(metrics["accuracy"] - metrics["top1_accuracy"]) > 1e-12:
        raise AssertionError("Top-1 and legacy accuracy disagree")
    if prediction_export_path is not None:
        if sample_offset != len(sample_ids) or len(set(sample_ids)) != len(sample_ids):
            raise ValueError("Export requires one unique ID per evaluation sample")
        from pathlib import Path
        import json
        import os
        import tempfile
        destination = Path(prediction_export_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite prediction export: {destination}")
        metadata = dict(export_metadata or {})
        if metadata.get("split", "val") != "val":
            raise ValueError("Export metadata must identify validation split")
        metadata["split"] = "val"
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf8", dir=destination.parent,
                                         suffix=".tmp", delete=False) as handle:
            tmp_name = handle.name
            json.dump({"schema": 1, "task": task, "class_values": ranking_metrics.values.tolist(),
                       "metadata": metadata, "metrics": metrics,
                       "predictions": ranking_metrics.records}, handle)
        try:
            os.link(tmp_name, destination)  # no-clobber publish; do not overwrite on rerun
        finally:
            os.unlink(tmp_name)
    return metrics
