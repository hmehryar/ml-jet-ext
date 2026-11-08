"""T04-10-A5: globally weighted, task-aware classification ranking metrics.

Standalone utility: accepts logits/targets from either single- or dual-head models;
never modifies model state or evaluates test data. Class ordering follows the
extended dataset's explicit alpha and q0 grids.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch

CLASS_VALUES = {
    'alpha_only': (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
    'q0_only': (1.0, 1.5, 2.0, 2.5, 3.0),
}


def ordered_predictions(logits: torch.Tensor) -> torch.Tensor:
    """Sort descending by score, resolve ties by lower class index.

    Computation uses float32 for consistent BF16/FP32 ranking. NaN/Inf
    logits are rejected rather than silently producing misleading metrics.
    """
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
        raise ValueError('logits must have shape [N, C]')
    scores = logits.detach().to(dtype=torch.float32, device='cpu')
    if not torch.isfinite(scores).all():
        raise ValueError('Non-finite logits')
    return torch.argsort(scores, dim=1, descending=True, stable=True)


class ClassificationMetrics:
    """Streaming global top-k and physical-order metrics, with optional records."""

    def __init__(self, task: str, ks: Sequence[int] = (1, 2, 3), *, collect_records=False):
        if task not in CLASS_VALUES:
            raise ValueError('Unsupported task')
        self.task = task
        self.values = torch.tensor(CLASS_VALUES[task], dtype=torch.float64)
        self.classes = self.values.numel()
        if not ks or any(type(k) is not int or k < 1 or k > self.classes for k in ks):
            raise ValueError('top-k values must be integers between 1 and class count')
        self.ks = tuple(sorted(set(ks)))
        self.n = 0
        self.topk_hits = {k: 0 for k in self.ks}
        self.within_1 = self.within_2 = 0
        self.abs_sum = self.sq_sum = 0.0
        self.collect_records = bool(collect_records)
        self.records = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor, *, sample_ids=None):
        ranking = ordered_predictions(logits)
        truth = targets.detach().to(dtype=torch.long, device='cpu')
        if truth.ndim != 1 or len(truth) != len(ranking) or ranking.shape[1] != self.classes:
            raise ValueError('Label/logit dimensions do not match task')
        if (truth < 0).any() or (truth >= self.classes).any():
            raise ValueError('Target class outside configured grid')
        if sample_ids is not None and len(sample_ids) != len(truth):
            raise ValueError('sample_ids must match batch size')
        if self.collect_records and sample_ids is None:
            raise ValueError('Prediction export requires stable sample IDs')
        if sample_ids is not None and any(not isinstance(s, str) or not s for s in sample_ids):
            raise ValueError('Invalid sample ID')
        pred = ranking[:, 0]
        dist = (pred - truth).abs()
        self.within_1 += int((dist <= 1).sum())
        self.within_2 += int((dist <= 2).sum())
        physical_difference = self.values[pred] - self.values[truth]
        self.abs_sum += float(physical_difference.abs().sum())
        self.sq_sum += float(physical_difference.square().sum())
        for k in self.ks:
            self.topk_hits[k] += int((ranking[:, :k] == truth[:, None]).any(dim=1).sum())
        self.n += len(truth)
        if self.collect_records:
            scores = logits.detach().to(device='cpu', dtype=torch.float32).softmax(dim=1)
            for i, sid in enumerate(sample_ids):
                order = ranking[i].tolist()
                self.records.append({
                    'sample_id': sid, 'task': self.task,
                    'true_class': int(truth[i]), 'true_value': float(self.values[truth[i]]),
                    'predicted_class': int(pred[i]), 'predicted_value': float(self.values[pred[i]]),
                    'logits': logits[i].detach().to(device='cpu', dtype=torch.float32).tolist(),
                    'probabilities': scores[i].tolist(),
                    'ranked_classes': order,
                    'ranked_probabilities': [float(scores[i, j]) for j in order],
                })

    def compute(self):
        if not self.n:
            raise ValueError('No evaluation samples')
        out = {f'top{k}_accuracy': self.topk_hits[k] / self.n for k in self.ks}
        out.update(within_1_class_accuracy=self.within_1 / self.n,
                   within_2_class_accuracy=self.within_2 / self.n,
                   physical_mae=self.abs_sum / self.n,
                   physical_rmse=math.sqrt(self.sq_sum / self.n))
        return out
