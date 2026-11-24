"""T04-04 parameter-group optimizer construction for CLIP/SigLIP moment models.

Standalone and opt-in for Stage 1. Does not change legacy T03 optimizer behavior.
"""
from __future__ import annotations

import math

import torch

GROUPS = ('backbone', 'adapter', 'heads')


def _valid_rate(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be a finite, positive number')
    return float(value)


def _valid_weight_decay(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('weight_decay must be finite and nonnegative')
    return float(value)


def parameter_groups(model, learning_rates, *, weight_decay=0.01):
    """Assign each trainable tensor exactly once to a named optimizer group.

    Frozen tensors stay excluded. Empty groups (e.g. S1 backbone) are omitted.
    All trainable parameters are checked by identity before returning.
    """
    if not isinstance(learning_rates, dict) or set(learning_rates) != set(GROUPS):
        raise ValueError('learning_rates must contain exactly backbone, adapter and heads')
    rates = {name: _valid_rate(learning_rates[name], name) for name in GROUPS}
    decay = _valid_weight_decay(weight_decay)
    if not hasattr(model, 'backbone') or not hasattr(model, 'input_adapter') or not hasattr(model, 'alpha_head') or not hasattr(model, 'q0_head'):
        raise TypeError('Expected moment vision classifier with backbone, input_adapter and two heads')
    modules = {'backbone': (model.backbone,), 'adapter': (model.input_adapter,),
               'heads': (model.alpha_head, model.q0_head)}
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    seen = set()
    result = []
    for name in GROUPS:
        params = []
        for module in modules[name]:
            for p in module.parameters():
                if not p.requires_grad:
                    continue
                identity = id(p)
                if identity in seen:
                    raise RuntimeError('Trainable tensor appears in multiple optimizer groups')
                seen.add(identity)
                params.append(p)
        if params:
            result.append({'params': params, 'lr': rates[name], 'weight_decay': decay, 'name': name})
    if not expected or seen != expected:
        raise RuntimeError('Optimizer groups do not exactly cover trainable parameters')
    return result


def build_t04_optimizer(model, learning_rates, *, optimizer_type='adamw', weight_decay=0.01):
    """Build AdamW or explicitly requested legacy Adam with deterministic groups."""
    if optimizer_type not in ('adam', 'adamw'):
        raise ValueError('optimizer_type must be adam or adamw')
    if optimizer_type == 'adam' and weight_decay != 0:
        raise ValueError('Legacy Adam requires weight_decay=0 to preserve historical behavior')
    groups = parameter_groups(model, learning_rates, weight_decay=weight_decay)
    cls = torch.optim.AdamW if optimizer_type == 'adamw' else torch.optim.Adam
    return cls(groups)


def learning_rate_summary(optimizer):
    """Stable named LR mapping for checkpoint/epoch telemetry."""
    summary = {}
    for group in optimizer.param_groups:
        name = group.get('name')
        if name not in GROUPS or name in summary:
            raise ValueError('Optimizer has missing or duplicate group name')
        summary[name] = float(group['lr'])
    return summary
