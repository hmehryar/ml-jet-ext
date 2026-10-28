"""Serializable moment-loss settings; safe to import during CPU-only staging."""
import math


def resolve_moment_loss(settings=None):
    settings = settings or {}
    kind = settings.get('type', 'cross_entropy')
    if kind not in ('cross_entropy', 'ce_ordinal'):
        raise ValueError('Moment loss.type must be cross_entropy or ce_ordinal')
    weight = float(settings.get('ordinal_weight', 1.0 if kind == 'ce_ordinal' else 0.0))
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('loss.ordinal_weight must be finite and nonnegative')
    if kind == 'cross_entropy' and weight != 0:
        raise ValueError('Use loss.type=ce_ordinal for a nonzero ordinal_weight')
    return {'type': kind, 'ordinal_weight': weight,
            'ordinal_normalization': 'adjacent_physical_gap / full_parameter_range'}
