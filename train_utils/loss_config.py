"""Typed, backward-compatible moment loss configuration (T04-07)."""
import math

KINDS = ('cross_entropy', 'ce_ordinal', 'physical_mae', 'ce_physical_mae')


def resolve_moment_loss(settings=None):
    if settings is None: settings = {}
    if not isinstance(settings, dict): raise ValueError('moment_loss must be a mapping')
    kind = settings.get('type', 'cross_entropy')
    # Do not silently reinterpret historical configurations.
    if kind not in KINDS: raise ValueError(f'Unsupported moment loss.type: {kind}')
    fields = {'cross_entropy': {'type', 'ordinal_weight'}, 'ce_ordinal': {'type', 'ordinal_weight'},
              'physical_mae': {'type'}, 'ce_physical_mae': {'type', 'mae_weight'}}
    # Accept only canonical metadata emitted by this resolver.
    metadata = {}
    if kind in ('cross_entropy', 'ce_ordinal'):
        metadata['ordinal_normalization'] = (
            'adjacent_physical_gap / full_parameter_range'
        )
    else:
        metadata['physical_normalization'] = 'full_parameter_range'
        metadata['physical_objective'] = 'expected_absolute_deviation'

    extra = set(settings) - fields[kind] - set(metadata)
    if extra:
        raise ValueError(
            f'Unknown settings for {kind}: {sorted(extra)}'
        )
    for key, expected in metadata.items():
        if key in settings and settings[key] != expected:
            raise ValueError(
                f'Invalid {key} for {kind}: {settings[key]!r}'
            )
    weight = settings.get('ordinal_weight', 1.0) if kind == 'ce_ordinal' else settings.get('mae_weight', 1.0)
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
        raise ValueError('Loss coefficient must be finite and nonnegative')
    if kind == 'cross_entropy':
        if settings.get('ordinal_weight', 0.0) != 0 or isinstance(settings.get('ordinal_weight', 0.0), bool):
            raise ValueError('CE requires ordinal_weight=0')
        # Preserve historical serialized CE contract for old checkpoints/manifests.
        return {'type': 'cross_entropy', 'ordinal_weight': 0.0,
                'ordinal_normalization': 'adjacent_physical_gap / full_parameter_range'}
    if kind == 'ce_ordinal':
        return {'type': kind, 'ordinal_weight': float(weight),
                'ordinal_normalization': 'adjacent_physical_gap / full_parameter_range'}
    if kind == 'physical_mae':
        return {'type': kind, 'physical_normalization': 'full_parameter_range',
                'physical_objective': 'expected_absolute_deviation'}
    return {'type': kind, 'mae_weight': float(weight),
            'physical_normalization': 'full_parameter_range',
            'physical_objective': 'expected_absolute_deviation'}
