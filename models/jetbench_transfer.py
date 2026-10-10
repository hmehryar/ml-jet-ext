"""Verified transfer of the historical JetBench ViT-Tiny backbone to moment inputs.

Does not modify the existing factory or training loop. Use in a pilot first.
"""
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from torch import nn


class MomentInputAdapter(nn.Module):
    """Trainable C->3 projection; deterministic M1-replication initialization."""
    def __init__(self, channels: int):
        super().__init__()
        if not 1 <= channels <= 5:
            raise ValueError(f"Expected 1..5 moments, got {channels}")
        self.projection = nn.Conv2d(channels, 3, kernel_size=1, bias=False)
        with torch.no_grad():
            self.projection.weight.zero_()
            self.projection.weight[:, 0, 0, 0] = 1.0

    def forward(self, x):
        return self.projection(x)


class AdaptedMomentViT(nn.Module):
    def __init__(self, base_model: nn.Module, channels: int):
        super().__init__()
        self.input_adapter = MomentInputAdapter(channels)
        self.model = base_model

    def forward(self, x):
        return self.model(self.input_adapter(x))


def read_trusted_jetbench_checkpoint(path):
    """Read a locally-authored historical checkpoint using restricted loading."""
    safe = [np.core.multiarray.scalar, np.dtype, type(np.dtype('float64'))]
    with torch.serialization.safe_globals(safe):
        checkpoint = torch.load(str(Path(path)), map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError('Expected checkpoint dictionary')
    state = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    if not isinstance(state, (dict, OrderedDict)):
        raise TypeError('Checkpoint has no usable state_dict')
    return checkpoint, state


def transfer_backbone(base_model, source_state, min_coverage=0.90):
    """Load only name+shape-matched backbone tensors, validating coverage by numel."""
    destination = base_model.state_dict()
    # Several historical implementations register the SAME backbone under
    # both `backbone` and `features`. Preserve duplicate aliases if present.
    backbone_prefixes = ('backbone.', 'features.')
    eligible = {k: v for k, v in destination.items()
                if k.startswith(backbone_prefixes) and torch.is_tensor(v)}
    if not eligible:
        raise RuntimeError('No backbone.* or features.* tensors found in target; inspect model names')
    matching = {k: v for k, v in source_state.items()
                if k in eligible and torch.is_tensor(v) and v.shape == eligible[k].shape}
    total = sum(v.numel() for v in eligible.values())
    covered = sum(eligible[k].numel() for k in matching)
    coverage = covered / total
    if coverage < min_coverage:
        missed = [k for k in eligible if k not in matching]
        raise RuntimeError(f'Backbone tensor coverage {coverage:.1%} < {min_coverage:.0%}; '
                           f'first unmatched keys: {missed[:12]}')
    # This load cannot replace output heads: only backbone/features are supplied.
    base_model.load_state_dict(matching, strict=False)
    # Verify every loaded value actually landed in model state.
    loaded = base_model.state_dict()
    assert all(torch.equal(loaded[k].cpu(), v.cpu()) for k, v in matching.items())
    return {'coverage': coverage, 'matched_tensors': len(matching),
            'target_backbone_tensors': len(eligible), 'source_tensors': len(source_state)}
