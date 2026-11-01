"""T04-03 adaptation masks for CLIP and SigLIP vision encoders.

Call before optimizer construction. No checkpoint loading occurs in this module.
"""
from __future__ import annotations

STRATEGIES = ('S1', 'S2', 'S3', 'S4')


def apply_strategy(backbone, strategy):
    """Configure trainability; return trainable transformer block indices.

    S1: frozen backbone; S2: last two blocks and final pooling/normalization;
    S3: fully trainable pretrained encoder; S4: fully trainable scratch encoder.
    The caller must enforce S3 pretrained / S4 scratch initialization.
    """
    if strategy not in STRATEGIES:
        raise ValueError('strategy must be one of S1, S2, S3, S4')
    # S1 and S3 are generic nn.Module policies: older T03/T04 tests and
    # custom vision backbones have no HF-specific `vision_model` attribute.
    # S2 alone depends on the CLIP/SigLIP transformer module layout.
    if strategy == 'S1':
        backbone.requires_grad_(False)
        backbone.eval()
        return []
    if strategy in ('S3', 'S4'):
        backbone.requires_grad_(True)
        return []

    if not hasattr(backbone, 'vision_model'):
        raise TypeError('S2 requires a CLIP/SigLIP vision_model')
    vision = backbone.vision_model
    if not hasattr(vision, 'encoder') or not hasattr(vision.encoder, 'layers'):
        raise TypeError('S2 requires vision_model.encoder.layers')
    layers = vision.encoder.layers
    if len(layers) < 2:
        raise ValueError('S2 requires at least two transformer blocks')
    backbone.requires_grad_(False)
    for block in list(layers)[-2:]:
        block.requires_grad_(True)
    for name in ('post_layernorm', 'head'):
        module = getattr(vision, name, None)
        if module is not None:
            module.requires_grad_(True)
    return list(range(len(layers) - 2, len(layers)))



def trainable_parameter_counts(model):
    return {
        'total': sum(p.numel() for p in model.parameters()),
        'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'backbone_total': sum(p.numel() for p in model.backbone.parameters()),
        'backbone_trainable': sum(p.numel() for p in model.backbone.parameters() if p.requires_grad),
    }
