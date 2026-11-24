"""Shared CLIP/SigLIP input adapter and two-head moment classifier."""
import torch
from torch import nn
from torch.nn import functional as F
from models.t04_strategies import apply_strategy


class MomentVisionClassifier(nn.Module):
    """Adapt [B,k,32,32] jet moments to a 224x224 vision encoder."""

    def __init__(self, backbone, input_shape=(5, 32, 32), freeze_backbone=True,
                 image_size=224, strategy=None):
        super().__init__()
        if (len(input_shape) != 3 or type(input_shape[0]) is not int
                or not 1 <= input_shape[0] <= 5 or tuple(input_shape[1:]) != (32, 32)):
            raise ValueError('input_shape must be (channels 1..5, 32, 32)')
        if image_size != 224:
            raise ValueError('Selected encoders expect 224x224 inputs')
        if strategy is None:
            strategy = 'S1' if freeze_backbone else 'S3'
        self.dataset_type = 'moment'
        self.input_channels = input_shape[0]
        self.image_size = image_size
        self.backbone = backbone
        self.input_adapter = nn.Conv2d(self.input_channels, 3, kernel_size=1)
        hidden_size = backbone.config.hidden_size
        self.alpha_head = nn.Linear(hidden_size, 7)
        self.q0_head = nn.Linear(hidden_size, 5)
        self.strategy = strategy
        self.freeze_backbone = strategy == 'S1'
        self.unfrozen_transformer_indices = apply_strategy(self.backbone, strategy)

    def train(self, mode=True):
        super().train(mode)
        if self.strategy == 'S1':
            self.backbone.eval()
        elif self.strategy == 'S2':
            # Keep frozen backbone components in eval, allowing the last two
            # blocks and final pooling/norm modules to use train semantics.
            self.backbone.eval()
            if mode:
                for block in list(self.backbone.vision_model.encoder.layers)[-2:]:
                    block.train(True)
                for name in ('post_layernorm', 'head'):
                    module = getattr(self.backbone.vision_model, name, None)
                    if module is not None:
                        module.train(True)
        return self

    def forward(self, x):
        if (x.ndim != 4 or x.shape[1] != self.input_channels
                or tuple(x.shape[-2:]) != (32, 32)):
            raise ValueError(f'Expected [B,{self.input_channels},32,32], got {tuple(x.shape)}')
        x = self.input_adapter(x)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        # Do not use no_grad: frozen-backbone gradients must reach adapter.
        features = self.backbone(pixel_values=x).pooler_output
        if features is None:
            raise RuntimeError('Vision backbone did not return pooled features')
        return {'alpha_output': self.alpha_head(features),
                'q0_output': self.q0_head(features)}


def build_optimizer(model, learning_rate):
    if learning_rate <= 0:
        raise ValueError('learning_rate must be positive')
    return torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
