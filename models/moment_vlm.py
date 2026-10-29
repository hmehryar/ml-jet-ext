"""Shared CLIP/SigLIP input adapter and two-head moment classifier."""
import torch
from torch import nn
from torch.nn import functional as F


class MomentVisionClassifier(nn.Module):
    """Adapt [B,k,32,32] jet moments to an RGB pretrained vision encoder."""

    def __init__(self, backbone, input_shape=(5, 32, 32), freeze_backbone=True,
                 image_size=224):
        super().__init__()
        if (len(input_shape) != 3 or type(input_shape[0]) is not int
                or not 1 <= input_shape[0] <= 5 or tuple(input_shape[1:]) != (32, 32)):
            raise ValueError("input_shape must be (channels 1..5, 32, 32)")
        if image_size != 224:
            raise ValueError("The selected pretrained checkpoints expect 224x224 inputs")
        self.dataset_type = "moment"
        self.input_channels = input_shape[0]
        self.image_size = image_size
        self.backbone = backbone
        self.input_adapter = nn.Conv2d(self.input_channels, 3, kernel_size=1)
        hidden_size = backbone.config.hidden_size
        self.alpha_head = nn.Linear(hidden_size, 7)
        self.q0_head = nn.Linear(hidden_size, 5)
        self.freeze_backbone = bool(freeze_backbone)
        if self.freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, x):
        if (x.ndim != 4 or x.shape[1] != self.input_channels
                or tuple(x.shape[-2:]) != (32, 32)):
            raise ValueError(f"Expected [B,{self.input_channels},32,32], got {tuple(x.shape)}")
        x = self.input_adapter(x)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        # Never use torch.no_grad(): encoder gradients must reach the adapter.
        features = self.backbone(pixel_values=x).pooler_output
        if features is None:
            raise RuntimeError("Vision backbone did not return pooled features")
        return {"alpha_output": self.alpha_head(features),
                "q0_output": self.q0_head(features)}


def build_optimizer(model, learning_rate):
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    return torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                            lr=learning_rate)
