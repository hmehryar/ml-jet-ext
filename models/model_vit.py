"""timm ViT model with opt-in M1--M5 input and two-label output.

Legacy is the default: three-channel backbone with energy-loss, alpha and q0
heads, retaining the original grayscale replication and 224x224 interpolation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ViTClassifier(nn.Module):
    def __init__(self, input_shape=(1, 32, 32), pretrained=True,
                 model_name='vit_base_patch16_224', dataset_type='legacy'):
        super().__init__()
        if dataset_type not in ('legacy', 'moment'):
            raise ValueError(f"Unsupported dataset_type: {dataset_type!r}")
        if len(input_shape) != 3 or not 1 <= int(input_shape[0]) <= 5:
            raise ValueError(f"Invalid input_shape: {input_shape!r}")

        self.backbone_name = 'vit'
        self.dataset_type = dataset_type
        self.input_channels = int(input_shape[0]) if dataset_type == 'moment' else 3
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained,
            in_chans=self.input_channels, num_classes=0,
        )
        self.features = self.backbone
        feature_dim = self.backbone.num_features

        if dataset_type == 'legacy':
            self.energy_loss_head = nn.Linear(feature_dim, 1)
        self.alpha_head = nn.Linear(feature_dim, 7)
        self.q0_head = nn.Linear(feature_dim, 5)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
        if self.dataset_type == 'moment' and x.shape[1] != self.input_channels:
            raise ValueError(f"Moment model expects {self.input_channels} channels, got {x.shape[1]}")
        if self.dataset_type == 'legacy':
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            elif x.shape[1] != 3:
                raise ValueError(f"Legacy ViT expects 1 or 3 channels, got {x.shape[1]}")
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        feats = self.features(x)
        output = {
            'alpha_output': self.alpha_head(feats),
            'q0_output': self.q0_head(feats),
        }
        if self.dataset_type == 'legacy':
            output = {'energy_loss_output': self.energy_loss_head(feats), **output}
        return output


def create_model(backbone='vit_gaussian', input_shape=(1, 32, 32),
                 learning_rate=0.0001, dataset_type='legacy'):
    """Return (model, optimizer), maintaining the original positional API."""
    if not backbone.startswith('vit'):
        raise ValueError(f"Unsupported backbone for model_vit: {backbone!r}")
    suffix = backbone[len('vit_'):]
    if suffix == 'augreg_in21k_ft_in1k':
        model_name, pretrained = 'vit_tiny_patch16_224.augreg_in21k_ft_in1k', True
    elif suffix == 'augreg_in21k':
        model_name, pretrained = 'vit_tiny_patch16_224.augreg_in21k', True
    elif suffix in ('gaussian', ''):
        model_name, pretrained = 'vit_tiny_patch16_224', False
    else:
        raise ValueError(f"Unrecognized vit variant: {suffix!r}")

    print(f"Using ViT model: {model_name}, pretrained: {pretrained}, dataset_type: {dataset_type}")
    model = ViTClassifier(
        input_shape=input_shape, pretrained=pretrained,
        model_name=model_name, dataset_type=dataset_type,
    )
    if suffix == 'gaussian':
        from models.model import weights_init_normal
        model.apply(weights_init_normal)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, optimizer
