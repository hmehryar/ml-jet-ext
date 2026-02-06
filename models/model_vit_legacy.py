import torch
import torch.nn as nn
import timm
# from mamba_ssm import Mamba  # Assuming `mamba-ssm` is installed for Mamba model
import torch.nn.functional as F
class ViTClassifier(nn.Module):
    def __init__(self, input_shape=(1, 32, 32), pretrained=True, model_name='vit_base_patch16_224'):
        super().__init__()
        self.backbone_name = 'vit'

        # Load ViT backbone from timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=3,  # Use 3 channels, replicate grayscale input if needed
            num_classes=0
        )
        self.features = self.backbone
        self.features.num_features = self.backbone.num_features  # Get embedding dim

        # Multi-task heads
        self.energy_loss_head = nn.Linear(self.features.num_features, 1)
        self.alpha_head = nn.Linear(self.features.num_features, 3)
        self.q0_head = nn.Linear(self.features.num_features, 4)

    def forward(self, x):
        # If input is grayscale (1 channel), replicate to 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        # Resize: (B, 3, 32, 32) → (B, 3, 224, 224)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        feats = self.features(x)  # Shape: (B, C)
        return {
            'energy_loss_output': self.energy_loss_head(feats),
            'alpha_output': self.alpha_head(feats),
            'q0_output': self.q0_head(feats),
        }
    
from models.model import weights_init_normal
def create_model(backbone="vit_gaussian", 
                 input_shape=(1, 32, 32),
                 learning_rate=0.0001,
                 ):

    if backbone.startswith('vit'):
        suffix = backbone[len('vit_'):]
        if suffix == 'augreg_in21k_ft_in1k':
            model_name = 'vit_tiny_patch16_224.augreg_in21k_ft_in1k'
            pretrained = True
        elif suffix == 'augreg_in21k':
            model_name = 'vit_tiny_patch16_224.augreg_in21k'
            pretrained = True
        elif suffix == 'gaussian':
            model_name = 'vit_tiny_patch16_224'
            pretrained = False
        elif suffix == '':
            model_name = 'vit_tiny_patch16_224'
            pretrained = False
        else:
            raise ValueError(f"Unrecognized vit variant: '{suffix}'")

        print(f"Using ViT model: {model_name}, pretrained: {pretrained}")
        model = ViTClassifier(input_shape=input_shape, pretrained=pretrained, model_name=model_name)
        if suffix == 'gaussian':
            model.apply(weights_init_normal)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        return model, optimizer