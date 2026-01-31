# model_torch.py

import torch
import torch.nn as nn
import timm

from timm.models.swin_transformer import SwinTransformer
from timm.models.vision_transformer import VisionTransformer


import argparse

class MultiHeadClassifier(nn.Module):
    def __init__(self, backbone='efficientnet', input_shape=(1, 32, 32), 
                 d_model=512, init_weights='imagenet'):
        super(MultiHeadClassifier, self).__init__()

        self.backbone_name = backbone.lower()

        # ----------------------
        # Shared Backbone (Flexible)
        # ----------------------

        if self.backbone_name == 'efficientnet':
            # EfficientNet: Modify input layer to accept 1 channel instead of 3
            self.backbone = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
            self.backbone.conv_stem = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.features = self.backbone
        elif self.backbone_name == 'convnext':
            print(f"[INFO] Using ConvNeXt backbone with input shape: {input_shape}")
            # ConvNeXt: Modify input layer to accept 1 channel instead of 3
            self.backbone = timm.create_model('convnext_small.in12k_ft_in1k', pretrained=False, in_chans=1, num_classes=0)
            # self.backbone.stem[0] = nn.Conv2d(1, 96, kernel_size=(4, 4), stride=(4, 4), padding=(1, 1), bias=False)  # Adjust the first conv layer
            # self.backbone.norm = nn.LayerNorm(96)  # Adjust normalization layer (optional, depending on the model)
            self.backbone.conv_stem = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            print(f"[INFO] ConvNeXt backbone initialized with input shape: {input_shape}")
            self.features = self.backbone
        elif self.backbone_name == 'swin':
            # 1) Build a Swin that takes 1×32×32 inputs directly:
            #    - patch_size divides 32, e.g. 4 → produces (32/4=8) patches per dim
            #    - window_size also divides 8, e.g. 4 → non‐overlapping windows
            self.backbone = SwinTransformer(
                img_size=32,
                patch_size=4,
                in_chans=1,
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=4,
                mlp_ratio=4.,
                qkv_bias=True,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.1,
                norm_layer=nn.LayerNorm,
                patch_norm=True,
                use_checkpoint=False,
                num_classes=0  # <— so it returns features, not 1000‐class logits
            )
            self.features=self.backbone
            # 2) The final feature dim is embed_dim * 2^(len(depths)-1)
            #    Since swin builds hierarchies, use its num_features attr:
            self.features.num_features = self.backbone.num_features
        elif self.backbone_name == 'vit':
            # ViT: Modify input layer to accept 1 channel instead of 3
            # self.backbone = timm.create_model('vit_base_patch16_224', pretrained=False, in_chans=1, num_classes=0)
            # self.backbone.patch_embed.proj = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16), bias=False)
            # self.features = self.backbone

            # 1) Build a ViT that takes 1×32×32 inputs directly:
            #    - img_size=32, patch_size=4 → (32/4=8)^2 = 64 patches
            #    - embed_dim small for speed, e.g. 192
            #    - depth=4 layers, num_heads=3 (must divide 192)
            self.backbone = VisionTransformer(
                img_size=32,
                patch_size=4,
                in_chans=1,
                embed_dim=192,
                depth=4,
                num_heads=3,
                mlp_ratio=4.0,
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.1,
                norm_layer=nn.LayerNorm,
                num_classes=0,        # returns features
            )

            self.features=self.backbone
            # 2) Feature dimension from ViT’s output:
            self.features.num_features = self.backbone.embed_dim  # 192
        elif self.backbone_name == 'mamba_vision':
            from models.model_mamba import MambaVisionMultiHead
              # Use the MambaVisionMultiHead as a self-contained backbone
            self.features = MambaVisionMultiHead(
                in_chans=input_shape[0],
                img_size=input_shape[1]
            )
        else:
            raise ValueError(f"Unsupported backbone model: {self.backbone_name}")

        # ----------------------
        # Output Heads
        # ----------------------
        # Output Heads for other backbones
        if self.backbone_name != 'mamba_vision':
            # For EfficientNet/ConvNeXt/SwinV2, the feature dimension is typically `num_features` or `hidden_size`
            self.energy_loss_head = nn.Linear(self.features.num_features, 1)  # Sigmoid for binary output
            self.alpha_head = nn.Linear(self.features.num_features, 3)  # Softmax for 3-class output
            self.q0_head = nn.Linear(self.features.num_features, 4)  # Softmax for 4-class output


    def forward(self, x):
        if self.backbone_name == 'mamba_vision':
            return self.features(x)
        
        feats = self.features(x)
        # if self.backbone_name == 'convnext_cifar' and x.shape[1] == 1:
        #     x = x.repeat(1, 3, 1, 1)
        #     feats = self.features(pixel_values=x) 
        return {
            'energy_loss_output': self.energy_loss_head(feats),
            'alpha_output': self.alpha_head(feats),
            'q0_output': self.q0_head(feats)
        }
# Helper: Weight Initialization

def weights_init_normal(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# from models.model_convnext import ConvNeXtClassifier
from models.model_vit import ViTClassifier
# from models.model_mamba import MambaClassifier
# ---------------------------
# Model Creation Helper
# ---------------------------
def create_model(backbone='efficientnet',
                input_shape=(1, 32, 32),
                learning_rate=0.001, 
                d_model=512,
                init_weights='imagenet'):# 'imagenet', 'cifar', or 'gaussian'
    """
    Helper function to create and compile a MultiHeadClassifier.
    """
    # if backbone == 'convnext_cifar':
    #     model = ConvNeXtCIFARClassifier(input_shape=input_shape,pretrained=pretrained)
    if backbone.startswith('convnext'):
        suffix = backbone[len('convnext_'):]  # get everything after 'convnext_'

        if suffix == 'fb_in22k_ft_in1k':
            model_name = 'convnext_tiny.fb_in22k_ft_in1k'
            pretrained = True
        elif suffix == 'fb_in1k':
            model_name = 'convnext_tiny.fb_in1k'
            pretrained = True
        elif suffix == 'gaussian':
            model_name = 'convnext_tiny'
            pretrained = False
        elif suffix == '':
            model_name = 'convnext_tiny'
            pretrained = False
        else:
            raise ValueError(f"Unrecognized convnext variant: '{suffix}'")
        
        print(f"Using model: {model_name}, pretrained: {pretrained}")


        model = ConvNeXtClassifier(input_shape=input_shape, pretrained=pretrained, model_name=model_name)

        if suffix == 'gaussian':
            model.apply(weights_init_normal)
    # Handle ViT Variants
    elif backbone.startswith('vit'):
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
    elif backbone.startswith('mambaout'):
        suffix = backbone[len('mambaout_'):]

        if suffix == 'base_plus_rw.sw_e150_in12k_ft_in1k':
            model_name = 'mambaout_base_plus_rw.sw_e150_in12k_ft_in1k'
            pretrained = True
        elif suffix == 'base_plus_rw.sw_e150_in12k':
            model_name = 'mambaout_base_plus_rw.sw_e150_in12k'
            pretrained = True
        elif suffix == 'base_plus_rw_gaussian':
            model_name = 'mambaout_base_plus_rw'
            pretrained = False
        elif suffix == 'base_plus_rw':
            model_name = 'mambaout_base_plus_rw'
            pretrained = False
        else:
            raise ValueError(f"Unrecognized mamba variant: '{suffix}'")

        print(f"Using Mamba model: {model_name}, pretrained: {pretrained}")
        model = MambaClassifier(input_shape=input_shape, pretrained=pretrained, model_name=model_name)
        if suffix == 'gaussian':
            model.apply(weights_init_normal)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        return model, optimizer
    else:
        model = MultiHeadClassifier(backbone=backbone, input_shape=input_shape,
                                d_model=d_model, init_weights=init_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return model, optimizer

# Example input to test the model
# python models/model_torch.py --backbone efficientnet
# python models/model_torch.py --backbone convnext
# python models/model_torch.py --backbone swin
# python models/model_torch.py --backbone mamba

# ---------------------------
# Main function to test the model
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Test different backbone models for ML-JET multi-head classifier.")
    parser.add_argument('--backbone', type=str, default='efficientnet',
                        help='Backbone model to test (options: efficientnet, convnext, swin, mamba)')
    parser.add_argument('--input_shape', type=int, nargs=3, default=[1, 32, 32],
                        help='Input shape as three integers (default: 1 32 32)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension (for Mamba)')  # Add d_model argument

    args = parser.parse_args()

    print(f"\n[INFO] Testing backbone: {args.backbone}")
    print(f"[INFO] Input shape: {args.input_shape}")
    print(f"[INFO] Learning rate: {args.learning_rate}")
    print(f"[INFO] d_model: {args.d_model}")

    # Create the model with specified parameters
    model, _ = create_model(backbone=args.backbone,
                                     input_shape=tuple(args.input_shape),
                                     learning_rate=args.learning_rate,
                                    d_model=args.d_model)
    # Display the model summary
    print("\n[INFO] Model Summary:")
    print(model)
    

    # # Example random input for testing
    x_gray = torch.randn(16, 1, 32, 32)
    # 1-channel grayscale batch
    print(f"\n[INFO] Model output for input shape {x_gray.shape}:")
    out = model(x_gray)

    print(out)  # Should be a dict with logits

    
    # for key, value in outputs.items():
    #     print(f"{key}: {value.shape}")

if __name__ == "__main__":
    main()
