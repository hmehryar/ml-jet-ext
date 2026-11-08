"""T04-10 single-task jet moment classifiers.

Additive implementation: the existing T03 and T04 dual-head classes are not
imported or modified. Each instance has precisely one classification head.

CLIP/SigLIP use the same 1x1 adapter, interpolation, and S1-S4 policies as
MomentVisionClassifier. ViT-Tiny support is provisional until the separately
planned cross-transformer S1-S4 parity benchmark is complete.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

TASK_CLASSES = {"alpha_only": 7, "q0_only": 5}
TASK_OUTPUT = {"alpha_only": "alpha_output", "q0_only": "q0_output"}


def validate_task(task: str) -> str:
    if task not in TASK_CLASSES:
        raise ValueError("T04-10 task must be alpha_only or q0_only")
    return task


class SingleHeadMomentVisionClassifier(nn.Module):
    """One trainable head over a CLIP or SigLIP vision encoder.

    `backbone` is a vision model whose forward(pixel_values=...) returns
    pooler_output and whose config includes hidden_size.
    """

    def __init__(self, backbone: nn.Module, *, task: str, channels: int = 5,
                 strategy: str = "S1", image_size: int = 224):
        super().__init__()
        self.task = validate_task(task)
        if not isinstance(channels, int) or isinstance(channels, bool) or not 1 <= channels <= 5:
            raise ValueError("channels must be an integer between 1 and 5")
        if image_size != 224:
            raise ValueError("CLIP/SigLIP encoder input must be 224x224")
        if strategy not in ("S1", "S2", "S3", "S4"):
            raise ValueError("strategy must be S1, S2, S3, or S4")
        self.dataset_type = "moment"
        self.input_channels = channels
        self.image_size = image_size
        self.strategy = strategy
        self.backbone = backbone
        self.input_adapter = nn.Conv2d(channels, 3, kernel_size=1)
        # This intentionally instantiates NO other head.
        self.classification_head = nn.Linear(backbone.config.hidden_size, TASK_CLASSES[task])
        from models.t04_strategies import apply_strategy
        self.unfrozen_transformer_indices = apply_strategy(self.backbone, strategy)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.strategy == "S1":
            self.backbone.eval()
        elif self.strategy == "S2":
            self.backbone.eval()
            if mode:
                vision = self.backbone.vision_model
                for block in list(vision.encoder.layers)[-2:]:
                    block.train(True)
                for name in ("post_layernorm", "head"):
                    layer = getattr(vision, name, None)
                    if layer is not None:
                        layer.train(True)
        return self

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != self.input_channels or tuple(x.shape[-2:]) != (32, 32):
            raise ValueError(f"Expected [B,{self.input_channels},32,32], got {tuple(x.shape)}")
        x = self.input_adapter(x)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        features = self.backbone(pixel_values=x).pooler_output
        if features is None:
            raise RuntimeError("Vision backbone did not produce pooler_output")
        return {TASK_OUTPUT[self.task]: self.classification_head(features)}


class SingleHeadViTClassifier(nn.Module):
    """Single-task ViT-Tiny input path (provisional T03-style architecture).

    A caller supplies an initialized timm ViT feature extractor. The historical
    JetBench transfer weights and normalization must be independently audited
    before using this variant for inter-architecture claims.
    """

    def __init__(self, backbone: nn.Module, *, task: str, channels: int = 5,
                 use_adapter: bool = True):
        super().__init__()
        self.task = validate_task(task)
        if not 1 <= channels <= 5:
            raise ValueError("channels must be 1..5")
        self.input_channels = channels
        self.dataset_type = "moment"
        self.backbone = backbone
        self.input_adapter = nn.Conv2d(channels, 3, 1, bias=False) if use_adapter else nn.Identity()
        if not use_adapter and channels != 3:
            raise ValueError("Without adapter, ViT must receive exactly 3 channels")
        hidden_size = getattr(backbone, "num_features", None)
        if not isinstance(hidden_size, int) or hidden_size < 1:
            raise ValueError("ViT backbone must expose num_features")
        self.classification_head = nn.Linear(hidden_size, TASK_CLASSES[task])

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != self.input_channels or tuple(x.shape[-2:]) != (32, 32):
            raise ValueError(f"Expected [B,{self.input_channels},32,32], got {tuple(x.shape)}")
        x = F.interpolate(self.input_adapter(x), (224, 224), mode="bilinear", align_corners=False)
        return {TASK_OUTPUT[self.task]: self.classification_head(self.backbone(x))}


def build_clip_siglip_single_head(*, architecture: str, task: str, channels: int,
                                  strategy: str, checkpoint: str,
                                  revision: str | None, pretrained: bool = True):
    """Lazy HF load, matching T04 backbone configuration and strategy semantics."""
    validate_task(task)
    expected = {"clip": "openai/clip-vit-base-patch16",
                "siglip": "google/siglip-base-patch16-224"}
    if architecture not in expected:
        raise ValueError("architecture must be clip or siglip")
    if (strategy == "S4") != (not pretrained):
        raise ValueError("S4 must be scratch; S1-S3 must use pretrained weights")
    from transformers import (AutoConfig, CLIPVisionModel, SiglipVisionModel)
    model_cls = CLIPVisionModel if architecture == "clip" else SiglipVisionModel
    kw = {"revision": revision} if revision is not None else {}
    if pretrained:
        backbone = model_cls.from_pretrained(checkpoint, use_safetensors=True, **kw)
    else:
        config = AutoConfig.from_pretrained(checkpoint, **kw)
        # HF CLIP/SigLIP configs nest the vision model config.
        backbone = model_cls(config.vision_config)
    model = SingleHeadMomentVisionClassifier(
        backbone, task=task, channels=channels, strategy=strategy)
    model.checkpoint_name = checkpoint
    model.checkpoint_revision = revision
    model.pretrained_backbone = pretrained
    model.architecture = architecture
    model.backbone_name = architecture
    return model


def build_trainable_optimizer(model: nn.Module, *, lr_backbone: float = 1e-5,
                              lr_adapter: float = 1e-4, lr_head: float = 1e-4,
                              weight_decay: float = 0.01):
    """Build AdamW with disjoint, task-only parameter groups."""
    groups = []
    for name, module, lr in (("backbone", model.backbone, lr_backbone),
                             ("adapter", model.input_adapter, lr_adapter),
                             ("heads", model.classification_head, lr_head)):
        if lr <= 0:
            raise ValueError(f"Invalid learning rate for {name}")
        params = [p for p in module.parameters() if p.requires_grad]
        if params:
            groups.append({"params": params, "lr": float(lr), "group_name": name})
    if not groups:
        raise RuntimeError("No trainable parameters")
    optimizer = torch.optim.AdamW(groups, weight_decay=weight_decay)
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise RuntimeError("Optimizer does not match trainable parameters")
    return optimizer
