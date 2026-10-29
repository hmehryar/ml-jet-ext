"""Pretrained CLIP ViT-B/16 jet-moment classifier."""
from models.moment_vlm import MomentVisionClassifier, build_optimizer

DEFAULT_CHECKPOINT = "openai/clip-vit-base-patch16"


class CLIPClassifier(MomentVisionClassifier):
    def __init__(self, input_shape=(5, 32, 32), checkpoint=DEFAULT_CHECKPOINT,
                 freeze_backbone=True, pretrained=True):
        from transformers import CLIPVisionConfig, CLIPVisionModel
        backbone = (CLIPVisionModel.from_pretrained(checkpoint, use_safetensors=True) if pretrained
                    else CLIPVisionModel(CLIPVisionConfig()))
        super().__init__(backbone, input_shape=input_shape,
                         freeze_backbone=freeze_backbone)
        self.backbone_name = "clip"
        self.checkpoint_name = checkpoint


def create_model(backbone="clip_vit_b16", input_shape=(5, 32, 32),
                 learning_rate=1e-4, dataset_type="moment",
                 freeze_backbone=True, pretrained=True):
    if backbone != "clip_vit_b16":
        raise ValueError(f"Unsupported CLIP backbone: {backbone}")
    if dataset_type != "moment":
        raise ValueError("CLIP model supports moment dataset only")
    model = CLIPClassifier(input_shape=input_shape,
                           freeze_backbone=freeze_backbone, pretrained=pretrained)
    return model, build_optimizer(model, learning_rate)
