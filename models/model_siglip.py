"""Pretrained SigLIP ViT-B/16 jet-moment classifier."""
from models.moment_vlm import MomentVisionClassifier, build_optimizer

DEFAULT_CHECKPOINT = "google/siglip-base-patch16-224"


class SigLIPClassifier(MomentVisionClassifier):
    def __init__(self, input_shape=(5, 32, 32), checkpoint=DEFAULT_CHECKPOINT,
                 freeze_backbone=True, pretrained=True):
        from transformers import SiglipVisionConfig, SiglipVisionModel
        backbone = (SiglipVisionModel.from_pretrained(checkpoint) if pretrained
                    else SiglipVisionModel(SiglipVisionConfig()))
        super().__init__(backbone, input_shape=input_shape,
                         freeze_backbone=freeze_backbone)
        self.backbone_name = "siglip"
        self.checkpoint_name = checkpoint


def create_model(backbone="siglip_vit_b16", input_shape=(5, 32, 32),
                 learning_rate=1e-4, dataset_type="moment",
                 freeze_backbone=True, pretrained=True):
    if backbone != "siglip_vit_b16":
        raise ValueError(f"Unsupported SigLIP backbone: {backbone}")
    if dataset_type != "moment":
        raise ValueError("SigLIP model supports moment dataset only")
    model = SigLIPClassifier(input_shape=input_shape,
                             freeze_backbone=freeze_backbone, pretrained=pretrained)
    return model, build_optimizer(model, learning_rate)
