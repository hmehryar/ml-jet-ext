"""Pretrained CLIP ViT-B/16 jet-moment classifier."""
from models.moment_vlm import MomentVisionClassifier, build_optimizer

DEFAULT_CHECKPOINT = 'openai/clip-vit-base-patch16'

class CLIPClassifier(MomentVisionClassifier):
    def __init__(self, input_shape=(5, 32, 32), checkpoint=DEFAULT_CHECKPOINT,
                 revision=None, freeze_backbone=True, pretrained=True):
        from transformers import CLIPVisionConfig, CLIPVisionModel
        if pretrained:
            kw = {'use_safetensors': True}
            if revision is not None:
                kw['revision'] = revision
            vision = CLIPVisionModel.from_pretrained(checkpoint, **kw)
        else:
            vision = CLIPVisionModel(CLIPVisionConfig())
        super().__init__(vision, input_shape=input_shape, freeze_backbone=freeze_backbone)
        self.backbone_name = 'clip'
        self.checkpoint_name = checkpoint
        self.checkpoint_revision = revision


def create_model(backbone='clip_vit_b16', input_shape=(5, 32, 32), learning_rate=1e-4,
                 dataset_type='moment', freeze_backbone=True, pretrained=True,
                 checkpoint=DEFAULT_CHECKPOINT, revision=None):
    if backbone != 'clip_vit_b16' or dataset_type != 'moment':
        raise ValueError('Unsupported CLIP backbone or dataset type')
    model = CLIPClassifier(input_shape=input_shape, checkpoint=checkpoint,
                           revision=revision, freeze_backbone=freeze_backbone,
                           pretrained=pretrained)
    return model, build_optimizer(model, learning_rate)
