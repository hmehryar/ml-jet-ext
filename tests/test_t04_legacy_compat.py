"""T04-03 regression: legacy vision module does not use HF internals."""
from types import SimpleNamespace
import torch
from torch import nn
from models.moment_vlm import MomentVisionClassifier

class LegacyVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.encoder = nn.Conv2d(3, 8, 1)
    def forward(self, pixel_values):
        return SimpleNamespace(pooler_output=self.encoder(pixel_values).mean((2, 3)))

def test_legacy_s1_and_s3():
    for frozen in (True, False):
        model = MomentVisionClassifier(LegacyVision(), (1, 32, 32), freeze_backbone=frozen)
        model.train()
        assert model.backbone.training is (not frozen)
        assert all(p.requires_grad != frozen for p in model.backbone.parameters())
        outputs = model(torch.randn(2, 1, 32, 32))
        assert outputs['alpha_output'].shape == (2, 7)
        sum(x.sum() for x in outputs.values()).backward()
        assert model.input_adapter.weight.grad is not None

def test_legacy_s2_rejected():
    import pytest
    with pytest.raises(TypeError, match='S2 requires'):
        MomentVisionClassifier(LegacyVision(), strategy='S2')
