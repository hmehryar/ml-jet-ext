"""CPU-only shared-adapter tests; no pretrained checkpoint download."""
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from models.moment_vlm import MomentVisionClassifier, build_optimizer


class FakeVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.encoder = nn.Conv2d(3, 8, 1)

    def forward(self, pixel_values):
        return SimpleNamespace(pooler_output=self.encoder(pixel_values).mean((2, 3)))


@pytest.mark.parametrize("channels", range(1, 6))
@pytest.mark.parametrize("frozen", [True, False])
def test_shapes_and_gradients(channels, frozen):
    model = MomentVisionClassifier(FakeVision(), (channels, 32, 32),
                                   freeze_backbone=frozen)
    model.train()
    assert model.backbone.training is (not frozen)
    outputs = model(torch.randn(2, channels, 32, 32))
    assert set(outputs) == {"alpha_output", "q0_output"}
    assert outputs["alpha_output"].shape == (2, 7)
    assert outputs["q0_output"].shape == (2, 5)
    sum(v.sum() for v in outputs.values()).backward()
    assert model.input_adapter.weight.grad is not None
    assert model.alpha_head.weight.grad is not None
    assert model.q0_head.weight.grad is not None
    assert (model.backbone.encoder.weight.grad is None) is frozen
    optim = build_optimizer(model, 1e-4)
    assert all(p.requires_grad for group in optim.param_groups for p in group["params"])


def test_rejects_invalid_inputs():
    model = MomentVisionClassifier(FakeVision(), (5, 32, 32))
    with pytest.raises(ValueError):
        model(torch.randn(2, 4, 32, 32))
    with pytest.raises(ValueError):
        model(torch.randn(2, 5, 224, 224))
    with pytest.raises(ValueError):
        MomentVisionClassifier(FakeVision(), (6, 32, 32))
