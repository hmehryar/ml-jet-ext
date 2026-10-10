"""Fast model-interface tests using a minimal timm backbone stub."""
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn


class TinyBackbone(nn.Module):
    num_features = 8

    def __init__(self, in_chans):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, 8, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.pool(self.conv(x)).flatten(1)


@pytest.fixture
def vit(monkeypatch):
    fake_timm = types.ModuleType('timm')
    fake_timm.create_model = lambda model_name, pretrained, in_chans, num_classes: TinyBackbone(in_chans)
    monkeypatch.setitem(sys.modules, 'timm', fake_timm)
    path = Path(__file__).resolve().parents[1] / 'models' / 'model_vit.py'
    spec = importlib.util.spec_from_file_location('isolated_model_vit', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize('channels', [1, 2, 5])
def test_moment_forward_and_backward(vit, channels):
    model = vit.ViTClassifier(input_shape=(channels, 32, 32), pretrained=False, dataset_type='moment')
    x = torch.randn(2, channels, 32, 32)
    outputs = model(x)
    assert set(outputs) == {'alpha_output', 'q0_output'}
    assert outputs['alpha_output'].shape == (2, 7)
    assert outputs['q0_output'].shape == (2, 5)
    loss = nn.CrossEntropyLoss()(outputs['alpha_output'], torch.tensor([0, 6]))
    loss += nn.CrossEntropyLoss()(outputs['q0_output'], torch.tensor([1, 4]))
    loss.backward()
    assert model.backbone.conv.weight.grad is not None


def test_legacy_three_heads_and_grayscale_compat(vit):
    model = vit.ViTClassifier(input_shape=(1, 32, 32), pretrained=False)
    assert model.backbone.conv.in_channels == 3
    out = model(torch.randn(2, 1, 32, 32))
    assert set(out) == {'energy_loss_output', 'alpha_output', 'q0_output'}
    assert out['energy_loss_output'].shape == (2, 1)
    assert out['alpha_output'].shape == (2, 7)
    assert out['q0_output'].shape == (2, 5)


def test_invalid_moment_channels(vit):
    model = vit.ViTClassifier(input_shape=(2, 32, 32), pretrained=False, dataset_type='moment')
    with pytest.raises(ValueError):
        model(torch.randn(1, 3, 32, 32))
