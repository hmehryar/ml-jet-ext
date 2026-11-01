from types import SimpleNamespace
import pytest
import torch
from torch import nn
from models.moment_vlm import MomentVisionClassifier, build_optimizer
from models.t04_strategies import trainable_parameter_counts

class FakeVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.vision_model = nn.Module()
        self.vision_model.encoder = nn.Module()
        self.vision_model.encoder.layers = nn.ModuleList([nn.Linear(3, 3) for _ in range(4)])
        self.vision_model.post_layernorm = nn.LayerNorm(3)
        self.vision_model.head = nn.Linear(3, 4)

    def forward(self, pixel_values):
        z = pixel_values.mean(dim=(2, 3))
        for block in self.vision_model.encoder.layers:
            z = torch.tanh(block(z))
        z = self.vision_model.post_layernorm(z)
        return SimpleNamespace(pooler_output=self.vision_model.head(z))

@pytest.mark.parametrize('strategy', ('S1','S2','S3','S4'))
@pytest.mark.parametrize('channels', (1,2,3,4,5))
def test_masks_and_gradients(strategy, channels):
    model = MomentVisionClassifier(FakeVision(), input_shape=(channels,32,32), strategy=strategy)
    model.train()
    layers = model.backbone.vision_model.encoder.layers
    active = [any(p.requires_grad for p in layer.parameters()) for layer in layers]
    assert active == {'S1':[False]*4, 'S2':[False,False,True,True], 'S3':[True]*4, 'S4':[True]*4}[strategy]
    assert all(p.requires_grad for p in model.input_adapter.parameters())
    assert all(p.requires_grad for p in model.alpha_head.parameters())
    if strategy == 'S1':
        assert not model.backbone.training
    if strategy == 'S2':
        assert not model.backbone.training and layers[-1].training and not layers[0].training
    optimizer = build_optimizer(model, 1e-4)
    params = {id(p) for g in optimizer.param_groups for p in g['params']}
    assert params == {id(p) for p in model.parameters() if p.requires_grad}
    out = model(torch.randn(2,channels,32,32))
    assert out['alpha_output'].shape == (2,7)
    assert out['q0_output'].shape == (2,5)
    (out['alpha_output'].sum() + out['q0_output'].sum()).backward()
    assert model.input_adapter.weight.grad is not None
    for i, layer in enumerate(layers):
        assert (layer.weight.grad is not None) == active[i]
    counts=trainable_parameter_counts(model)
    assert counts['total'] >= counts['trainable'] > 0
    assert (counts['backbone_trainable']==0) == (strategy=='S1')

def test_invalid_strategy():
    with pytest.raises(ValueError):
        MomentVisionClassifier(FakeVision(), strategy='S5')
