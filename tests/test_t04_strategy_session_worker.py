"""T04-03 stage 2: strategy selection before optimizer construction."""
import pytest
import torch
from torch import nn
from train_utils.t04_session_worker import validate_strategy_settings, build_strategy_model


def cfg(strategy):
    return {'strategy': strategy, 'model': 'clip', 'moments': [1, 2, 3],
            'precision': 'fp32', 'gradient_accumulation': 1,
            'moment_loss': {'type': 'cross_entropy'},
            'learning_rates': {'adapter': .0001, 'heads': .0001, 'backbone': .0001},
            'checkpoint': 'test/checkpoint', 'checkpoint_revision': 'f' * 40}


@pytest.mark.parametrize('strategy', ('S1', 'S2', 'S3', 'S4'))
def test_build_strategy_factory_and_optimizer(strategy):
    seen = {}
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.p1 = nn.Parameter(torch.ones(2))
            self.p2 = nn.Parameter(torch.ones(2), requires_grad=strategy != 'S1')
            self.strategy = strategy
            self.pretrained_backbone = strategy != 'S4'
    def factory(**kwargs):
        seen.update(kwargs)
        model = Model()
        return model, torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=kwargs['learning_rate'])
    model, optimizer = build_strategy_model(cfg(strategy), {'clip': (factory, 'clip_vit_b16')})
    assert seen['strategy'] == strategy
    assert seen['pretrained'] == (strategy != 'S4')
    assert seen['freeze_backbone'] == (strategy == 'S1')
    assert seen['revision'] == 'f' * 40
    assert sum(p.numel() for group in optimizer.param_groups for p in group['params']) == (2 if strategy == 'S1' else 4)


@pytest.mark.parametrize('change', [
    {'strategy': 'S9'},
    {'precision': 'invalid'},
    {'gradient_accumulation': 0},
    {'gradient_accumulation': -1},
    {'moment_loss': {'type': 'unknown_loss'}},
    {'learning_rates': {'adapter': .0001, 'heads': .0002, 'backbone': .0002}},
])
def test_reject_unsupported_before_model_load(change):
    c = cfg('S2')
    c.update(change)
    with pytest.raises((ValueError, NotImplementedError)):
        validate_strategy_settings(c)


def test_reject_mismatched_optimizer_mask():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.ones(2))
            self.q = nn.Parameter(torch.ones(2))
            self.strategy = 'S3'
            self.pretrained_backbone = True
    def bad_factory(**kwargs):
        m = Model()
        return m, torch.optim.Adam([m.p], lr=.0001)
    with pytest.raises(RuntimeError, match='Optimizer parameters'):
        build_strategy_model(cfg('S3'), {'clip': (bad_factory, 'clip_vit_b16')})

@pytest.mark.parametrize('precision', ['fp32', 'bf16', 'fp16'])
@pytest.mark.parametrize('accumulation', [1, 2, 4])
def test_accept_supported_precision_and_accumulation(precision, accumulation):
    c = cfg('S2')
    c['precision'] = precision
    c['gradient_accumulation'] = accumulation

    assert validate_strategy_settings(c) == 'S2'
