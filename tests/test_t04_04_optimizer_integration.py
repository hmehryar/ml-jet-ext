"""T04-04 opt-in optimizer integration; old T04-03 runs remain supported."""
import copy
import pytest
import torch
from torch import nn
from train_utils.t04_session_worker import build_strategy_model, validate_strategy_settings


class MiniModel(nn.Module):
    def __init__(self, strategy):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.input_adapter = nn.Linear(2, 2)
        self.alpha_head = nn.Linear(2, 7)
        self.q0_head = nn.Linear(2, 5)
        self.strategy = strategy
        self.pretrained_backbone = strategy != 'S4'
        if strategy == 'S1':
            self.backbone.requires_grad_(False)


def cfg(strategy='S2', optimizer=None):
    config = {'strategy': strategy, 'model': 'clip', 'moments': [1, 2, 3],
              'precision': 'fp32', 'gradient_accumulation': 1,
              'moment_loss': {'type': 'cross_entropy'},
              'learning_rates': {'backbone': 1e-5, 'adapter': 1e-4, 'heads': 2e-4},
              'checkpoint': 'test/clip', 'checkpoint_revision': 'a' * 40}
    if optimizer is not None:
        config['optimizer'] = optimizer
    return config


def factory(**kwargs):
    m = MiniModel(kwargs['strategy'])
    return m, torch.optim.Adam((p for p in m.parameters() if p.requires_grad), lr=kwargs['learning_rate'])


@pytest.mark.parametrize('strategy', ['S1', 'S2', 'S3', 'S4'])
def test_opt_in_adamw_groups(strategy):
    m, opt = build_strategy_model(cfg(strategy, {'type':'adamw','weight_decay':.01}), {'clip': (factory, 'clip')})
    assert isinstance(opt, torch.optim.AdamW)
    names = [g['name'] for g in opt.param_groups]
    assert names == (['adapter','heads'] if strategy == 'S1' else ['backbone','adapter','heads'])
    assert {g['name']:g['lr'] for g in opt.param_groups} == {k:cfg()['learning_rates'][k] for k in names}
    assert all(g['weight_decay'] == .01 for g in opt.param_groups)
    assert {id(p) for g in opt.param_groups for p in g['params']} == {id(p) for p in m.parameters() if p.requires_grad}


def test_legacy_single_rate_optimizer_unchanged():
    c = cfg('S2')
    c['learning_rates'] = dict.fromkeys(('backbone','adapter','heads'), 1e-4)
    m, opt = build_strategy_model(c, {'clip': (factory, 'clip')})
    assert type(opt) is torch.optim.Adam
    assert len(opt.param_groups) == 1
    assert 'name' not in opt.param_groups[0]


def test_opt_in_adam_has_named_groups_and_no_decay():
    m, opt = build_strategy_model(cfg('S2', {'type':'adam','weight_decay':0}), {'clip': (factory, 'clip')})
    assert type(opt) is torch.optim.Adam
    assert [g['name'] for g in opt.param_groups] == ['backbone','adapter','heads']


def test_unequal_legacy_rates_still_rejected():
    with pytest.raises(NotImplementedError):
        validate_strategy_settings(cfg())


def test_named_optimizer_checkpoint_restores_state_and_lrs():
    c = cfg('S2', {'type':'adamw','weight_decay':.01})
    model, opt = build_strategy_model(c, {'clip': (factory, 'clip')})
    x = torch.ones(2, 2)
    loss = model.alpha_head(model.input_adapter(model.backbone(x))).sum() + model.q0_head(model.input_adapter(model.backbone(x))).sum()
    loss.backward(); opt.step()
    state = copy.deepcopy(opt.state_dict())
    restored, opt2 = build_strategy_model(c, {'clip': (factory, 'clip')})
    opt2.load_state_dict(state)
    assert [g['name'] for g in opt2.param_groups] == ['backbone','adapter','heads']
    assert [g['lr'] for g in opt2.param_groups] == [1e-5,1e-4,2e-4]
    assert len(opt2.state) == len(opt.state)
