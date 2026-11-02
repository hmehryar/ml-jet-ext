"""T04-04 Stage 1: independent optimizer tests without Hugging Face downloads."""
import pytest
import torch
from torch import nn
from train_utils.t04_optimizers import build_t04_optimizer, parameter_groups, learning_rate_summary


class TinyModel(nn.Module):
    def __init__(self, frozen=False):
        super().__init__()
        self.backbone = nn.Linear(3, 4)
        self.input_adapter = nn.Linear(3, 3)
        self.alpha_head = nn.Linear(4, 7)
        self.q0_head = nn.Linear(4, 5)
        if frozen:
            self.backbone.requires_grad_(False)


RATES = {'backbone': 1e-5, 'adapter': 1e-4, 'heads': 2e-4}


@pytest.mark.parametrize('frozen', [True, False])
def test_parameter_membership_and_rates(frozen):
    m = TinyModel(frozen=frozen)
    groups = parameter_groups(m, RATES)
    names = [g['name'] for g in groups]
    assert names == (['adapter', 'heads'] if frozen else ['backbone', 'adapter', 'heads'])
    assert {id(p) for g in groups for p in g['params']} == {id(p) for p in m.parameters() if p.requires_grad}
    assert len([p for g in groups for p in g['params']]) == len({id(p) for g in groups for p in g['params']})
    for g in groups:
        assert g['lr'] == RATES[g['name']]
        assert g['weight_decay'] == 0.01


def test_adamw_step_and_state_resume():
    m = TinyModel()
    opt = build_t04_optimizer(m, RATES)
    assert isinstance(opt, torch.optim.AdamW)
    loss = sum(p.square().sum() for p in m.parameters())
    loss.backward()
    opt.step()
    state = opt.state_dict()
    m2 = TinyModel()
    opt2 = build_t04_optimizer(m2, RATES)
    opt2.load_state_dict(state)
    assert learning_rate_summary(opt2) == RATES
    assert len(opt2.state) == len(opt.state)


def test_legacy_adam_opt_in():
    m = TinyModel(frozen=True)
    opt = build_t04_optimizer(m, {'backbone': 1e-4, 'adapter': 1e-4, 'heads': 1e-4},
                              optimizer_type='adam', weight_decay=0)
    assert isinstance(opt, torch.optim.Adam)
    assert all(g['weight_decay'] == 0 for g in opt.param_groups)


@pytest.mark.parametrize('bad', [0, -1, float('nan'), float('inf'), True])
def test_reject_bad_rate(bad):
    with pytest.raises(ValueError):
        parameter_groups(TinyModel(), dict(RATES, backbone=bad))


@pytest.mark.parametrize('bad', [-1, float('nan'), float('inf'), True])
def test_reject_bad_weight_decay(bad):
    with pytest.raises(ValueError):
        parameter_groups(TinyModel(), RATES, weight_decay=bad)


def test_reject_missing_group():
    with pytest.raises(ValueError):
        parameter_groups(TinyModel(), {'heads': 1e-4})


def test_reject_unclassified_parameters():
    m = TinyModel()
    m.extra = nn.Parameter(torch.ones(2))
    with pytest.raises(RuntimeError, match='exactly cover'):
        parameter_groups(m, RATES)


def test_reject_unsupported_optimizer():
    with pytest.raises(ValueError):
        build_t04_optimizer(TinyModel(), RATES, optimizer_type='sgd')
