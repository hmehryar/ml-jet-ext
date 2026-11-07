"""CPU regression & numerical validation for T04-07."""
import math
import pytest
import torch
from torch.nn import functional as F
from train_utils.ordinal_loss import PhysicalMAELoss, CEOrdinalLoss, build_moment_criterion
from train_utils.loss_config import resolve_moment_loss
from train_utils.t04_precision import _losses
from train_utils.t04_session_worker import validate_strategy_settings

ALPHA=[.20,.25,.30,.35,.40,.45,.50]
Q0=[1.,1.5,2.,2.5,3.]

def test_grids_and_distance():
    l=PhysicalMAELoss(Q0)
    z=torch.tensor([[0.,0.,0.,0.,0.]],requires_grad=True)
    t=torch.tensor([0])
    assert l(z,t).item()==pytest.approx(.5,abs=1e-6)
    z2=torch.tensor([[0.,-8.,-8.,-8.,8.]])
    z3=torch.tensor([[8.,-8.,-8.,-8.,0.]])
    assert l(z2,t)>l(z3,t)
    l(z,t).backward(); assert torch.isfinite(z.grad).all()

def test_correct_physical_grids():
    c=build_moment_criterion({'type':'physical_mae'})
    assert c['alpha_output'].class_values.tolist()==pytest.approx(ALPHA)
    assert c['q0_output'].class_values.tolist()==pytest.approx(Q0)

@pytest.mark.parametrize('kind', ['physical_mae','ce_physical_mae'])
def test_finite_gradients_extreme_logits(kind):
    for grid in (ALPHA,Q0):
        n=len(grid); x=torch.tensor([[10000.]+[-10000.]*(n-1)],requires_grad=True)
        target=torch.tensor([n-1])
        l=PhysicalMAELoss(grid,include_ce=kind.startswith('ce_'),mae_weight=.5)
        value=l(x,target);value.backward()
        assert torch.isfinite(value) and torch.isfinite(x.grad).all()

@pytest.mark.parametrize('dtype', [torch.float32,torch.float64,torch.float16,torch.bfloat16])
def test_low_precision_numerical_stability(dtype):
    x=torch.randn(4,7,dtype=dtype,requires_grad=True)
    loss=PhysicalMAELoss(ALPHA,include_ce=True,mae_weight=.5)(x,torch.tensor([0,1,3,6]))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()

def test_manual_expected_distance_and_gradient_check():
    loss=PhysicalMAELoss([1.,1.5,2.])
    x=torch.tensor([[.2,-.4,.3]],dtype=torch.float64,requires_grad=True)
    expected=(x.softmax(-1)*torch.tensor([[1.,.5,0.]],dtype=torch.float64)).sum()
    assert loss(x, torch.tensor([2])).item() == pytest.approx(expected.item())
    assert torch.autograd.gradcheck(lambda y:loss(y,torch.tensor([2])),(x,),eps=1e-6,atol=1e-5)

def test_ce_plus_zero_matches_ce():
    x=torch.randn(5,7,dtype=torch.float64,requires_grad=True);y=torch.tensor([0,1,3,6,4])
    v=PhysicalMAELoss(ALPHA,include_ce=True,mae_weight=0)(x,y)
    assert torch.allclose(v,F.cross_entropy(x,y),atol=1e-12)
    assert torch.allclose(torch.autograd.grad(v,x,retain_graph=True)[0],torch.autograd.grad(F.cross_entropy(x,y),x)[0])

def test_expected_loss_not_argmax_mae():
    l=PhysicalMAELoss([1.,2.,3.]); x=torch.zeros(1,3);target=torch.tensor([1])
    assert l(x,target).item()==pytest.approx(1/3,abs=1e-6)

def test_legacy_ce_and_ordinal_are_unchanged():
    c=build_moment_criterion(); assert all(isinstance(v,torch.nn.CrossEntropyLoss) for v in c.values())
    q=build_moment_criterion({'type':'ce_ordinal','ordinal_weight':.5})
    assert all(isinstance(v,CEOrdinalLoss) for v in q.values())
    assert resolve_moment_loss({'type':'cross_entropy'})['ordinal_weight']==0

@pytest.mark.parametrize('bad',[{'type':'ce_physical_mae','mae_weight':float('nan')},
  {'type':'ce_physical_mae','mae_weight':-1.}, {'type':'physical_mae','mae_weight':.4},
  {'type':'ce_ordinal','mae_weight':.5},{'type':'ce_soft_mae'},
  {'type':'cross_entropy','ordinal_weight':2}, {'type':'ce_physical_mae','mae_weight':True}])
def test_bad_config_rejected(bad):
    with pytest.raises(ValueError): resolve_moment_loss(bad)

def test_both_heads_component_and_gradient():
    logits={'alpha_output':torch.randn(2,7,requires_grad=True),'q0_output':torch.randn(2,5,requires_grad=True)}
    labels={'alpha_output':torch.tensor([1,6]),'q0_output':torch.tensor([2,4])}
    criteria=build_moment_criterion({'type':'ce_physical_mae','mae_weight':.5})
    total,individual,parts=_losses(logits,labels,criteria,{})
    total.backward()
    assert all(torch.isfinite(v.grad).all() for v in logits.values())
    assert 'loss_alpha_physical_mae' in parts and 'loss_q0_physical_mae' in parts
    assert 'loss_alpha_ce' in parts and 'loss_q0_ce' in parts

def test_worker_gate_accepts_new_losses():
    cfg={'strategy':'S4','precision':'bf16','gradient_accumulation':1,
         'moment_loss':{'type':'ce_physical_mae','mae_weight':.5},
         'learning_rates':{'heads':1e-4,'adapter':1e-4,'backbone':1e-5},
         'optimizer':{'type':'adamw','weight_decay':.01}}
    assert validate_strategy_settings(cfg)=='S4'


@pytest.mark.parametrize(
    "settings",
    [
        {"type": "cross_entropy"},
        {"type": "ce_ordinal", "ordinal_weight": 1.0},
        {"type": "physical_mae"},
        {"type": "ce_physical_mae", "mae_weight": 0.5},
    ],
)
def test_loss_config_round_trip(settings):
    resolved = resolve_moment_loss(settings)
    assert resolve_moment_loss(resolved) == resolved

    criterion = build_moment_criterion(resolved)
    assert set(criterion) == {"alpha_output", "q0_output"}


@pytest.mark.parametrize(
    "settings",
    [
        {
            "type": "ce_ordinal",
            "ordinal_normalization": "invalid",
        },
        {
            "type": "physical_mae",
            "physical_normalization": "invalid",
        },
        {
            "type": "ce_physical_mae",
            "physical_objective": "invalid",
        },
    ],
)
def test_reject_invalid_loss_metadata(settings):
    with pytest.raises(ValueError):
        resolve_moment_loss(settings)
