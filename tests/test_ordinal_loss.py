"""Ordinal-distance semantics, CE compatibility and two-epoch integration."""
import csv
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from train_utils.loss_config import resolve_moment_loss
from train_utils.ordinal_loss import (CEOrdinalLoss, build_moment_criterion,
                                     physical_class_metrics)
from train_utils.moment_training import run_moment_training_loop


def ordinal_for_prediction(values, truth, prediction):
    logits = torch.full((1, len(values)), -50.0)
    logits[0, prediction] = 50.0
    return CEOrdinalLoss(values).components(logits, torch.tensor([truth]))[2].item()


def test_farther_errors_cost_more_and_exact_is_zero():
    values = [0.20,0.25,0.30,0.35,0.40,0.45,0.50]
    assert ordinal_for_prediction(values,2,2) == pytest.approx(0.0,abs=1e-10)
    near = ordinal_for_prediction(values,2,3)
    far = ordinal_for_prediction(values,2,6)
    assert near == pytest.approx(1/6)
    assert far == pytest.approx(4/6)


def test_physical_gaps_and_unit_invariance():
    # Nonuniform gaps, not just class indices, determine the penalty.
    assert ordinal_for_prediction([0,1,4],0,1) == pytest.approx(0.25)
    assert ordinal_for_prediction([0,1,4],1,2) == pytest.approx(0.75)
    assert ordinal_for_prediction([100,110,140],1,2) == pytest.approx(0.75)
    assert ordinal_for_prediction([0,0.05,0.10],0,1) == pytest.approx(
        ordinal_for_prediction([1,1.5,2],0,1))


def test_distribution_shape_matters_even_with_same_mean():
    loss = CEOrdinalLoss([0,1,2])
    target = torch.tensor([1])
    centered = torch.tensor([[-50.0,50.0,-50.0]])
    bimodal = torch.tensor([[0.0,-50.0,0.0]])
    assert loss.components(centered,target)[2] < loss.components(bimodal,target)[2]


def test_ce_default_and_zero_lambda_match_original_gradients():
    torch.manual_seed(42)
    logits = torch.randn(4,7,requires_grad=True)
    target = torch.tensor([0,2,4,6])
    ce = nn.CrossEntropyLoss()(logits,target)
    hybrid = CEOrdinalLoss([0.20,0.25,0.30,0.35,0.40,0.45,0.50],0)(logits,target)
    torch.testing.assert_close(hybrid,ce,rtol=0,atol=0)
    torch.testing.assert_close(torch.autograd.grad(hybrid,logits,retain_graph=True)[0],
                               torch.autograd.grad(ce,logits)[0],rtol=0,atol=0)
    assert all(type(loss) is nn.CrossEntropyLoss for loss in build_moment_criterion().values())


def test_hybrid_decomposition_and_gradient():
    logits = torch.randn(3,5,requires_grad=True)
    loss = CEOrdinalLoss([1,1.5,2,2.5,3],0.7)
    total,ce,ordinal = loss.components(logits,torch.tensor([0,2,4]))
    torch.testing.assert_close(total,ce+0.7*ordinal)
    total.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum()>0
    assert 0 <= ordinal.item() <= 1


@pytest.mark.parametrize('weight',[-1,float('nan'),float('inf')])
def test_invalid_weight(weight):
    with pytest.raises(ValueError):
        resolve_moment_loss({'type':'ce_ordinal','ordinal_weight':weight})


@pytest.mark.parametrize('values',[[1],[1,1],[2,1],[0,float('nan')]])
def test_invalid_grid(values):
    with pytest.raises(ValueError):
        CEOrdinalLoss(values)


def test_physical_metrics_use_parameter_values():
    metrics = physical_class_metrics([0,1],[1,3],[1,1.5,2,2.5,3])
    assert metrics['mae'] == pytest.approx(0.75)
    assert metrics['rmse'] == pytest.approx((0.625)**0.5)
    assert metrics['normalized_mae'] == pytest.approx(0.375)


def test_two_epoch_hybrid_training_and_checkpoint(tmp_path):
    class Dataset(torch.utils.data.Dataset):
        def __len__(self): return 8
        def __getitem__(self,i):
            return torch.ones(3)*(i/8),{'alpha_output':torch.tensor(i%7),'q0_output':torch.tensor(i%5)}
    class Model(nn.Module):
        def __init__(self):
            super().__init__(); self.fc=nn.Linear(3,12)
        def forward(self,x):
            logits=self.fc(x)
            return {'alpha_output':logits[:,:7],'q0_output':logits[:,7:]}
    spec = resolve_moment_loss({'type':'ce_ordinal','ordinal_weight':1.0})
    cfg = SimpleNamespace(output_dir=str(tmp_path),epochs=2,patience=10,moments=[1,2],moment_loss=spec)
    model=Model(); before=model.fc.weight.detach().clone()
    opt=torch.optim.Adam(model.parameters(),lr=0.01)
    criterion=build_moment_criterion(spec)
    loader=DataLoader(Dataset(),batch_size=3)  # uneven last batch tests sample-weighted logs
    run_moment_training_loop(cfg,loader,loader,'cpu',model,criterion,opt)
    with (tmp_path/'moment_epoch_metrics.csv').open() as f: rows=list(csv.DictReader(f))
    assert [int(row['epoch']) for row in rows]==[1,2]
    for row in rows:
        assert float(row['val_alpha_loss']) == pytest.approx(
            float(row['val_alpha_ce_loss'])+float(row['val_alpha_ordinal_loss']))
        assert float(row['val_q0_mae']) >= 0
        assert 0 <= float(row['val_joint_accuracy']) <= 1
    state=torch.load(tmp_path/'last_model.pth',weights_only=True)
    assert state['epoch']==2 and state['loss_config']==spec
    assert state['criterion_settings']['alpha_output']['ordinal_weight']==1.0
    assert not torch.equal(before,model.fc.weight)
    json.dumps(state['metrics'],allow_nan=False)
    from train_utils.t03_reporting import generate_t03_report
    summary=generate_t03_report(model,loader,'cpu',tmp_path,expected_samples=8)
    best_row=rows[summary['best_epoch']-1]
    assert summary['loss_config']['type']=='ce_ordinal'
    assert summary['alpha']['mae']==pytest.approx(float(best_row['val_alpha_mae']))
    assert summary['q0']['rmse']==pytest.approx(float(best_row['val_q0_rmse']))
    assert (tmp_path/'plots/alpha_loss_components.png').is_file()
    assert (tmp_path/'plots/q0_physical_mae.pdf').is_file()
