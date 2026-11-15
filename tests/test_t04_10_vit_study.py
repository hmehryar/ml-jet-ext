"""Stage A ViT E1-E5 smoke matrix and CPU-only regression tests."""
from pathlib import Path
import pytest
import torch
from torch import nn
from tools.t04_10_vit_pilot import rows, selected, E_VALUES, TREATMENTS, loss_spec
from models.t04_10_single_head import SingleHeadViTClassifier
from train_utils.t04_10_preflight import load_settings

class FakeViT(nn.Module):
    num_features=8
    def __init__(self):
        super().__init__()
        self.proj=nn.Linear(3,8)
    def forward(self,x):
        return self.proj(x.mean((-1,-2)))

@pytest.mark.parametrize('task,classes',[('alpha_only',7),('q0_only',5)])
@pytest.mark.parametrize('e', E_VALUES)
def test_shapes_gradients_and_adapter(task,classes,e):
    model=SingleHeadViTClassifier(FakeViT(),task=task,channels=e)
    x=torch.randn(2,e,32,32)
    out=model(x)
    assert len(out)==1
    assert next(iter(out.values())).shape==(2,classes)
    # M1 is replicated into 3 channels; all others start at zero.
    with torch.no_grad():
        assert torch.all(model.input_adapter.weight[:,0,0,0] == 1)
        if e>1:assert torch.all(model.input_adapter.weight[:,1:,0,0] == 0)
    nn.CrossEntropyLoss()(next(iter(out.values())),torch.tensor([0,classes-1])).backward()
    for name,parameter in model.named_parameters():
        if parameter.requires_grad:assert parameter.grad is not None, name
    assert not hasattr(model,'q0_head') and not hasattr(model,'alpha_head')


def test_exact_matrix_and_fail_closed_selector():
    all_rows=list(rows())
    assert len(all_rows)==10
    assert len({r['run_id'] for r in all_rows})==10
    assert {r['e'] for r in all_rows}==set(E_VALUES)
    assert {r['task'] for r in all_rows}=={'alpha_only','q0_only'}
    assert {r['seed'] for r in all_rows}=={42}
    assert TREATMENTS==(('ce',None),)
    assert len(selected())==10
    assert len(selected(only=all_rows[0]['run_id']))==1
    assert len(selected(exclude=[all_rows[0]['run_id']]))==9
    for field in ('only','exclude'):
        with pytest.raises(ValueError,match='Empty'):
            selected(**{field:''})
    with pytest.raises(ValueError,match='Unknown'):
        selected(only='invalid')
    with pytest.raises(ValueError,match='Unknown'):
        selected(treatments=['physical_mae'])

@pytest.mark.parametrize('e', E_VALUES)
def test_scratch_preflight_E1_to_E5(e,tmp_path):
    import yaml
    from tools.t04_10_vit_pilot import BASE_TEMPLATE, SCRATCH_MODEL, SCRATCH_REVISION
    cfg=yaml.safe_load(BASE_TEMPLATE.read_text())
    cfg.update(model='vit',task='alpha_only',strategy='S4',moments=list(range(1,e+1)),
               checkpoint=SCRATCH_MODEL,checkpoint_revision=SCRATCH_REVISION,
               vit_initialization='random')
    path=tmp_path/'config.yml'
    path.write_text(yaml.safe_dump(cfg))
    assert len(load_settings(path)['moments'])==e
    cfg['vit_initialization']='jetbench'
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='random'):
        load_settings(path)
