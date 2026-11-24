import copy
import math
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from train_utils.t04_precision import run_epoch, accumulation_windows, make_scaler, autocast_context

class Samples(Dataset):
    def __len__(self): return 7
    def __getitem__(self, i):
        x=torch.tensor([float(i)/10, float(7-i)/10])
        return x, {'alpha_output':torch.tensor(i%7), 'q0_output':torch.tensor(i%5)}

class Heads(nn.Module):
    def __init__(self):
        super().__init__(); self.alpha=nn.Linear(2,7); self.q0=nn.Linear(2,5)
    def forward(self,x): return {'alpha_output':self.alpha(x),'q0_output':self.q0(x)}

@pytest.fixture
def criterion(): return {'alpha_output':nn.CrossEntropyLoss(),'q0_output':nn.CrossEntropyLoss()}

def test_final_window():
    assert accumulation_windows(5,2)==[(0,2),(2,4),(4,5)]
    with pytest.raises(ValueError): accumulation_windows(5,0)

def test_fp32_unequal_final_window_matches_reference(criterion):
    torch.manual_seed(10)
    original=Heads()
    accumulated=copy.deepcopy(original); reference=copy.deepcopy(original)
    micro=DataLoader(Samples(),batch_size=2,shuffle=False)
    # First window of 4 and second window of 3 samples
    large=DataLoader(Samples(),batch_size=4,shuffle=False)
    opt1=torch.optim.SGD(accumulated.parameters(),lr=.01)
    opt2=torch.optim.SGD(reference.parameters(),lr=.01)
    metrics=run_epoch(micro,accumulated,criterion,'cpu',optimizer=opt1,accumulation=2)
    run_epoch(large,reference,criterion,'cpu',optimizer=opt2,accumulation=1)
    assert metrics['optimizer_steps']==2
    assert metrics['microbatches']==4
    for a,b in zip(accumulated.parameters(),reference.parameters()):
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)

def test_fp32_single_step(criterion):
    model=Heads(); opt=torch.optim.Adam(model.parameters())
    report=run_epoch(DataLoader(Samples(),batch_size=3),model,criterion,'cpu',optimizer=opt)
    assert report['optimizer_steps']==3 and math.isfinite(report['loss'])
    assert 'alpha' in report and 'q0' in report

def test_evaluation_does_not_update(criterion):
    model=Heads(); old=[p.clone() for p in model.parameters()]
    report=run_epoch(DataLoader(Samples(),batch_size=3),model,criterion,'cpu')
    assert report['optimizer_steps']==0
    assert all(torch.equal(a,b) for a,b in zip(old,model.parameters()))

def test_amp_requires_cuda():
    with pytest.raises(ValueError): make_scaler('cpu','fp16')
    with pytest.raises(ValueError): autocast_context('cpu','bf16')
    assert make_scaler('cpu','fp32') is None
    assert make_scaler('cpu','bf16') is None

def test_fp16_requires_scaler(criterion):
    with pytest.raises(ValueError): run_epoch(DataLoader(Samples(),batch_size=2),Heads(),criterion,'cpu',optimizer=torch.optim.SGD(Heads().parameters(),lr=.1),precision='fp16')
