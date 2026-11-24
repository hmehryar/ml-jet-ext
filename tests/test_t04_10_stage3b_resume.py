"""Stage 3B CPU contract tests. Real HF backbones are tested via Slurm separately."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from tools.t04_10_stage3b_resume import check_metrics, validate_state, _optimizer_steps
from train_utils.t04_10_checkpoint import fit_fixed_budget
from train_utils.t04_10_training import run_single_head_epoch
from models.t04_10_single_head import SingleHeadMomentVisionClassifier, build_trainable_optimizer


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config=SimpleNamespace(hidden_size=4)
        self.proj=nn.Conv2d(3,4,1)
    def forward(self,pixel_values):
        return SimpleNamespace(pooler_output=self.proj(pixel_values).mean((2,3)))

class TinyDataset(Dataset):
    def __len__(self): return 4
    def __getitem__(self,i):
        return torch.ones(2,32,32)*(.2+.1*i),{
            'alpha_output':torch.tensor(i%7), 'q0_output':torch.tensor(i%5)}


def make(task):
    m=SingleHeadMomentVisionClassifier(TinyBackbone(),task=task,channels=2,strategy='S3')
    o=build_trainable_optimizer(m)
    s=torch.optim.lr_scheduler.ReduceLROnPlateau(o,mode='min')
    return m,o,s

@pytest.mark.parametrize('task',['alpha_only','q0_only'])
def test_two_epoch_checkpoint_contract_cpu(tmp_path,task):
    dl=DataLoader(TinyDataset(),batch_size=2)
    digest=hashlib.sha256(b'fixture config').hexdigest()
    model,opt,sch=make(task)
    fit_fixed_budget(model=model,train_loader=dl,val_loader=dl,optimizer=opt,
        task=task,output_dir=tmp_path/'training',config_sha256=digest,epochs=1,
        scheduler=sch)
    state1=torch.load(tmp_path/'training/last.pth',map_location='cpu',weights_only=False)
    validate_state(state1,task=task,epoch=1,config_sha256=digest)
    step1=_optimizer_steps(state1)
    del model,opt,sch
    model,opt,sch=make(task)
    fit_fixed_budget(model=model,train_loader=dl,val_loader=dl,optimizer=opt,
        task=task,output_dir=tmp_path/'training',config_sha256=digest,epochs=2,
        scheduler=sch,resume=True)
    state2=torch.load(tmp_path/'training/last.pth',map_location='cpu',weights_only=False)
    validate_state(state2,task=task,epoch=2,config_sha256=digest)
    # CPU run does not set CUDA RNG; the CUDA-specific gate is intentionally GPU-only.
    assert _optimizer_steps(state2)>step1
    assert state2['history'][0] == state1['history'][0]
    assert len(json.loads((tmp_path/'training/history.json').read_text()))==2
    assert Path(tmp_path/'training/best.pth').is_file()
    for row in state2['history']:
        check_metrics(row['val'],task=task,expected_n=4)


def test_check_metrics_catches_bad_ranking():
    metrics={'task':'q0_only','n':2,'accuracy':1.,'macro_f1':1.,
       'top1_accuracy':1.,'top2_accuracy':0.5,'top3_accuracy':1.,
       'within_1_class_accuracy':1.,'within_2_class_accuracy':1.,
       'physical_mae':0.,'physical_rmse':0.,'loss':0.5,
       'confusion_matrix':[[2,0,0,0,0]]+[[0]*5 for _ in range(4)]}
    with pytest.raises(AssertionError,match='Top-k monotonicity'):
        check_metrics(metrics,task='q0_only',expected_n=2)


def test_validate_state_rejects_wrong_task(tmp_path):
    dl=DataLoader(TinyDataset(),batch_size=2)
    digest='c'*64
    m,o,s=make('alpha_only')
    fit_fixed_budget(model=m,train_loader=dl,val_loader=dl,optimizer=o,
        task='alpha_only',output_dir=tmp_path/'train',config_sha256=digest,
        epochs=1,scheduler=s)
    x=torch.load(tmp_path/'train/last.pth',weights_only=False)
    with pytest.raises(AssertionError,match='metadata'):
        validate_state(x,task='q0_only',epoch=1,config_sha256=digest)


def test_restore_rng_accepts_cuda_mapped_state():
    import torch
    from train_utils.t04_10_checkpoint import _rng_state, _restore_rng

    state = _rng_state()

    if torch.cuda.is_available():
        state["torch"] = state["torch"].cuda()
        if state["cuda"] is not None:
            state["cuda"] = [s.cuda() for s in state["cuda"]]

    _restore_rng(state)

    assert torch.equal(
        torch.get_rng_state(),
        state["torch"].cpu()
    )
