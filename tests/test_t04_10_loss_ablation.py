"""Read-only identity and objective checks for new T04-10 loss ablations."""
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from tools.t04_10_loss_ablation import matrix, treatment, TREATMENTS, selected
from train_utils.loss_config import resolve_moment_loss
from train_utils.ordinal_loss import build_moment_criterion
from train_utils.t04_10_session_training import train_session


def test_exact_t04_07_matrix_84_unique(tmp_path):
    rows=list(matrix(tmp_path))
    assert len(rows)==84 and len({r['run_id'] for r in rows})==84
    assert len({str(r['run']) for r in rows})==84
    assert {r['loss']['type'] for r in rows} == {'physical_mae','ce_ordinal','ce_physical_mae'}
    assert {r['seed'] for r in rows}=={42,123,2026}
    assert len(TREATMENTS)==7
    assert set(r['treatment'] for r in rows)=={
        'physical_mae','ce_ordinal_lambda0p1','ce_ordinal_lambda0p5','ce_ordinal_lambda1p0',
        'ce_physical_mae_lambda0p1','ce_physical_mae_lambda0p5','ce_physical_mae_lambda1p0'}
    assert all(len([r for r in rows if r['model']==model and r['task']==task and r['seed']==seed])==7
               for model in ('clip','siglip') for task in ('alpha_only','q0_only')
               for seed in (42,123,2026))


def test_exact_t04_07_treatment_source():
    from tools.t04_07_production import TREATMENTS as old
    assert TREATMENTS == old


def test_filter_and_exclude(tmp_path):
    rows=selected(root=tmp_path,treatments=['physical_mae'])
    assert len(rows)==12
    assert len(selected(selector=rows[0]['run_id'],root=tmp_path))==1
    assert len(selected(exclude=rows[0]['run_id'],root=tmp_path))==83


@pytest.mark.parametrize('kind,weight', TREATMENTS)
def test_loss_objectives_finite_and_gradients(kind,weight):
    for task,key,k in [('alpha_only','alpha_output',7),('q0_only','q0_output',5)]:
        x=torch.randn(5,k,requires_grad=True)
        target=torch.tensor([0,1,2,3,4])
        loss=build_moment_criterion(treatment(kind,weight))[key](x,target)
        assert torch.isfinite(loss)
        loss.backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()


def test_coefficients_canonical_and_rejected():
    for kind,weight in TREATMENTS:
        assert resolve_moment_loss(treatment(kind,weight))['type']==kind
    for bad in (0,-1,True,float('nan'),float('inf'),0.75):
        with pytest.raises(ValueError): treatment('ce_ordinal',bad)
    with pytest.raises(ValueError): treatment('physical_mae',.5)


def test_train_session_accepts_criterion_and_loss_identity(tmp_path,monkeypatch):
    import train_utils.t04_10_sessions as sessions
    from test_t04_10_stage6_sessions import components
    import hashlib
    run=tmp_path/'run';run.mkdir()
    (run/'config.resolved.json').write_text('{}')
    ch=hashlib.sha256((run/'config.resolved.json').read_bytes()).hexdigest()
    (run/'run_manifest.json').write_text(json.dumps({'config_sha256':ch}))
    monkeypatch.setattr(sessions,'_validate_run',lambda p:Path(p).resolve())
    monkeypatch.setattr(__import__('train_utils.t04_10_session_training',fromlist=['latest_checkpoint']),
                        'latest_checkpoint',sessions.latest_checkpoint)
    from train_utils.t04_10_session_training import run_session
    with sessions.experiment_lock(run):
        session,_=run_session(run=run,mode='fresh',task='alpha_only',epochs=2,config_sha256=ch,
             criterion=build_moment_criterion(treatment('ce_ordinal',.5))['alpha_output'],
             loss_sha256='a'*64,**components('alpha_only'))
    pointer=sessions.latest_checkpoint(run)
    state=torch.load(pointer['absolute_path'],weights_only=False,map_location='cpu')
    assert state['loss_sha256']=='a'*64
    with sessions.experiment_lock(run):
        with pytest.raises(sessions.SessionError,match='identity'):
            run_session(run=run,mode='resume',task='alpha_only',epochs=3,config_sha256=ch,
                 criterion=build_moment_criterion(treatment('ce_ordinal',.5))['alpha_output'],
                 loss_sha256='b'*64,**components('alpha_only'))
