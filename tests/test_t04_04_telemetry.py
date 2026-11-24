import json
import torch
import pytest
from train_utils.t04_telemetry import epoch_telemetry, save_session_epoch_telemetry


def test_group_lr_and_batch_telemetry(tmp_path):
    p = torch.nn.Parameter(torch.ones(2))
    opt = torch.optim.AdamW([{'params': [p], 'lr':1e-4, 'name':'heads'}])
    import time
    info = epoch_telemetry(time.perf_counter()-1, 'cpu', opt,
        {'samples':11,'optimizer_steps':3,'optimizer_steps_completed':3,'optimizer_steps_skipped':0},
        {}, 1, 4)
    assert info['learning_rates'] == {'heads':1e-4}
    assert info['configured_effective_batch'] == 4
    assert info['samples_per_second'] > 0
    assert info['gpu_name'] is None
    assert info['optimizer_steps_completed'] == 3
    path = save_session_epoch_telemetry(tmp_path, 1, info)
    assert json.loads(path.read_text())['epoch'] == 1
    with pytest.raises(FileExistsError):
        save_session_epoch_telemetry(tmp_path, 1, info)


def test_duplicate_lr_group_rejected():
    a,b = torch.nn.Parameter(torch.ones(1)),torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.AdamW([{'params':[a],'lr':1e-4,'name':'heads'},
                              {'params':[b],'lr':2e-4,'name':'heads'}])
    import time
    with pytest.raises(RuntimeError):
        epoch_telemetry(time.perf_counter()-1,'cpu',opt,{}, {},1,1)
