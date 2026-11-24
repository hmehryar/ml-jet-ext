import csv
import random
import numpy as np
import pytest
import torch

from train_utils.t04_resume_state import (
    ResumeError, atomic_torch_save, checkpoint_state, load_checkpoint,
    restore_training, reconcile_epoch_csv, seed_loader_epoch,
)


def fixture():
    model = torch.nn.Linear(3, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.08, momentum=0.9)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=0, factor=0.5)
    return model, opt, sched


def train_step(model, opt, sched):
    x = torch.randn(7, 3)
    y = torch.randn(7, 2)
    opt.zero_grad()
    loss = (model(x) - y).square().mean()
    loss.backward()
    opt.step()
    sched.step(1.0)
    return float(loss.item())


def make_state(epoch, model, opt, sched, loss):
    return checkpoint_state(epoch=epoch, model=model, optimizer=opt, scheduler=sched,
        scaler=None, best_metric=1., best_epoch=1, stagnant=epoch-1,
        manifest_sha256='manifest', config_sha256='config',
        metrics_row={'epoch': epoch, 'loss': loss}, metrics_columns=['epoch','loss'])


def test_interrupted_resume_is_bitwise_equivalent(tmp_path):
    torch.manual_seed(91); np.random.seed(91); random.seed(91)
    model, opt, sched = fixture()
    control = [train_step(model, opt, sched) for _ in range(6)]
    final = {k: v.detach().clone() for k,v in model.state_dict().items()}
    lr = opt.param_groups[0]['lr']

    torch.manual_seed(91); np.random.seed(91); random.seed(91)
    model, opt, sched = fixture()
    resumed = [train_step(model,opt,sched) for _ in range(3)]
    ckpt = tmp_path/'last_model.pth'
    atomic_torch_save(make_state(3,model,opt,sched,resumed[-1]),ckpt)
    # Simulate a fresh process with unrelated RNG state.
    torch.manual_seed(555); random.seed(555); np.random.seed(555)
    model2,opt2,sched2 = fixture()
    state = load_checkpoint(ckpt, manifest_sha256='manifest', config_sha256='config')
    assert restore_training(state,model=model2,optimizer=opt2,scheduler=sched2)[0] == 3
    resumed += [train_step(model2,opt2,sched2) for _ in range(3)]
    assert control == resumed
    assert opt2.param_groups[0]['lr'] == lr
    assert all(torch.equal(v, model2.state_dict()[k]) for k,v in final.items())


def test_checkpoint_identity_rejected(tmp_path):
    m,o,s=fixture()
    p=tmp_path/'last_model.pth'
    atomic_torch_save(make_state(1,m,o,s,1.0),p)
    with pytest.raises(ResumeError,match='identity'):
        load_checkpoint(p,manifest_sha256='other',config_sha256='config')


def test_old_and_corrupt_checkpoint_rejected(tmp_path):
    p=tmp_path/'old.pth'; torch.save({'epoch':1},p)
    with pytest.raises(ResumeError,match='not a T04-02'):
        load_checkpoint(p,manifest_sha256='manifest',config_sha256='config')
    p.write_bytes(b'corrupted')
    with pytest.raises(ResumeError,match='unreadable'):
        load_checkpoint(p,manifest_sha256='manifest',config_sha256='config')


def test_csv_crash_repair_and_duplicate_detection(tmp_path):
    m,o,s=fixture(); state=make_state(1,m,o,s,3.5)
    p=tmp_path/'metrics.csv'
    reconcile_epoch_csv(p,state)
    reconcile_epoch_csv(p,state)
    assert len(list(csv.DictReader(p.open()))) == 1
    with p.open('a') as f: f.write('1,3.5\n')
    with pytest.raises(ResumeError,match='duplicate'):
        reconcile_epoch_csv(p,state)


def test_epoch_loader_determinism():
    data=torch.arange(100)
    generator=torch.Generator()
    loader=torch.utils.data.DataLoader(data,batch_size=10,shuffle=True,generator=generator,num_workers=0)
    seed_loader_epoch(loader,42,3)
    a=torch.cat(list(loader))
    seed_loader_epoch(loader,42,3)
    b=torch.cat(list(loader))
    assert torch.equal(a,b)
    seed_loader_epoch(loader,42,4)
    assert not torch.equal(a,torch.cat(list(loader)))


def test_scheduler_mismatch_rejected():
    m,o,s=fixture(); state=make_state(1,m,o,s,3.5)
    with pytest.raises(ResumeError,match='Scheduler'):
        restore_training(state,model=m,optimizer=o,scheduler=None)
