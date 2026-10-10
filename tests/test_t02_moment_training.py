from types import SimpleNamespace
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_utils.train_epoch import train_one_epoch
from train_utils.evaluate import evaluate
from train_utils.moment_training import run_moment_training_loop


class TinyMomentNet(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(channels * 8 * 8, 12), nn.ReLU())
        self.alpha_head = nn.Linear(12, 7)
        self.q0_head = nn.Linear(12, 5)

    def forward(self, x):
        feat = self.encoder(x)
        return {'alpha_output': self.alpha_head(feat), 'q0_output': self.q0_head(feat)}


class Dataset(torch.utils.data.Dataset):
    def __init__(self, channels):
        self.x = torch.rand(8, channels, 8, 8)
        self.alpha = torch.arange(8) % 7
        self.q0 = torch.arange(8) % 5
    def __len__(self): return 8
    def __getitem__(self, i):
        return self.x[i], {'alpha_output': self.alpha[i], 'q0_output': self.q0[i]}


def test_two_head_training_and_eval(tmp_path):
    for channels in (1, 2, 5):
        loader = DataLoader(Dataset(channels), batch_size=4)
        model = TinyMomentNet(channels)
        optim = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = {'alpha_output': nn.CrossEntropyLoss(), 'q0_output': nn.CrossEntropyLoss()}
        before = model.encoder[1].weight.detach().clone()
        train = train_one_epoch(loader, model, criterion, optim, 'cpu', dataset_type='moment')
        assert not torch.equal(before, model.encoder[1].weight)
        assert 0 <= train['accuracy'] <= 1
        assert 'loss_energy' not in train and 'energy' not in train
        val = evaluate(loader, model, criterion, 'cpu', dataset_type='moment')
        assert len(val['alpha']['confusion_matrix']) == 7
        assert len(val['q0']['confusion_matrix']) == 5
        assert len(val['q0']['confusion_matrix'][0]) == 5
        assert 'energy' not in val


def test_epoch_checkpoint_and_metrics(tmp_path):
    loader = DataLoader(Dataset(2), batch_size=4)
    model = TinyMomentNet(2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = {'alpha_output': nn.CrossEntropyLoss(), 'q0_output': nn.CrossEntropyLoss()}
    cfg = SimpleNamespace(output_dir=str(tmp_path), epochs=2, patience=3,
                          loss_weights={'alpha_output': 1.0, 'q0_output': 1.0}, moments=[1,2])
    best_epoch, best_acc, best_metrics = run_moment_training_loop(
        cfg, loader, loader, 'cpu', model, criterion, opt)
    assert best_epoch in (1, 2)
    assert 0 <= best_acc <= 1
    assert (tmp_path / 'last_model.pth').is_file()
    assert (tmp_path / 'best_model.pth').is_file()
    assert (tmp_path / 'moment_epoch_metrics.csv').is_file()
    assert torch.load(tmp_path / 'best_model.pth', map_location='cpu', weights_only=False)['dataset_type'] == 'moment'


def test_missing_head_rejected():
    loader = DataLoader(Dataset(1), batch_size=4)
    model = TinyMomentNet(1)
    criterion = {'alpha_output': nn.CrossEntropyLoss()}
    import pytest
    with pytest.raises(KeyError):
        evaluate(loader, model, criterion, 'cpu', dataset_type='moment')
