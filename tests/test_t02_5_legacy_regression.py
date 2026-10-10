"""Legacy three-head train/evaluate regression without requiring legacy dataset files."""
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from train_utils.train_epoch import train_one_epoch
from train_utils.evaluate import evaluate


class LegacyFakeDataset(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        return torch.full((1, 8, 8), index / 8, dtype=torch.float32), {
            'energy_loss_output': torch.tensor(index % 2, dtype=torch.long),
            'alpha_output': torch.tensor(index % 7, dtype=torch.long),
            'q0_output': torch.tensor(index % 5, dtype=torch.long),
        }


class LegacyFakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(64, 12), nn.ReLU())
        self.energy = nn.Linear(12, 1)
        self.alpha = nn.Linear(12, 7)
        self.q0 = nn.Linear(12, 5)

    def forward(self, inputs):
        features = self.encoder(inputs)
        return {
            'energy_loss_output': self.energy(features),
            'alpha_output': self.alpha(features),
            'q0_output': self.q0(features),
        }


def test_legacy_three_head_train_and_evaluate():
    loader = DataLoader(LegacyFakeDataset(), batch_size=4)
    model = LegacyFakeModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criteria = {
        'energy_loss_output': nn.BCEWithLogitsLoss(),
        'alpha_output': nn.CrossEntropyLoss(),
        'q0_output': nn.CrossEntropyLoss(),
    }
    initial = model.encoder[1].weight.detach().clone()
    train = train_one_epoch(loader, model, criteria, optimizer, 'cpu', dataset_type='legacy')
    assert not torch.equal(initial, model.encoder[1].weight)
    assert all(x in train for x in ('loss_energy', 'accuracy_energy', 'accuracy_alpha', 'accuracy_q0'))
    val = evaluate(loader, model, criteria, 'cpu', dataset_type='legacy')
    assert all(x in val for x in ('energy', 'alpha', 'q0', 'loss_energy', 'accuracy'))
    assert 0 <= val['accuracy'] <= 1


def test_legacy_default_dispatch():
    loader = DataLoader(LegacyFakeDataset(), batch_size=4)
    model = LegacyFakeModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criteria = {'energy_loss_output': nn.BCEWithLogitsLoss(),
                'alpha_output': nn.CrossEntropyLoss(), 'q0_output': nn.CrossEntropyLoss()}
    train = train_one_epoch(loader, model, criteria, optimizer, 'cpu')
    assert 'loss_energy' in train
