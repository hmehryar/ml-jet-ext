"""Regression checks for streaming accumulation; no retained loss graphs."""
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from train_utils.t04_precision import run_epoch


class Uneven(Dataset):
    def __len__(self):
        return 7

    def __getitem__(self, i):
        return torch.tensor([i / 10., (7 - i) / 10.], dtype=torch.float32), {
            'alpha_output': torch.tensor(i % 7),
            'q0_output': torch.tensor(i % 5),
        }


class DualHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Linear(2, 7)
        self.q0 = nn.Linear(2, 5)

    def forward(self, x):
        return {'alpha_output': self.alpha(x), 'q0_output': self.q0(x)}


def test_streaming_uneven_final_window_matches_large_batches():
    torch.manual_seed(300)
    initial = DualHeads()
    streamed, reference = copy.deepcopy(initial), copy.deepcopy(initial)
    criterion = {'alpha_output': nn.CrossEntropyLoss(),
                 'q0_output': nn.CrossEntropyLoss()}
    a = torch.optim.SGD(streamed.parameters(), lr=0.01)
    b = torch.optim.SGD(reference.parameters(), lr=0.01)
    result = run_epoch(DataLoader(Uneven(), batch_size=2), streamed, criterion,
                       'cpu', optimizer=a, precision='fp32', accumulation=2)
    run_epoch(DataLoader(Uneven(), batch_size=4), reference, criterion,
              'cpu', optimizer=b, precision='fp32', accumulation=1)
    assert result['optimizer_steps'] == 2
    assert result['microbatches'] == 4
    assert result['samples'] == 7
    for actual, expected in zip(streamed.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_streaming_backward_happens_before_next_forward():
    """A microbatch's loss must be consumed before the next is loaded."""
    class Guarded(Dataset):
        def __len__(self):
            return 4
        def __getitem__(self, index):
            return Uneven()[index]

    class CountingModel(DualHeads):
        def __init__(self):
            super().__init__()
            self.forward_calls = 0
            self.backward_calls = 0
            self.alpha.weight.register_hook(self._mark_backward)
        def _mark_backward(self, grad):
            self.backward_calls += 1
            return grad
        def forward(self, x):
            assert self.forward_calls == self.backward_calls, (
                'Previous microbatch graph was retained until a later forward')
            self.forward_calls += 1
            return super().forward(x)

    model = CountingModel()
    criterion = {'alpha_output': nn.CrossEntropyLoss(),
                 'q0_output': nn.CrossEntropyLoss()}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    report = run_epoch(DataLoader(Guarded(), batch_size=1), model, criterion,
                       'cpu', optimizer=optimizer, accumulation=4)
    assert report['optimizer_steps'] == 1
    assert model.forward_calls == model.backward_calls == 4
