"""Regression: legacy T04 session calls omit the optional microbatch size."""
import time

import torch

from train_utils.t04_telemetry import epoch_telemetry


def test_legacy_session_unknown_microbatch_does_not_crash():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD([parameter], lr=0.03, momentum=0.9)
    info = epoch_telemetry(
        time.perf_counter() - 0.01, 'cpu', optimizer,
        {'accuracy': 0.5, 'loss': 1.0}, {'accuracy': 0.5}, None, 1,
    )
    assert info['microbatch_size'] is None
    assert info['configured_effective_batch'] is None
    assert info['gradient_accumulation'] == 1
    assert info['learning_rates'] == {'group_0': 0.03}


def test_configured_effective_batch_preserved_when_known():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([{'params': [parameter], 'lr': 1e-4, 'name': 'heads'}])
    info = epoch_telemetry(time.perf_counter() - 0.01, 'cpu', optimizer, {}, {}, 2, 4)
    assert info['configured_effective_batch'] == 8
    assert info['learning_rates'] == {'heads': 1e-4}
