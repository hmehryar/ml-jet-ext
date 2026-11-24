"""CUDA-only T04-04 synthetic acceptance. No datasets, HF checkpoints or run snapshots.

Run: python -m tools.t04_04_gpu_acceptance --output /path/to/report.json
"""
import argparse
import copy
import json
import math
import os
import platform
import socket
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_utils.t04_precision import make_scaler, run_epoch


class FixedSamples(Dataset):
    def __len__(self):
        return 11

    def __getitem__(self, index):
        # All samples deterministic; no random transforms or shuffled ordering.
        x = torch.tensor([index / 10., (11 - index) / 10.,
                          (index % 4) / 5., 0.25 + index / 40.], dtype=torch.float32)
        return x, {'alpha_output': torch.tensor(index % 7, dtype=torch.long),
                   'q0_output': torch.tensor((index * 3) % 5, dtype=torch.long)}


class TinyDualHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 16), nn.GELU(), nn.Linear(16, 12), nn.GELU())
        self.alpha = nn.Linear(12, 7)
        self.q0 = nn.Linear(12, 5)

    def forward(self, images):
        hidden = self.encoder(images)
        return {'alpha_output': self.alpha(hidden), 'q0_output': self.q0(hidden)}


def parameters(model):
    return torch.cat([p.detach().float().reshape(-1).cpu() for p in model.parameters()])


def execute(initial, device, precision, microbatch, accumulation):
    model = copy.deepcopy(initial).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.035)
    scaler = make_scaler(device, precision)
    criteria = {'alpha_output': nn.CrossEntropyLoss(), 'q0_output': nn.CrossEntropyLoss()}
    before = parameters(model)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = run_epoch(DataLoader(FixedSamples(), batch_size=microbatch, shuffle=False,
                                  num_workers=0), model, criteria, device,
                       optimizer=optimizer, precision=precision,
                       accumulation=accumulation, scaler=scaler)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    after = parameters(model)
    if not math.isfinite(result['loss']) or not torch.isfinite(after).all():
        raise AssertionError(f'Nonfinite loss/parameters in {precision} run')
    if torch.equal(before, after):
        raise AssertionError(f'No parameter updates in {precision} run')
    if result['samples'] != 11 or result['optimizer_steps'] != 3:
        raise AssertionError(f'Wrong sample or step count: {result}')
    evaluation = run_epoch(DataLoader(FixedSamples(), batch_size=microbatch,
                                     shuffle=False, num_workers=0), model,
                           criteria, device, precision=precision)
    if not math.isfinite(evaluation['loss']):
        raise AssertionError(f'Nonfinite {precision} validation loss')
    extra = {}
    if precision == 'fp16':
        if scaler is None or not scaler.is_enabled():
            raise AssertionError('FP16 must have an enabled scaler')
        # Serializable/restoreable state is required by the immutable checkpoint format.
        saved = copy.deepcopy(scaler.state_dict())
        restored = make_scaler(device, 'fp16')
        restored.load_state_dict(saved)
        if restored.state_dict() != saved:
            raise AssertionError('GradScaler state round-trip mismatch')
        extra['scaler_state_roundtrip'] = True
        extra['scaler_scale'] = float(scaler.get_scale())
    elif scaler is not None:
        raise AssertionError(f'{precision} must not construct a GradScaler')
    return {'precision': precision, 'microbatch': microbatch,
            'accumulation': accumulation, 'loss': result['loss'],
            'validation_loss': evaluation['loss'], 'optimizer_steps_attempted': result['optimizer_steps'],
            'microbatches': result['microbatches'], 'samples': result['samples'],
            'mean_effective_batch': result['effective_batch_mean'],
            'elapsed_seconds': elapsed,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
            'parameter_l2_delta': float(torch.linalg.vector_norm(after - before)),
            'parameter_after': after, **extra}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, help='JSON report path outside immutable T04 runs')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required; submit via the Slurm launcher')
    device = torch.device('cuda:0')
    gpu_name = torch.cuda.get_device_name(device)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 unsupported on this CUDA device')
    torch.manual_seed(1729)
    torch.cuda.manual_seed_all(1729)
    initial = TinyDualHead()
    cases = {}
    # 11 samples -> 4+4+3 sample windows in all four runs.
    for label, precision, batch, accumulation in [
        ('fp32_reference', 'fp32', 4, 1),
        ('fp32_accumulated', 'fp32', 1, 4),
        ('bf16_accumulated', 'bf16', 1, 4),
        ('fp16_accumulated', 'fp16', 1, 4),
    ]:
        cases[label] = execute(initial, device, precision, batch, accumulation)
        print(f"{label}: loss={cases[label]['loss']:.6f} "
              f"steps={cases[label]['optimizer_steps_attempted']} "
              f"peak_allocated={cases[label]['peak_allocated_bytes']/2**20:.2f}MiB", flush=True)
    reference = cases['fp32_reference']['parameter_after']
    accumulated = cases['fp32_accumulated']['parameter_after']
    max_abs = float((reference - accumulated).abs().max())
    torch.testing.assert_close(reference, accumulated, rtol=3e-4, atol=3e-5)
    print(f'FP32 reference vs accumulation: max_abs_parameter_diff={max_abs:.3g} PASS', flush=True)
    # Check AMP outputs remain close on this simple workload without requiring bitwise equality.
    for label in ('bf16_accumulated', 'fp16_accumulated'):
        delta = float((cases[label]['parameter_after'] - reference).abs().max())
        cases[label]['max_abs_diff_vs_fp32_reference'] = delta
        if delta > 0.03:
            raise AssertionError(f'{label} differs unexpectedly from FP32: {delta}')
    result = {
        'status': 'PASS', 'gpu_name': gpu_name,
        'cuda_device_capability': list(torch.cuda.get_device_capability(device)),
        'torch_version': torch.__version__, 'cuda_runtime': torch.version.cuda,
        'hostname': socket.gethostname(), 'python_version': platform.python_version(),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'fp32_max_abs_parameter_diff': max_abs,
        'cases': {},
        'scope': 'synthetic CUDA acceptance only; not dataset-backed run/resume',
    }
    for label, item in cases.items():
        result['cases'][label] = {k: v for k, v in item.items() if k != 'parameter_after'}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(f'PASS: GPU acceptance report saved to {path}', flush=True)


if __name__ == '__main__':
    main()
