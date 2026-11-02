"""Isolated, dataset-backed T04-05 microbatch feasibility probe.

Run one model/strategy/batch per process; never reserve a T04 session.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import time
import traceback


def classify_failure(exc):
    message = str(exc).lower()
    if 'out of memory' in message or 'cublas_status_alloc_failed' in message:
        return 'oom'
    return 'error'


def probe(config, batch_size, steps):
    import torch
    from data.moment_loader import get_moment_dataloaders
    from train_utils.t04_preflight import preflight
    from train_utils.t04_runner import FACTORIES
    from train_utils.t04_session_worker import build_strategy_model
    from train_utils.ordinal_loss import build_moment_criterion
    from train_utils.t04_precision import autocast_context

    cfg, audit = preflight(config)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 unsupported on this GPU')
    if steps < 1:
        raise ValueError('steps must be positive')
    device = torch.device('cuda')
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    c = dict(cfg.__dict__)
    c['batch_size'] = batch_size
    c['gradient_accumulation'] = 1
    loader, _, _ = get_moment_dataloaders(
        dataset_root=cfg.dataset_root, moments=cfg.moments, batch_size=batch_size,
        num_workers=cfg.num_workers, seed=cfg.seed,
        validate_files=True, check_normalization=True)
    model, optimizer = build_strategy_model(c, FACTORIES)
    model.to(device)
    criterion = build_moment_criterion(cfg.moment_loss, device=device)
    model.train()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    samples = 0
    for idx, (images, targets) in enumerate(loader):
        if idx >= steps:
            break
        optimizer.zero_grad(set_to_none=True)
        x = images.to(device, non_blocking=True)
        y_alpha = targets['alpha_output'].to(device, non_blocking=True)
        y_q0 = targets['q0_output'].to(device, non_blocking=True)
        with autocast_context(device, 'bf16'):
            output = model(x)
            loss = criterion['alpha_output'](output['alpha_output'], y_alpha) + criterion['q0_output'](output['q0_output'], y_q0)
        if not bool(torch.isfinite(loss).all()):
            raise FloatingPointError('Nonfinite training loss')
        loss.backward()
        optimizer.step()
        samples += x.shape[0]
        del x, output, loss, y_alpha, y_q0
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if not samples:
        raise RuntimeError('No training samples processed')
    return dict(status='pass', model=cfg.model, strategy=cfg.strategy,
                split_protocol=cfg.split_protocol, source_overlap=audit['global_cross_split_overlap'],
                batch_size=batch_size, steps=steps, samples=samples,
                elapsed_seconds=elapsed, samples_per_second=samples / elapsed,
                cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
                cuda_total_bytes=int(torch.cuda.get_device_properties(device).total_memory),
                gpu_name=torch.cuda.get_device_name(device))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--batch-size', required=True, type=int)
    p.add_argument('--steps', default=8, type=int)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    if a.batch_size not in (2, 4, 8):
        p.error('batch size must be 2, 4, or 8')
    path = Path(a.output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Refuse to overwrite evidence, even from a previous failed probe.
    if path.exists():
        p.error('report already exists: ' + str(path))
    status = 0
    try:
        result = probe(a.config, a.batch_size, a.steps)
    except Exception as exc:
        result = dict(status=classify_failure(exc), config=str(Path(a.config).resolve()),
                      batch_size=a.batch_size, steps=a.steps,
                      error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc(limit=12))
        status = 10 if result['status'] == 'oom' else 1
    # Atomic report creation; no partial JSON artifacts.
    with path.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    print('T04-05_PROBE', json.dumps({k:v for k,v in result.items() if k != 'traceback'}, sort_keys=True), flush=True)
    raise SystemExit(status)


if __name__ == '__main__':
    main()
