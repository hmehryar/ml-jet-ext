#!/usr/bin/env python3
"""T02.5 one FULL training epoch, full validation, and checkpoint round trip.
Run from repo root: PYTHONPATH="$PWD" python tests/validate_t02_5_full_epoch.py ...
Does not read or evaluate the held-out test split.
"""
import argparse
import copy
import csv
from pathlib import Path

import torch

from config import get_config
from data.loader import get_dataloaders
from train_utils.training_loop import build_model_optimizer_and_criterion_from_cfg
from train_utils.moment_training import run_moment_training_loop


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # older PyTorch releases
        return torch.load(path, map_location=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config/config_moment_example.yaml')
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--output', default='training_output/t02_5_validation')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--moments', type=int, nargs='+', default=None)
    args = ap.parse_args()

    cfg = get_config(args.config)
    assert cfg.dataset_type == 'moment'
    cfg.epochs = 1
    cfg.patience = max(2, int(cfg.patience))
    cfg.num_workers = args.workers
    cfg.batch_size = args.batch_size
    if args.moments:
        assert len(set(args.moments)) == len(args.moments)
        assert all(1 <= x <= 5 for x in args.moments)
        cfg.moments = args.moments
        cfg.input_shape = (len(args.moments), 32, 32)
    cfg.output_dir = str(Path(args.output).resolve())
    output = Path(cfg.output_dir)
    if any((output / name).exists() for name in ('last_model.pth', 'best_model.pth', 'moment_epoch_metrics.csv')):
        raise FileExistsError(f'Output directory contains previous results: {output}. Choose a fresh --output')

    if not torch.cuda.is_available():
        raise RuntimeError('T02.5 full epoch must run in a CUDA-allocated GPU job')
    device = torch.device('cuda:0')
    torch.manual_seed(int(getattr(cfg, 'seed', 42)))
    torch.cuda.manual_seed_all(int(getattr(cfg, 'seed', 42)))
    train_loader, val_loader, test_loader = get_dataloaders(cfg, device=device)
    assert tuple(map(lambda loader: len(loader.dataset), (train_loader, val_loader, test_loader))) == (11200, 1400, 1400)
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    print('Moments:', cfg.moments, 'train/val/test:', 11200, 1400, 1400, flush=True)
    model, criterion, optimizer = build_model_optimizer_and_criterion_from_cfg(cfg, device)
    best_epoch, best_acc, _ = run_moment_training_loop(
        cfg, train_loader, val_loader, device, model, criterion, optimizer,
        scheduler=None, start_epoch=0,
    )
    assert best_epoch == 1
    assert 0 <= best_acc <= 1

    metrics_csv = output / 'moment_epoch_metrics.csv'
    with metrics_csv.open(newline='') as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    assert int(rows[0]['epoch']) == 1
    for key in ('train_loss', 'val_loss', 'val_alpha_accuracy', 'val_q0_accuracy', 'val_joint_accuracy'):
        assert torch.isfinite(torch.tensor(float(rows[0][key]))), key

    last = load_checkpoint(output / 'last_model.pth', device)
    best = load_checkpoint(output / 'best_model.pth', device)
    assert last['epoch'] == best['epoch'] == 1
    assert last['dataset_type'] == best['dataset_type'] == 'moment'
    assert list(last['moments']) == list(cfg.moments)
    assert 'optimizer_state_dict' in last
    assert bool(last['optimizer_state_dict']['state']), 'Optimizer must have state after training'

    # Independent model/optimizer reconstruction from the saved weights.
    restored_model, _, restored_optimizer = build_model_optimizer_and_criterion_from_cfg(cfg, device)
    restored_model.load_state_dict(last['model_state_dict'], strict=True)
    restored_optimizer.load_state_dict(last['optimizer_state_dict'])
    assert len(restored_optimizer.state_dict()['state']) == len(last['optimizer_state_dict']['state'])
    original_params = dict(model.named_parameters())
    restored_params = dict(restored_model.named_parameters())
    for name in original_params:
        torch.testing.assert_close(original_params[name].detach(), restored_params[name].detach(), rtol=0, atol=0)

    # Check prediction identity on the SAME validation batch, with eval mode.
    batch, _ = next(iter(val_loader))
    batch = batch.to(device)
    model.eval(); restored_model.eval()
    with torch.inference_mode():
        original = model(batch)
        restored = restored_model(batch)
    for head in ('alpha_output', 'q0_output'):
        torch.testing.assert_close(original[head], restored[head], rtol=0, atol=0)
    assert 'energy_loss_output' not in restored

    # Verify the restored optimizer can execute a new step without changing the saved file.
    train_images, train_labels = next(iter(train_loader))
    train_images = train_images.to(device)
    restored_model.train()
    predictions = restored_model(train_images)
    total_loss = sum(criterion[head](predictions[head], train_labels[head].to(device))
                     * float(getattr(cfg, 'loss_weights', {}).get(head, 1.0))
                     for head in ('alpha_output', 'q0_output'))
    restored_optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    restored_optimizer.step()

    print('PASS: full 11,200-sample GPU training epoch', flush=True)
    print('PASS: full 1,400-sample validation epoch', flush=True)
    print('PASS: metrics CSV and best/last checkpoints', flush=True)
    print('PASS: strict model/optimizer restoration and identical predictions', flush=True)
    print('PASS: resumed optimizer step', flush=True)
    print('Held-out test set: not evaluated', flush=True)
    print('Metrics:', rows[0], flush=True)

if __name__ == '__main__':
    main()
