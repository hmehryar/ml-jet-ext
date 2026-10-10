"""One-batch T02.4 smoke test; run from repo root with python -m tests.smoke_t02_4_real_batch
or PYTHONPATH="$PWD" python tests/smoke_t02_4_real_batch.py.
"""
import argparse
import torch
from config import get_config
from data.loader import get_dataloaders
from train_utils.training_loop import build_model_optimizer_and_criterion_from_cfg
from train_utils.train_epoch import train_one_epoch
from train_utils.evaluate import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config/config_moment_example.yaml')
    p.add_argument('--workers', type=int, default=0)
    args = p.parse_args()
    cfg = get_config(args.config)
    if cfg.dataset_type != 'moment':
        raise ValueError('Smoke script expects moment configuration')
    cfg.num_workers = args.workers
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, _ = get_dataloaders(cfg, device=device)
    model, criterion, optimizer = build_model_optimizer_and_criterion_from_cfg(cfg, device)
    # One batch from each; train_one_epoch/evaluate receive finite iterables.
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    print('Device:', device)
    print('Train tensor:', tuple(train_batch[0].shape))
    train = train_one_epoch([train_batch], model, criterion, optimizer, device,
                            loss_weights=cfg.loss_weights, dataset_type='moment')
    val = evaluate([val_batch], model, criterion, device,
                   loss_weights=cfg.loss_weights, dataset_type='moment')
    print('Train:', {k: round(v, 4) for k, v in train.items() if isinstance(v, float)})
    print('Validation:', {k: round(v, 4) for k, v in val.items() if isinstance(v, float)})
    assert 'energy' not in val and 'loss_energy' not in train
    print('PASS: real-data one-batch training and validation')

if __name__ == '__main__':
    main()
