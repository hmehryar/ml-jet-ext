#!/usr/bin/env python3
"""Generate five independent cumulative-moment experiment configs."""
from pathlib import Path
import argparse
import yaml

DEFAULT_CHECKPOINT = ("/wsu/home/gy/gy40/gy4065/ml-jet-param-predictor/training_output/"
    "ViT_tiny_patch16_224_gaussian_g500_bs32_ep50_lr1e-04_ds7200000_g500_"
    "sched_ReduceLROnPlateau/best_model.pth")
DEFAULT_DATASET = ("/rs/rs_grp_mljet/ml-jet-ext/"
    "pTHat50-250_eCM5020_c0005_pT90-110_R0p4_agg500_m1-5_normalized")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='experiments/exp_03_cumulative_moment_ablation')
    p.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    p.add_argument('--dataset-root', default=DEFAULT_DATASET)
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--workers', type=int, default=1)
    args = p.parse_args()
    root = Path(args.root)
    (root / 'config').mkdir(parents=True, exist_ok=True)
    (root / 'training_output').mkdir(parents=True, exist_ok=True)
    for k in range(1, 6):
        name = f'E{k}_m1_pretrained' if k == 1 else f'E{k}_m1_m{k}_pretrained'
        cfg = {
            'model_tag': name,
            'backbone': 'vit_gaussian',
            'dataset_type': 'moment',
            'moments': list(range(1, k + 1)),
            'moment_fold': None,
            'dataset_root': args.dataset_root,
            'dataset_size': 14000,
            'group_size': 1,
            'batch_size': args.batch_size,
            'num_workers': args.workers,
            'pin_memory': True,
            'seed': args.seed,
            'epochs': args.epochs,
            'learning_rate': args.lr,
            'patience': 100,
            'initialization': {'type': 'gaussian', 'mean': 0.0, 'std': 0.02},
            'transfer_mode': 'jetbench_full',
            'checkpoint_path': args.checkpoint,
            'output_dir': str(root / 'training_output'),
            'scheduler': {'type': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.5, 'patience': 15, 'min_lr': 0.0000001},
            'loss': {'weights': {'alpha_output': 1.0, 'q0_output': 1.0}},
        }
        target = root / 'config' / (name + '.yml')
        with target.open('w') as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(target)

if __name__ == '__main__':
    main()
