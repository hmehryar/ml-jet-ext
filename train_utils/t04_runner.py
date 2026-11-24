"""T04 vision-encoder runner; supports legacy pilots and verified prepared S1 runs."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import importlib.metadata
import subprocess
import sys
from datetime import datetime, timezone

import torch
import yaml

from data.moment_loader import get_moment_dataloaders
from models.model_clip import create_model as create_clip
from models.model_siglip import create_model as create_siglip
from train_utils.ordinal_loss import build_moment_criterion
from train_utils.moment_training import run_moment_training_loop
from train_utils.t03_reporting import generate_t03_report

FACTORIES = {'clip': (create_clip, 'clip_vit_b16'), 'siglip': (create_siglip, 'siglip_vit_b16')}


def load_config(path):
    with open(path, encoding='utf8') as f:
        settings = yaml.safe_load(f)
    if not isinstance(settings, dict):
        raise ValueError('YAML configuration must be a mapping')
    if settings.get('model') not in FACTORIES:
        raise ValueError('Unsupported T04 model')
    moments = settings.get('moments')
    if (not isinstance(moments, list) or not moments or
        any(type(m) is not int or m not in range(1, 6) for m in moments) or
        len(set(moments)) != len(moments)):
        raise ValueError('moments must contain distinct integers 1..5')
    for key in ('dataset_root', 'output_dir'):
        if not settings.get(key):
            raise ValueError(f'Missing required setting: {key}')
    return settings


def _runtime_details(model, device):
    pkg = {}
    for p in ('torch', 'transformers', 'numpy', 'PyYAML'):
        try:
            pkg[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            pkg[p] = None
    gpu = None
    if device.type == 'cuda':
        prop = torch.cuda.get_device_properties(device)
        gpu = {'name': prop.name, 'total_memory_bytes': prop.total_memory,
               'compute_capability': list(torch.cuda.get_device_capability(device)),
               'cuda_runtime': torch.version.cuda, 'cudnn': torch.backends.cudnn.version()}
        try:
            gpu['nvidia_smi'] = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=name,uuid,memory.total,driver_version',
                 '--format=csv,noheader'], text=True, timeout=15).strip()
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            gpu['nvidia_smi'] = None
    return {
        'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version, 'packages': pkg, 'gpu': gpu,
        'slurm': {name: os.environ.get(name) for name in
                  ('SLURM_JOB_ID', 'SLURM_JOB_PARTITION', 'SLURM_JOB_QOS',
                   'SLURM_JOB_GPUS', 'SLURM_STEP_GPUS', 'CUDA_VISIBLE_DEVICES')},
        'parameters': {'total': sum(p.numel() for p in model.parameters()),
                       'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad)},
        'backbone_parameters': {'total': sum(p.numel() for p in model.backbone.parameters()),
                                'trainable': sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)},
        'checkpoint': getattr(model, 'checkpoint_name', None),
        'checkpoint_revision': getattr(model, 'checkpoint_revision', None),
    }


def _from_prepared(path):
    """Validate before importing model factories via the executing snapshot."""
    from train_utils.t04_preflight import verify
    out = Path(path).expanduser().resolve()
    verify(out)
    source = out / 'source'
    # Refuse invocation from the mutable checkout, even if its files are identical.
    if Path(__file__).resolve() != source / 'train_utils/t04_runner.py':
        raise RuntimeError('Prepared T04 runs must execute the frozen source snapshot')
    cfg = json.loads((out / 'config.resolved.json').read_text())
    # Explicit, fail-closed feature gates until T04-02/03/04 implement them.
    if cfg['strategy'] != 'S1':
        raise NotImplementedError('S2/S3/S4 training is implemented in T04-03, not T04-01')
    if cfg['precision'] != 'fp32' or cfg['gradient_accumulation'] != 1:
        raise NotImplementedError('Mixed precision/accumulation belong to T04-04')
    from train_utils.loss_config import resolve_moment_loss
    resolve_moment_loss(cfg['moment_loss'])
    rates = cfg['learning_rates']
    if rates['adapter'] != rates['heads']:
        raise NotImplementedError('Parameter-group LR belongs to T04-04')
    if cfg['scheduler'].get('type') != 'ReduceLROnPlateau':
        raise ValueError('Prepared runs require RLRP')
    if cfg['min_epochs'] > 1:
        raise NotImplementedError('Minimum epoch gate belongs to T04-02')
    return {
        'model': cfg['model'], 'moments': cfg['moments'], 'dataset_root': cfg['dataset_root'],
        'output_dir': str(out), 'seed': cfg['seed'], 'batch_size': cfg['batch_size'],
        'num_workers': cfg['num_workers'], 'epochs': cfg['epochs'], 'patience': cfg['patience'],
        'learning_rate': rates['heads'], 'freeze_backbone': True,
        'moment_loss': cfg['moment_loss'],
        '_prepared': True, '_checkpoint': cfg['checkpoint'],
        '_checkpoint_revision': cfg['checkpoint_revision'],
        '_scheduler': cfg['scheduler'], '_strategy': cfg['strategy']
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--config')
    group.add_argument('--prepared-run')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    if args.prepared_run and args.check:
        parser.error('--check is only available with --config')
    settings = _from_prepared(args.prepared_run) if args.prepared_run else load_config(args.config)
    root = Path(settings['dataset_root']).expanduser().resolve()
    out = Path(settings['output_dir']).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Dataset root missing: {root}')
    for name in ('train.csv', 'val.csv', 'test.csv'):
        if not (root / 'splits' / name).is_file():
            raise FileNotFoundError(root / 'splits' / name)
    if not settings.get('_prepared') and out.exists():
        raise FileExistsError(f'Output directory already exists: {out}')
    if args.check:
        print('T04 configuration: PASS')
        print('Model:', settings['model'])
        print('Moments:', settings['moments'])
        print('Dataset:', root)
        print('Output:', out)
        return
    if not torch.cuda.is_available():
        raise RuntimeError('T04 GPU training requires CUDA')
    device = torch.device('cuda')
    seed = int(settings.get('seed', 42))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    moments = settings['moments']
    train_loader, val_loader, _ = get_moment_dataloaders(
        dataset_root=root, moments=moments,
        batch_size=int(settings.get('batch_size', 4)),
        num_workers=int(settings.get('num_workers', 0)),
        seed=seed, validate_files=True, check_normalization=True)
    factory, backbone = FACTORIES[settings['model']]
    kwargs = {}
    if settings.get('_prepared'):
        kwargs = {'checkpoint': settings['_checkpoint'],
                  'revision': settings['_checkpoint_revision']}
    model, optimizer = factory(
        backbone=backbone, input_shape=(len(moments), 32, 32),
        dataset_type='moment', freeze_backbone=bool(settings.get('freeze_backbone', True)),
        pretrained=True, learning_rate=float(settings.get('learning_rate', 1e-4)), **kwargs)
    model = model.to(device)
    if settings.get('_prepared'):
        # No existing training products may be overwritten. Reservation files are allowed.
        for name in ('best_model.pth', 'last_model.pth', 'moment_epoch_metrics.csv', '.training.lock'):
            if (out / name).exists():
                raise FileExistsError(f'Prepared run already used: {out / name}')
        runtime = _runtime_details(model, device)
        (out / 'runtime_environment.json').write_text(json.dumps(runtime, indent=2))
    criterion = build_moment_criterion(settings.get('moment_loss', {'type': 'cross_entropy'}), device=device)
    cfg = SimpleNamespace(output_dir=str(out), epochs=int(settings.get('epochs', 2)),
                          patience=int(settings.get('patience', 5)), moments=moments,
                          loss_weights={'alpha_output': 1.0, 'q0_output': 1.0},
                          moment_loss=settings.get('moment_loss', {'type': 'cross_entropy'}))
    scheduler = None
    if settings.get('_prepared'):
        from train_utils.t03_runtime import resolve_scheduler
        spec = resolve_scheduler(settings['_scheduler'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=spec['mode'], factor=spec['factor'],
            patience=spec['patience'], min_lr=spec['min_lr'])
        cfg.scheduler = spec
    elif not out.exists():
        out.mkdir(parents=True)
        (out / 'config.resolved.json').write_text(json.dumps(settings, indent=2))
    best_epoch, best_acc, _ = run_moment_training_loop(
        cfg, train_loader, val_loader, device, model, criterion, optimizer, scheduler=scheduler)
    print('Best epoch:', best_epoch)
    print('Best validation joint accuracy:', best_acc)
    generate_t03_report(model, val_loader, device, out,
                        expected_samples=len(val_loader.dataset))
    print('T04 training completed successfully.')


if __name__ == '__main__':
    main()
