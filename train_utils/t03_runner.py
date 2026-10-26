"""Execute a prepared T03 snapshot. Imported by the maintained notebook."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import socket
import subprocess
import sys
from types import SimpleNamespace

from train_utils.t03_runtime import sha256, verify_source


def update_manifest(out, **updates):
    path = out / 'run_manifest.json'
    manifest = json.loads(path.read_text())
    manifest.update(updates)
    manifest['updated_utc'] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(path)


def check_execution_host(profile):
    hostname = socket.gethostname().lower()
    if any(token in hostname for token in ('login', 'headnode')):
        raise RuntimeError('Training is forbidden on login nodes')
    if profile == 'hpc' and not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('HPC training requires an allocated Slurm compute job')
    if Path('/wsu').exists() and not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('WSU execution requires an allocated Slurm compute job')


def verify_inputs(manifest):
    for item in manifest['inputs'].values():
        if sha256(item['path']) != item['sha256']:
            raise RuntimeError('Input changed since preparation: ' + item['path'])


def environment(torch, device):
    packages = sorted({dist.metadata['Name']: dist.version
                       for dist in importlib.metadata.distributions() if dist.metadata['Name']}.items())
    driver = None
    if device.type == 'cuda':
        try:
            driver = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version,uuid,name',
                                             '--format=csv,noheader'], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return {'python': sys.version, 'executable': sys.executable,
            'platform': platform.platform(), 'hostname': socket.gethostname(),
            'packages': dict(packages), 'torch': torch.__version__,
            'cuda_runtime': torch.version.cuda, 'driver': driver,
            'cudnn': torch.backends.cudnn.version(),
            'device': str(device), 'gpus': [dict(name=torch.cuda.get_device_name(i),
                memory_bytes=torch.cuda.get_device_properties(i).total_memory,
                capability=list(torch.cuda.get_device_capability(i)))
                for i in range(torch.cuda.device_count())] if device.type == 'cuda' else [],
            'determinism': {'algorithms': torch.are_deterministic_algorithms_enabled(),
                'cudnn_deterministic': torch.backends.cudnn.deterministic,
                'cudnn_benchmark': torch.backends.cudnn.benchmark,
                'cuda_matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
                'cudnn_tf32': torch.backends.cudnn.allow_tf32},
            'execution_environment': {k: v for k, v in os.environ.items()
                if k in ('SLURM_JOB_ID', 'SLURM_JOB_NAME', 'SLURM_JOB_NODELIST',
                   'SLURM_JOB_PARTITION', 'SLURM_JOB_QOS', 'SLURM_JOB_ACCOUNT',
                   'SLURM_CPUS_PER_TASK', 'SLURM_MEM_PER_NODE', 'SLURM_NTASKS',
                   'SLURM_JOB_GPUS', 'SLURM_STEP_GPUS', 'CUDA_VISIBLE_DEVICES',
                   'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                   'PYTHONHASHSEED', 'CUBLAS_WORKSPACE_CONFIG')}}


def run(out):
    out = Path(out).resolve()
    verify_source(out)
    source = out / 'source'
    if Path(__file__).resolve() != source / 'train_utils/t03_runner.py':
        raise RuntimeError('Execute the archived notebook export, not mutable checkout modules')
    cfg = SimpleNamespace(**json.loads((source / 'config.resolved.json').read_text()))
    if Path(cfg.output_dir) != out:
        raise ValueError('Prepared run/output mismatch')
    check_execution_host(cfg.profile)
    # No automatic retries or resume: ownership is exclusive even after failure.
    with (out / '.execution.lock').open('x') as lock:
        lock.write(str(os.getpid()))
    manifest = json.loads((out / 'run_manifest.json').read_text())
    try:
        verify_inputs(manifest)
        import numpy as np
        import torch
        from torch import nn
        from data.loader import get_dataloaders
        from models.model_vit import create_model
        from models.jetbench_transfer import AdaptedMomentViT, read_trusted_jetbench_checkpoint, transfer_backbone
        from train_utils.moment_training import run_moment_training_loop
        if cfg.device != 'auto' and cfg.device != 'cpu' and not cfg.device.startswith('cuda'):
            raise ValueError('Device must be auto, cpu, cuda, or cuda:N')
        if cfg.device == 'auto' and not torch.cuda.is_available():
            raise RuntimeError('T03 auto device requires CUDA; use --device cpu only for development')
        device = torch.device('cuda' if cfg.device == 'auto' else cfg.device)
        if cfg.validation_suite and device.type != 'cuda':
            raise RuntimeError('The validation suite requires an allocated CUDA GPU')
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(cfg.seed)
        env = environment(torch, device)
        update_manifest(out, status='running', environment=env)
        (out / 'requirements.runtime.txt').write_text(''.join(
            name + '==' + version + '\n' for name, version in env['packages'].items()))
        train, val, test = get_dataloaders(cfg, device=device)
        expected = (11200, 1400, 1400) if cfg.moment_fold is None else (10080, 2520, 1400)
        if tuple(len(loader.dataset) for loader in (train, val, test)) != expected:
            raise ValueError('Unexpected published dataset partition sizes')
        checkpoint, state = read_trusted_jetbench_checkpoint(cfg.checkpoint_path)
        transfer_checks = []
        if cfg.validation_suite:
            # Synthetic checkpoint transfer for every cumulative representation.
            for channels in range(1, 6):
                base, unused = create_model(cfg.backbone, (3, 32, 32), cfg.learning_rate, dataset_type='moment')
                report = transfer_backbone(base, state, min_coverage=0.99)
                probe = AdaptedMomentViT(base, channels).to(device).train()
                opt = torch.optim.Adam(probe.parameters(), lr=cfg.learning_rate)
                outputs = probe(torch.rand(2, channels, 32, 32, device=device))
                loss = sum(nn.functional.cross_entropy(outputs[key], torch.zeros(2, dtype=torch.long, device=device))
                           for key in ('alpha_output', 'q0_output'))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if probe.input_adapter.projection.weight.grad is None:
                    raise RuntimeError('Missing adapter gradient')
                opt.step()
                transfer_checks.append(dict(channels=channels, **report))
                del probe, opt, outputs, loss, base, unused
            update_manifest(out, checkpoint_transfer_validation=transfer_checks)
            # Keep the full-epoch initialization independent of the probes.
            random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
            torch.cuda.manual_seed_all(cfg.seed)
        base, unused = create_model(cfg.backbone, (3, 32, 32), cfg.learning_rate, dataset_type='moment')
        del unused
        report = transfer_backbone(base, state, min_coverage=0.99)
        model = AdaptedMomentViT(base, len(cfg.moments)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        criterion = {key: nn.CrossEntropyLoss() for key in ('alpha_output', 'q0_output')}
        spec = cfg.scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
            mode=spec['mode'], factor=float(spec['factor']), patience=int(spec['patience']),
            min_lr=float(spec['min_lr']), threshold=float(spec.get('threshold', 1e-4)),
            threshold_mode=spec.get('threshold_mode', 'rel'), cooldown=int(spec.get('cooldown', 0)),
            eps=float(spec.get('eps', 1e-8)))
        update_manifest(out, transfer=report, source_checkpoint_epoch=int(checkpoint.get('epoch', -1)),
                        split_sizes=list(expected))
        best_epoch, best_acc, best_metrics = run_moment_training_loop(
            cfg, train, val, device, model, criterion, optimizer, scheduler=scheduler)
        verify_inputs(manifest)
        verify_source(out)
        update_manifest(out, status='training_complete', best_epoch=best_epoch,
                        best_joint_accuracy=best_acc, best_metrics=best_metrics,
                        full_epoch_validation=bool(cfg.validation_suite))
        try:
            from train_utils.t03_reporting import generate_t03_report
            generate_t03_report(model, val, device, out, expected_samples=expected[1])
            update_manifest(out, reporting='complete')
        except Exception as exc:
            update_manifest(out, reporting='failed', reporting_error=repr(exc))
            print('Reporting failed; training artifacts are preserved:', exc, flush=True)
        imported = {name: str(Path(module.__file__).resolve().relative_to(source))
                    for name, module in list(sys.modules.items())
                    if getattr(module, '__file__', None)
                    and source in Path(module.__file__).resolve().parents}
        update_manifest(out, status='complete', imported_project_modules=imported)
    except BaseException as exc:
        update_manifest(out, status='failed', error=repr(exc))
        raise


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if '--prepared-run' in argv:
        p = argparse.ArgumentParser()
        p.add_argument('--prepared-run', required=True, type=Path)
        run(p.parse_args(argv).prepared_run)
    else:
        from train_utils.t03_runtime import main as prepare_main
        prepare_main(argv)
