"""One-batch, dataset-free T04-03 GPU feasibility test (not an experiment run).

Use one model/strategy per process. Never writes into prepared experiment folders.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from models.model_clip import create_model as clip_factory
from models.model_siglip import create_model as siglip_factory

CHECKPOINTS = {
    'clip': ('clip_vit_b16', 'openai/clip-vit-base-patch16', clip_factory),
    'siglip': ('siglip_vit_b16', 'google/siglip-base-patch16-224', siglip_factory),
}


def _digest(model):
    # Compact deterministic check of the complete state; includes buffers.
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode('utf-8'))
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _optimizer_step(factory, name, strategy, revision, channels, batch_size, device, lr, checkpoint_test):
    backbone, checkpoint, build = factory[name]
    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)
    model, optimizer = build(
        backbone=backbone, input_shape=(channels, 32, 32),
        dataset_type='moment', freeze_backbone=(strategy == 'S1'),
        pretrained=(strategy != 'S4'), strategy=strategy,
        checkpoint=checkpoint, revision=revision, learning_rate=lr)
    model.to(device)
    model.train()
    x = torch.rand(batch_size, channels, 32, 32, device=device) * 2 - 1
    a = torch.arange(batch_size, device=device) % 7
    q = torch.arange(batch_size, device=device) % 5
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    t0 = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    out = model(x)
    assert out['alpha_output'].shape == (batch_size, 7)
    assert out['q0_output'].shape == (batch_size, 5)
    loss = F.cross_entropy(out['alpha_output'], a) + F.cross_entropy(out['q0_output'], q)
    assert torch.isfinite(loss).item()
    loss.backward()
    assert model.input_adapter.weight.grad is not None
    assert torch.isfinite(model.input_adapter.weight.grad).all().item()
    backbone_grads = sum(p.grad is not None for p in model.backbone.parameters())
    assert (backbone_grads == 0) if strategy == 'S1' else (backbone_grads > 0)
    assert all(p.grad is None for p in model.backbone.parameters() if not p.requires_grad)
    optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.monotonic() - t0
    peak_alloc = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    report = {
        'status':'PASS', 'model':name, 'strategy':strategy, 'channels':channels,
        'batch_size':batch_size, 'revision':revision, 'checkpoint':checkpoint,
        'pretrained':strategy != 'S4', 'device':torch.cuda.get_device_name(device),
        'total_parameters':sum(p.numel() for p in model.parameters()),
        'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'backbone_grad_parameter_tensors':backbone_grads,
        'loss':float(loss.detach()), 'step_seconds':round(elapsed, 4),
        'peak_allocated_gib':round(peak_alloc / (1024**3), 4),
        'peak_reserved_gib':round(peak_reserved / (1024**3), 4),
        'gpu_total_gib':round(torch.cuda.get_device_properties(device).total_memory / (1024**3), 4),
    }
    if checkpoint_test:
        # GPU-independent checkpoint/optimizer serialization roundtrip; does not
        # initialize a second giant backbone in GPU memory.
        import io
        state = {'model':model.state_dict(), 'optimizer':optimizer.state_dict()}
        digest_before = _digest(model)
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)
        try:
            loaded = torch.load(buffer, map_location='cpu', weights_only=True)
        except TypeError:
            loaded = torch.load(buffer, map_location='cpu')
        assert loaded['model'].keys() == state['model'].keys()
        assert loaded['optimizer']['param_groups'] == state['optimizer']['param_groups']
        for key, tensor in model.state_dict().items():
            assert torch.equal(tensor.detach().cpu(), loaded['model'][key]), key
        assert _digest(model) == digest_before
        report['checkpoint_roundtrip'] = 'PASS (state_dict & optimizer serialization; not full resume)'
    return report


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required: submit the Slurm job on an allocated GPU')
    device = torch.device('cuda:0')
    try:
        return _optimizer_step(CHECKPOINTS, args.model, args.strategy, args.revision,
                               args.channels, args.batch_size, device,
                               args.learning_rate, args.checkpoint_roundtrip)
    except torch.cuda.OutOfMemoryError as ex:
        torch.cuda.empty_cache()
        return {'status':'OOM', 'model':args.model, 'strategy':args.strategy,
                'batch_size':args.batch_size, 'channels':args.channels,
                'gpu_total_gib':round(torch.cuda.get_device_properties(device).total_memory/(1024**3),4),
                'error':str(ex)[:700]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=CHECKPOINTS)
    parser.add_argument('--strategy', required=True, choices=('S1','S2','S3','S4'))
    parser.add_argument('--channels', type=int, choices=(1,2,3,4,5), default=5)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--checkpoint-roundtrip', action='store_true')
    parser.add_argument('--output', help='New JSON result path outside prepared runs (exclusive creation)')
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.learning_rate <= 0:
        parser.error('batch-size and learning-rate must be positive')
    if len(args.revision) != 40 or any(c not in '0123456789abcdef' for c in args.revision.lower()):
        parser.error('revision must be a 40-character commit SHA')
    report = run(args)
    body = json.dumps(report, indent=2) + '\n'
    if args.output:
        out = Path(args.output).expanduser().resolve()
        # Prevent accidental writes to a prepared run. Output belongs in a new
        # dedicated results directory configured by the caller.
        if 'sessions' in out.parts or 'source' in out.parts:
            parser.error('output may not target source/ or sessions/')
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('x') as f:
            f.write(body)
    print(body, flush=True)
    return 0 if report['status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
