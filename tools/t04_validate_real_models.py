"""T04-03 Stage 3: validate ACTUAL Hugging Face CLIP/SigLIP vision encoders.

Run one combination per process (prefer --device cpu on a login/interactive node).
No dataset, optimizer steps, or training outputs are modified.
"""
import argparse
import gc
import json
from pathlib import Path
import sys

import torch

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
    
from models.model_clip import create_model as clip_factory
from models.model_siglip import create_model as siglip_factory
from models.t04_strategies import trainable_parameter_counts

CHECKPOINTS = {
    'clip': ('clip_vit_b16', 'openai/clip-vit-base-patch16', clip_factory),
    'siglip': ('siglip_vit_b16', 'google/siglip-base-patch16-224', siglip_factory),
}


def validate(model_name, strategy, channels, device, revision=None, output=None, backward=True):
    if channels not in range(1, 6):
        raise ValueError('channels must be in 1..5')
    backbone_name, checkpoint, factory = CHECKPOINTS[model_name]
    pretrained = strategy != 'S4'
    torch.manual_seed(42)
    model, optimizer = factory(backbone=backbone_name,
        input_shape=(channels, 32, 32), strategy=strategy,
        freeze_backbone=(strategy == 'S1'), pretrained=pretrained,
        checkpoint=checkpoint, revision=revision, learning_rate=1e-4)
    assert model.strategy == strategy
    assert model.pretrained_backbone == pretrained
    assert model.input_adapter.weight.shape == (3, channels, 1, 1)
    assert model.backbone.config.image_size == 224
    vision = model.backbone.vision_model
    layers = vision.encoder.layers
    assert len(layers) >= 2
    expected = {id(p) for p in model.input_adapter.parameters()}
    expected.update(id(p) for p in model.alpha_head.parameters())
    expected.update(id(p) for p in model.q0_head.parameters())
    if strategy == 'S2':
        for layer in list(layers)[-2:]:
            expected.update(id(p) for p in layer.parameters())
        for key in ('post_layernorm', 'head'):
            module = getattr(vision, key, None)
            if module is not None:
                expected.update(id(p) for p in module.parameters())
    elif strategy in ('S3', 'S4'):
        expected.update(id(p) for p in model.backbone.parameters())
    actual = {id(p) for p in model.parameters() if p.requires_grad}
    assert actual == expected, 'trainable parameter mask differs from expected architecture policy'
    optim = [id(p) for g in optimizer.param_groups for p in g['params']]
    assert len(optim) == len(set(optim)) and set(optim) == actual
    counts = trainable_parameter_counts(model)
    assert counts['total'] >= counts['trainable'] > 0
    if strategy == 'S1':
        assert counts['backbone_trainable'] == 0
        assert not model.backbone.training or model.freeze_backbone
    elif strategy in ('S3', 'S4'):
        assert counts['backbone_trainable'] == counts['backbone_total']
    else:
        assert 0 < counts['backbone_trainable'] < counts['backbone_total']
    model.to(device)
    model.train()

    if strategy == 'S1':
        assert not model.backbone.training, \
            'S1 frozen backbone must remain in eval mode'

    elif strategy == 'S2':
        assert not model.backbone.training, \
            'S2 frozen backbone container should remain in eval mode'

        vision = model.backbone.vision_model
        layers = list(vision.encoder.layers)

        # Frozen transformer blocks must remain in eval mode.
        assert all(not layer.training for layer in layers[:-2])

        # The final two trainable blocks must be in training mode.
        assert all(layer.training for layer in layers[-2:])

        # Trainable final normalization/pooling modules.
        for name in ('post_layernorm', 'head'):
            module = getattr(vision, name, None)
            if module is not None:
                assert module.training, f'{name} should be in training mode'

    else:  # S3 and S4
        assert model.backbone.training, \
            'Fully trainable backbone must be in training mode'
    gradient = {'checked': backward}
    if backward:
        # Signed channel input (including negative M3/M5 values) remains un-clipped.
        x = torch.linspace(-1., 1., channels * 32 * 32,
                           device=device).reshape(1, channels, 32, 32)
        out = model(x)
        assert out['alpha_output'].shape == (1, 7)
        assert out['q0_output'].shape == (1, 5)
        (out['alpha_output'].square().mean() + out['q0_output'].square().mean()).backward()
        assert model.input_adapter.weight.grad is not None
        assert torch.isfinite(model.input_adapter.weight.grad).all()
        assert model.input_adapter.weight.grad.abs().sum().item() > 0
        selected_grad = [n for n, p in model.backbone.named_parameters() if p.grad is not None]
        if strategy == 'S1':
            assert not selected_grad
        else:
            assert selected_grad, 'Trainable backbone had no gradients'
        assert all(p.grad is None for p in model.backbone.parameters() if not p.requires_grad)
        gradient['backbone_grad_parameter_tensors'] = len(selected_grad)
    report = {'model': model_name, 'strategy': strategy, 'channels': channels,
        'checkpoint': checkpoint, 'revision': revision, 'pretrained': pretrained,
        'device': str(device), 'transformer_layers': len(layers),
        'unfrozen_transformer_indices': model.unfrozen_transformer_indices,
        'counts': counts, 'gradient': gradient, 'status': 'PASS'}
    if output:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    del model, optimizer
    gc.collect()
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=CHECKPOINTS, required=True)
    p.add_argument('--strategy', choices=('S1','S2','S3','S4'), required=True)
    p.add_argument('--channels', type=int, choices=range(1,6), default=5)
    p.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    p.add_argument('--revision', help='Pinned 40-hex Hugging Face revision for audit reproducibility')
    p.add_argument('--output', help='New JSON path; existing file is never overwritten')
    p.add_argument('--skip-backward', action='store_true')
    args = p.parse_args()
    if args.revision and (len(args.revision) != 40 or any(c not in '0123456789abcdefABCDEF' for c in args.revision)):
        p.error('--revision must be a full 40-character commit SHA')
    validate(args.model, args.strategy, args.channels, torch.device(args.device),
             args.revision, args.output, not args.skip_backward)


if __name__ == '__main__':
    main()
