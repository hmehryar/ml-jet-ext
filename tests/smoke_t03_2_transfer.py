#!/usr/bin/env python3
"""Real timm ViT-Tiny checkpoint-transfer and moment-input gradient smoke test."""
import argparse
import json
import torch
from models.model_vit import create_model
from models.jetbench_transfer import AdaptedMomentViT, read_trusted_jetbench_checkpoint, transfer_backbone


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--min-coverage', type=float, default=0.90)
    args = parser.parse_args()
    device = torch.device('cuda' if (args.device == 'auto' and torch.cuda.is_available()) else
                          ('cpu' if args.device == 'auto' else args.device))
    ckpt, state = read_trusted_jetbench_checkpoint(args.checkpoint)
    print('Checkpoint epoch:', ckpt.get('epoch', '<unknown>'))
    print('Checkpoint metrics:', str(ckpt.get('metrics', '<unknown>'))[:500])
    print('Device:', device)
    for channels in range(1, 6):
        # The base keeps 3-channel architecture for exact pretrained compatibility.
        base_model, _ = create_model(backbone='vit_gaussian', input_shape=(3, 32, 32),
                                     learning_rate=1e-4, dataset_type='moment')
        info = transfer_backbone(base_model, state, min_coverage=args.min_coverage)
        model = AdaptedMomentViT(base_model, channels).to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        x = torch.rand(2, channels, 32, 32, device=device)
        y_a = torch.randint(7, (2,), device=device)
        y_q = torch.randint(5, (2,), device=device)
        outputs = model(x)
        assert outputs['alpha_output'].shape == (2, 7)
        assert outputs['q0_output'].shape == (2, 5)
        loss = (torch.nn.functional.cross_entropy(outputs['alpha_output'], y_a)
                + torch.nn.functional.cross_entropy(outputs['q0_output'], y_q))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        assert model.input_adapter.projection.weight.grad is not None
        optimizer.step()
        print(f'PASS: M1-M{channels}; backbone coverage={info["coverage"]:.1%}; '
              f'loaded={info["matched_tensors"]}/{info["target_backbone_tensors"]}; '
              f'loss={loss.item():.4f}', flush=True)
        del model, optimizer, outputs, base_model
    print('PASS: all five cumulative moment representations')


if __name__ == '__main__':
    main()
