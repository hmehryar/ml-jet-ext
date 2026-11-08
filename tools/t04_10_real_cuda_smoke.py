"""Real-checkpoint CUDA/BF16 one-batch smoke test, no jobs or dataset mutations."""
import argparse
import json
import torch
from data.moment_loader import get_moment_dataloaders
from models.t04_10_single_head import build_clip_siglip_single_head, build_trainable_optimizer, TASK_OUTPUT
from train_utils.t04_10_preflight import load_settings, inspect_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    c = load_settings(args.config)
    inspect_inputs(c)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for real pretrained-backbone smoke test')
    if c['precision'] == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 unavailable on this CUDA device')
    tr, _val, _test = get_moment_dataloaders(
        c['dataset_root'], moments=c['moments'], batch_size=1, num_workers=0,
        seed=c['seed'], pin_memory=False, validate_files=True, check_normalization=True)
    model = build_clip_siglip_single_head(
        architecture=c['model'], task=c['task'], channels=len(c['moments']),
        strategy=c['strategy'], checkpoint=c['checkpoint'],
        revision=c['checkpoint_revision'], pretrained=(c['strategy'] != 'S4')).cuda()
    opt = build_trainable_optimizer(model, lr_backbone=c['learning_rates']['backbone'],
        lr_adapter=c['learning_rates']['adapter'], lr_head=c['learning_rates']['heads'],
        weight_decay=c['optimizer']['weight_decay'])
    x, targets = next(iter(tr))
    x = x.cuda()
    y = targets[TASK_OUTPUT[c['task']]].cuda()
    model.train()
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(c['precision'] == 'bf16')):
        outputs = model(x)
        logits = outputs[TASK_OUTPUT[c['task']]]
        loss = torch.nn.functional.cross_entropy(logits.float(), y)
    if set(outputs) != {TASK_OUTPUT[c['task']]} or not torch.isfinite(loss):
        raise RuntimeError('Head isolation or finite loss validation failed')
    loss.backward()
    if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise RuntimeError('Nonfinite gradient')
    opt.step()
    print(json.dumps({'status':'PASS', 'model':c['model'], 'task':c['task'],
                      'shape':list(logits.shape), 'loss':float(loss.detach()),
                      'max_memory_allocated':torch.cuda.max_memory_allocated()}))

if __name__ == '__main__':
    main()
