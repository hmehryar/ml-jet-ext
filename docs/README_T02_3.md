# T02.3 — Configurable ViT input channels and task heads

## Scope

`models/model_vit.py` adds an opt-in `dataset_type='moment'` mode. The
previous function signature remains positional-compatible and **legacy is the
default**. Legacy behavior continues to use a three-channel timm backbone,
replicate one-channel inputs to RGB, and output energy_loss, alpha and q0.
Moment mode sets `in_chans` to `input_shape[0]`, accepts the actual tensor
channels without replication, and returns only alpha (7) and q0 (5) logits.

This is a standard timm Vision Transformer, **not ViT-CoMer**.

## Apply

From repository root, inspect first:

```bash
unzip -l ML_JET_Ext_T02_3_ViT_Heads.zip
unzip -p ML_JET_Ext_T02_3_ViT_Heads.zip models/model_vit.py | less
cp models/model_vit.py models/model_vit.py.pre_t02_3
unzip -o ML_JET_Ext_T02_3_ViT_Heads.zip -d .
```

## Tests

```bash
python -m pytest tests/test_t02_vit_moment_heads.py -v
```

Synthetic tests stub `timm` to verify model outputs, gradient flow, and
legacy interfaces. Next, test the actual timm implementation with:

```bash
python - <<'PY'
import torch
from models.model_vit import create_model
for c in (1, 2, 5):
    model, _ = create_model('vit_gaussian', (c, 32, 32), 1e-4, dataset_type='moment')
    model.eval()
    with torch.no_grad():
        y = model(torch.rand(1, c, 32, 32))
    assert {k: tuple(v.shape) for k, v in y.items()} == {
        'alpha_output': (1, 7), 'q0_output': (1, 5)}
    print('PASS', c, {k: tuple(v.shape) for k, v in y.items()})
PY
```

## Important: do not launch training yet

Your current `train_utils/training_loop.py` uses `create_model(cfg.backbone,
cfg.input_shape, cfg.learning_rate)` with **no dataset_type** and your
current `train_epoch.py` expects energy-loss labels. We intentionally leave
these unchanged for T02.3. T02.4 will pass `dataset_type` explicitly and
handle two-label losses/evaluation. The moment config should also use
`backbone: vit_gaussian` or another supported `vit_*` value; `backbone: vit`
works as a from-scratch alias in this factory.

## Pretrained model note

`timm` adapts pretrained patch projection weights when `in_chans` is not 3.
That initialization is not equivalent to a trained five-channel model; test
transfer strategies separately in later experiments.
