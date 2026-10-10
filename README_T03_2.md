# T03.2 — ViT-Tiny historical checkpoint transfer pilot

This is a self-contained, non-invasive pilot: **no existing repository files are overwritten**.
It constructs the existing two-head moment ViT with the checkpoint's original **3 input channels**, adds a trainable `Conv2d(C,3,1,bias=False)` input adapter, transfers only shape-matched `backbone.*`/`features.*` state entries, and verifies coverage before performing a forward/backward optimizer step for cumulative M1 through M1–M5.

The adapter initially copies M1 into each of the 3 channels; all moment channels can learn to contribute during fine-tuning. The experiment must use this **same adapter architecture** for scratch/Gaussian controls, to avoid input architecture confounding.

Run from repo root:

```bash
PYTHONPATH="$PWD" python tests/smoke_t03_2_transfer.py \
  --checkpoint "/wsu/home/gy/gy40/gy4065/ml-jet-param-predictor/training_output/ViT_tiny_patch16_224_gaussian_g500_bs32_ep50_lr1e-04_ds7200000_g500_sched_ReduceLROnPlateau/best_model.pth" \
  --device auto
```

**Guardrail:** the script fails if less than 90% of target backbone tensor *elements* match both name and shape; this is not a guarantee that source and target preprocessing or all architectural semantics are identical. It does not load legacy task heads or optimizer state. It does not yet integrate with the main training runner. Read-only checkpoint inspection; only random synthetic input is processed.

The historical checkpoint is a **ViT-Tiny**, not ViT-CoMer. Do not call its weights ViT-CoMer weights.
