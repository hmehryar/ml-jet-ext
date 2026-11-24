# T04-04 Stage 2: precision and gradient accumulation

This patch adds a T04-only epoch runner, preserving `train_utils/moment_training.py` for T03. Install files, then run `python tools/apply_t04_04_precision.py` from repo root. Never patch an existing prepared run.

- `fp32`+accumulation 1 continues using the original T04-02 epoch helper to minimize regressions.
- `bf16`: CUDA autocast, **no GradScaler**.
- `fp16`: CUDA autocast and GradScaler, serialized through existing T04-02 checkpoint machinery.
- Accumulation: final short window and unequal samples use `loss * (microbatch_samples / window_samples)`. One backward per microbatch and one optimizer update per window.
- Reports `optimizer_steps`, `microbatches`, `samples`, and `effective_batch_mean` in the returned train metrics. Stage 3 will persist these plus wall-clock and hardware telemetry into the checkpoint/CSV schema.
- Validation uses the same autocast but no optimizer step.
- The shared T04-02 epoch CSV layout remains unchanged to preserve established resume behavior. A fresh prepared run is required to exercise mixed precision.

## Test

```bash
python tools/apply_t04_04_precision.py
python -m pytest -q tests/test_t04_04_precision.py tests/test_t04_session_training.py tests/test_t04_resume_state.py tests/test_t04_strategy_session_worker.py
```

## Limitations

The new path expects mean-reduced per-microbatch losses, as supplied by current CE/ordinal criteria. It does not implement gradient clipping or optimizer-specific mixed-precision overrides. Exact FP32/BF16 equivalence is not expected due to lower precision. GPU BF16/FP16 tests are required before production use.
