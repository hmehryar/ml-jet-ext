# Stage 3: T04 epoch telemetry and FP16 diagnostics

Apply after Stage 2 streaming-accumulation ZIP and its 145 passing tests.

- `train_utils/t04_precision.py`: report attempted/completed/skipped FP16 optimizer updates using scaler scale decreases to detect overflow skips (unscaled FP32/BF16 steps always complete).
- `train_utils/t04_telemetry.py`: report per-group LR, time (entire epoch train+validation), samples per second (train samples divided by train+validation time), GPU peak bytes (train+validation), host maximum resident memory in KiB (`ru_maxrss`, Linux high-water mark), effective batch settings, hostname, GPU.
- `train_utils/t04_session_training.py`: write per-epoch session-local telemetry sidecars to `sessions/<session>/telemetry/epoch_XXXXXX.json` after committing checkpoint+global metrics CSV. **Do not modify checkpoint metric schema or historical epoch CSV**.
- Older sessions and immutable source snapshots remain unmodified.

**Caveats:** `optimizer_steps` means windows attempted, whereas `optimizer_steps_completed` counts scale-nondecreasing FP16 steps. `elapsed_seconds` includes train+validation; `samples_per_second` uses train samples as numerator, hence it is end-to-end epoch throughput, not pure training throughput. CUDA peak stats include training and validation. `host_maxrss_kib` is process high-water resident set, not current RAM. Session telemetry is auxiliary, not part of the resume-authoritative checkpoint; if a process crashes between checkpoint commit and sidecar publication, the sidecar may be absent and should not be treated as proof the checkpoint is invalid.

## Install

```bash
unzip -o T04-04_stage3_telemetry.zip -d .
python tools/apply_t04_04_stage3.py
python -m pytest -q tests/test_t04_04_telemetry.py tests/test_t04_04_precision.py tests/test_t04_04_streaming_accumulation.py tests/test_t04_session_training.py
```

## Acceptance before production

1. Run full 145+ test suite.
2. Execute synthetic CUDA FP16 validator; inspect scaler skipped/completed counts.
3. Prepare *new* BF16 CLIP S2 and SigLIP S2 runs and use session-aware launchers for fresh epoch and exact resume. Never alter previous prepared snapshots.
4. Verify per-epoch telemetry sidecars and checkpoint manifest hashes.
