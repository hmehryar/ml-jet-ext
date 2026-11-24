# T04-03 Stage 4 — GPU feasibility, then immutable session smoke

**Scope of this patch:** independent one-batch, actual CUDA forward/backward/optimizer step with loss, gradients, peak memory, and an in-memory state_dict + optimizer serialization check. It does **not** claim full T04-02 resume equivalence or launch actual training epochs.

Checkpoint revisions:

- CLIP: `2b0c5c9c317211fae649651224fe64fb56a4cd1d` (safetensors).
- SigLIP: `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`.

Run from repository root:

```bash
python -m pytest -q tests/test_t04_gpu_smoke.py tests/test_t04_strategies.py tests/test_t04_strategy_session_worker.py
```

Slurm test (start with CLIP S2 batch 1):

```bash
export T04_SMOKE_MODEL=clip T04_SMOKE_STRATEGY=S2
export T04_SMOKE_REVISION=2b0c5c9c317211fae649651224fe64fb56a4cd1d
export T04_SMOKE_BATCH_SIZE=1
sbatch --export=ALL slurm/t04_gpu_smoke.slurm
```

Check `squeue -u "$USER"` and `slurm-JOBID.out`. Save JSON under `~/t04-03-gpu-smoke-results`. Never modify an existing prepared-run source snapshot. Each strategy must use a separate prepared experiment; do not resume one strategy as another.

A 10 GB MIG partition may OOM for S2–S4; if so, reduce batch to 1 (already default) or request a larger GPU rather than changing strategies or running on a login node. For S3/S4 optimizer state on Adam the memory cost is much higher than for S1. If full-batch smoke passes, the next acceptance is a real *new* prepared S2/S3/S4 experiment and a separate resumed session via the existing T04-02 `t04_session_execute` entry point; verify last checkpoint, best retention and run integrity. Do not assume in-memory serialization is sufficient.

Cleanup: inspect `git status --short`, run all T04 tests; `git add` only relevant source/docs/tests/slurm; commit once GPU smoke and resume checks pass; push branch upstream. Do not stage completed run directories, datasets, HF cache, result files, or ZIP patches.
