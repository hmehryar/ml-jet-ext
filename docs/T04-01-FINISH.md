# T04-01 remaining integration and reproducibility gates

## What is ready

- Existing `t04_runner.py --config` pilot CLI is retained.
- New production invocation goes through `t04_preflight --check`, `--prepare-only`, then `t04_snapshot_execute --prepared-run`.
- Prepared run reserves an exclusive directory; records config, split CSV and external source-audit SHA-256; archives and hashes source; records Git and package provenance.
- Snapshot execution runs in a fresh interpreter with PYTHONPATH set to frozen `source/`, and independently verifies source/config/input hashes before model loading.
- Runtime metadata records GPU identity, CUDA runtime, Slurm allocation, parameter counts and the requested checkpoint revision; the Hugging Face vision factory supports pinned revision loading.
- Existing `best_model.pth`, `last_model.pth` and metrics cannot be overwritten by a prepared first execution. Re-running requires T04-02 resume support.

## Deliberate fail-closed boundaries

Only S1/frozen, FP32, gradient_accumulation=1, cross_entropy, and equal adapter/head LR are executable in this patch. Config parsing already understands S1–S4, BF16/FP16, CE+MAE and CE+ordinal but other training behaviors belong to T04-03/T04-04/T04-07. `min_epochs>1` is not silently ignored and is rejected until T04-02. A recorded RLRP scheduler is active for prepared S1 runs.

## Source audit is still a blocking evidence requirement

Your published split manifest reports 326,163 repeated source memberships and `cross_configuration_overlap_checked: false`; it cannot substantiate zero cross-partition overlap. The independent audit report must contain real evidence and bind itself to the split-file hashes. Do not fabricate the report or claim independent verification based solely on user attestation. The audit script itself is not included in this patch. As a result, production `--check` intentionally fails until the report is supplied and validated.

## Commands

```bash
python -m pytest -q tests/test_t04_preflight.py tests/test_t04_01_integration.py
python -m pytest -q tests/test_t03_runtime.py tests/test_t02_moment_training.py tests/test_moment_vlm.py
python -m train_utils.t04_preflight --config experiments/t04/config/production_s1.yml --check
python -m train_utils.t04_preflight --config experiments/t04/config/production_s1.yml --prepare-only --run-id s1-e5-seed42
python -m train_utils.t04_preflight --verify /absolute/prepared/run
sbatch --export=ALL,T04_PREPARED_RUN=/absolute/prepared/run slurm/train_t04_prepared.slurm
```

**Never use the finished pilot output paths for a prepared experiment.**

## Still required for closing GitHub T04-01

1. Produce or locate *independently evidenced* source-disjointness report with global/per-configuration membership counts and exact split SHA hashes.
2. Pin real Hugging Face model commit SHA for each model and confirm remote revision exists.
3. Run preflight and prepare-only on the actual 14,000-example dataset and verify immutable snapshots on HPC.
4. Run one brief prepared S1 execution and verify GPU/parameter metadata, checkpoint/report outputs and T03 regression suite.

T04-02 owns exact resume, T04-03 owns S2–S4 training, and T04-04 owns BF16/accumulation/AdamW/parameter-group LR.
