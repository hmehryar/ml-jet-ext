# T04-10 Stage A — ViT E1–E5 smoke test

## Scope and scientific intent

- **Exactly 10 new training runs**: ViT-Tiny scratch, S4, CE, alpha-only and q0-only × E1–E5, seed 42.
- Each run starts with **two total epochs**; a 2-epoch smoke test must **not** be used for ranking moment configurations.
- Each prepared run has its own immutable source snapshot, run manifest, loss manifest, latest checkpoint pointer, training history and session lineage.
- **Resume the same experiment** to 40, 80 and 200 epochs by changing `--target` in a later submission. No warm-start from one E to another.
- The source dataset is aggregate-disjoint, **not source-jet disjoint**. Keep test data unused for model selection.
- LR 1e-4 for all three parameter groups, AdamW wd .01, BF16, bs=8, 1 DataLoader worker, ReduceLROnPlateau (patience=4, factor=.5). This is a smoke/initial screening protocol, not a verified optimal scratch-ViT schedule.

## Install

1. `unzip T04-10-stage10-vit-E1-E5-smoke.zip -d /tmp/vit_stage_A`
2. `python /tmp/vit_stage_A/t04_stage10_vit_E1_E5_smoke_pkg/install.py "$PWD"`
3. `PYTHONPATH=. python -m pytest -q tests/test_t04_10_vit_study.py tests/test_t04_10_extension.py tests/test_t04_10_stage6_sessions.py tests/test_t04_10_loss_ablation.py tests/test_t04_07_losses.py`
4. `bash -n slurm/t04_10_vit_single.slurm tools/t04_10_vit_release.sh`
5. `python -m tools.t04_10_vit_pilot generate && python -m tools.t04_10_vit_pilot plan`
6. Commit source and generated YAMLs before `prepare`.
7. `python -m tools.t04_10_vit_pilot prepare` then `python -m tools.t04_10_vit_pilot preview --target 2`.

## Safety-first submission

Submit the ten **held** jobs only once:

`python -m tools.t04_10_vit_pilot submit --target 2 --apply`

Verify all ten `SUBMITTED_HELD` lines and Slurm resources. Use `bash tools/t04_10_vit_release.sh JOB_ID` to preview and append `--apply` to release individual verified jobs. Never rerun `submit` to fix queue delays. All submit calls reject unrelated queued `T04_10_vitA` jobs, but a partial submission may require manual reconciliation with `sacct`.

## Acceptance criteria

- 10/10 completed **two epochs**, across both tasks and E1–E5, CE finite.
- BF16 forward/backward, 1400 attempted optimizer steps/epoch if 11200 training samples and batch=8, zero skipped unless justified.
- Checkpoints SHA-256 verified against `latest_checkpoint.json`; second session resumes from committed epoch 2 to epoch 3 on a **single diagnostic selected run** (optional acceptance for later stages).
- No other head instantiated; adapter inputs `channels=E`, backbone random, no transfer/pretrained checkpoint.
- `sacct` Slurm state reconciled with committed history; output for final report excludes smoke scores from hyperparameter selection.

**Important:** The Stage A patch replaces no existing prepared run and does not submit anything at installation. If you previously installed the 48-run ZIP, do not overlay this patch; use the clean baseline commit.
