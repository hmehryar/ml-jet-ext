# T04-10 Stage 9: EXACT T04-07 loss matrix (84 new runs)

Reuses **all seven** T04-07 treatments from `tools/t04_07_production.py`:

| New treatment | Formula / weighting | New single-head runs |
|---|---|---:|
| Standalone `physical_mae` | normalized expected absolute physical deviation, **no CE** | 12 |
| `ce_physical_mae` | CE + 0.1, 0.5, 1.0 × physical MAE | 36 |
| `ce_ordinal` | CE + 0.1, 0.5, 1.0 × gap-weighted squared CDF | 36 |
| Existing `cross_entropy` | reuse completed seed-matched CE training | 0 |

**84 new runs + 12 existing CE baselines = 96 comparison cells.**
Four frozen `(model,task,strategy,moments)` combinations × 3 seeds × 7 treatments.

Model/task/moment/strategy configurations: CLIP alpha S4/E3, CLIP q0 S4/E4,
SigLIP alpha S3/E2, SigLIP q0 S4/E4. Seeds 42, 123, 2026.
Keep `epochs=80` in each legacy config for immutable provenance: per-submission
`--target 200` executes a fresh 200-epoch run and allows future resume to higher targets.
No CE checkpoint is used for ablation initialization. All previous training outputs
and T04-07 jobs are immutable and isolated. Splits remain aggregate-disjoint
with source overlap disclosure; do not use test data for selection.

## CLI: preflight is the default

```bash
python -m tools.t04_10_loss_ablation plan
# Expect NEW_RUN_COUNT 84 CE_BASELINES_REUSED 12 TOTAL_WITH_CE 96
python -m tools.t04_10_loss_ablation prepare
python -m tools.t04_10_loss_ablation preview --target 200
```

For GPU smoke acceptance run **one** held pilot (standalone MAE is useful to
check the third objective):

```bash
PILOT=t04_10_clip_alpha_only_seed42_physical_mae_v1
python -m tools.t04_10_loss_ablation submit --apply --only "$PILOT" --target 200
# Resource review then guarded release:
bash tools/t04_10_loss_release.sh JOBID
bash tools/t04_10_loss_release.sh JOBID --apply
```

After pilot tests verify actual loss, epoch telemetry, checkpoint hashes and
fresh session/resume, submit remaining jobs held:

```bash
python -m tools.t04_10_loss_ablation preview --exclude "$PILOT" --target 200
python -m tools.t04_10_loss_ablation submit --apply --exclude "$PILOT" --target 200
```

**IMPORTANT:** Held submission does not release. Validate each actual job ID
with `tools/t04_10_loss_release.sh` before release. The controller refuses to
submit while any `T04_10_loss` jobs are active/held: complete or cancel the
pilot first, and never retry a partially submitted batch. Submit smaller
batches by treatment using `--treatment physical_mae`,
`--treatment ce_ordinal_lambda0p1`, etc. if desired.

For 84 jobs, account for scheduler/QOS limits and file-system contention;
consider treatment-batched releases instead of running everything concurrently.
Raw train/val losses are not comparable across objectives; use selected-model
macro-F1, accuracy, ordinal tolerance and physical MAE/RMSE. Inference and
publication plots are separate from training. Validate with GPU smoke before
full rollout. Keep full 80-epoch CE and 200-epoch CE benchmarks separately.
