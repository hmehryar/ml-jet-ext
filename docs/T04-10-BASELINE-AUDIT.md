# T04-10 baseline audit and 18-run single-head pilot design

Issue: https://github.com/hmehryar/ml-jet-ext/issues/30
Audit date: 2026-11-07
Repository: hmehryar/ml-jet-ext
Inspected: `feat/clip-siglip-classifiers`, `feat/t04-06-ablation-registry`, `feat/t04-07-loss-ablations`.
Status: **Design only — no HPC jobs submitted or run configs modified.**

## Audit findings

- `models/moment_vlm.py` constructs both `alpha_head` (7 logits) and `q0_head` (5 logits) for CLIP/SigLIP. A task-aware model must construct only the selected head, leaving `joint` backward compatible.
- `models/model_vit.py` likewise constructs both heads. The T03 JetBench-transfer ViT-Tiny path (`models/jetbench_transfer.py`) is not directly matched to the T04-06 protocol.
- `train_utils/t04_preflight.py` restricts `model` to CLIP/SigLIP and does not declare a single-head task.
- `train_utils/t04_precision.py` iterates over both heads and computes joint metrics.
- `train_utils/t04_session_training.py` selects its checkpoint using validation joint accuracy; single-head mode needs task-specific checkpoint selection.
- `experiments/t04/config/t04_06_v2/clip_s4_e3_seed42.yml` and SigLIP analog are 80-epoch BF16 AdamW CE configurations using `fixed_budget_v2`. T04-07 pilot builds on CLIP S4/E3 but this is not proof that S4/E3 is the best validated configuration.
- ViT T03 E5 pretrained YAML uses a different batch size and epoch/early-stopping protocol; historical performance cannot be called matched without checking all settings.
- The published configs declare `aggregate_disjoint` and point to `source_disjointness.FAILED.json`. Do not represent source-jet independence as established.
- T04-07 production experiment configurations, registries, output directories, and Slurm jobs must remain untouched.

## Protocol decision and open evidence gates

1. Reconcile latest HPC run manifests/validation histories against the published repository before selecting CLIP and SigLIP architectures. Use only CE multi-head baselines on matching train/val/test hashes, strategy, moments, source checkpoint, optimizer, training budget, seed, and code provenance. If a matching dual-head comparison does not exist, schedule a separately counted control (do **not** count it among the 18 single-head runs).
2. Rank candidate architecture configurations on validation macro-F1 (report accuracy/physical MAE as secondary). Freeze exactly one configuration per architecture for both heads. Do not select based on held-out test performance.
3. For ViT, identify exact T03 JetBench transfer checkpoint hash, initialization, moments, splits and training controls. If no compatible dual-head baseline exists, train a matched ViT joint control independently.
4. The pilot uses 80 fixed-budget epochs and retains a task-specific best validation checkpoint. The LR scheduler can use active-task validation CE; no early stopping. Use BF16 after smoke-testing each architecture. Preserve seeds 42, 123, 2026.
5. Independently initialize both single-head models from their architecture's declared initialization (not a dual-head fitted checkpoint). Preserve split hashes, config hashes, code commit, checkpoint revision, preprocessing, datasets, hardware profile and random seeds.
6. Output namespace: `training_output/T04/t04_10_single_head` (distinct from T04-07); config namespace: `experiments/t04/config/t04_10` and job prefix `T04_10_*`. Use held-by-default Slurm submission only after smoke tests.

## 18 single-head run manifest

| Architecture | Task | Seeds | Count |
|---|---|---|---:|
| ViT | alpha_only | 42, 123, 2026 | 3 |
| ViT | q0_only | 42, 123, 2026 | 3 |
| CLIP | alpha_only | 42, 123, 2026 | 3 |
| CLIP | q0_only | 42, 123, 2026 | 3 |
| SigLIP | alpha_only | 42, 123, 2026 | 3 |
| SigLIP | q0_only | 42, 123, 2026 | 3 |
| **Total** | | | **18** |

Run ID format: `t04_10_{vit|clip|siglip}_{alpha_only|q0_only}_seed{42|123|2026}`.

## Required test gates

- 7/5 output dimensions, no inactive head instantiated, no inactive-head optimizer parameters; finite gradients and loss; CE only on active target.
- `joint` default behavior and T03/T04 legacy tests unchanged.
- BF16/CUDA real-batch smoke, mixed-precision optimizer grouping, deterministic seed behavior, exact epoch-boundary resume.
- Task-specific metrics (accuracy, macro-F1, physical MAE/RMSE), with inactive task and joint metrics explicitly unavailable (`null`/blank), rather than zero.
- Frozen configuration, SHA-256 manifest/splits/checkpoint provenance, output collisions refused, immutable snapshots, held job submissions.
- Train 80 epochs, select best checkpoint by active-task validation criterion; no test set use before selection is frozen.
- Three-seed paired comparisons and confidence/variability caveat; compute parameter counts, peak memory, GPU-hours and inference latency.

## Expert composition gate

After validation-only expert selection, evaluate `f(x) = (f_alpha(x), f_q0(x))` and joint accuracy (both heads correct), against matched joint-backbone baseline. Report doubled-backbone compute overhead. Call the composition a modular two-specialist ensemble, not a learned MoE unless a routing/gating module is trained and evaluated in a separately registered follow-up.

## Implementation locations (proposed)

- `models/t04_10_single_head.py` — explicit task-mode construction/adapters, preserving legacy classes.
- `train_utils/t04_10_training.py` — selected-task loss, metrics, checkpoint policy, resume integration.
- `train_utils/t04_10_preflight.py` — strict config/task validation, immutable preparation, split audit.
- `tools/generate_t04_10_pilot.py` — reproducible 18-run config generator and validation-only baseline manifest.
- `slurm/t04_10_single.slurm`, `slurm/submit_t04_10_held.sh` — isolated held-only jobs.
- `tests/test_t04_10_single_head.py`, `tests/test_t04_10_manifest.py`.

No branch, commits, issue edits, or executions were made through the GitHub integration because it returned HTTP 403 on branch creation. The document is an offline design artifact.
