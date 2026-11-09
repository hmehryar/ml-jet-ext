# T04-10 Stage 4 — T04-06 CE baseline audit and validation-only pilot freeze

**Issues:** #30 (single-head) and #31 (metrics). **No model training, GPU load, Slurm submissions, or T04-07 mutations.**

## Selection policy (fixed before inspecting validation scores)

- Source: **T04-06 v2**, `fixed_budget_v2`, CE dual-head, 80 epochs, S1–S4 × E1–E5 × seeds {42,123,2026} × {CLIP,SigLIP} = **120 separate runs**.
- Require exactly one unambiguous completed trajectory for every expected configuration. No silent partial-grid winner selection.
- Use **only validation metrics at the saved dual-head joint-accuracy-selected checkpoint epoch** (earliest epoch maximizing validation joint accuracy). The T04-06 training loop does not select checkpoints by per-head F1.
- Rank *separately for each (CLIP/SigLIP, alpha_only/q0_only)* by the 3-seed mean validation macro-F1. Resolve ties by higher validation accuracy, then lower physical MAE, then deterministic strategy/E order.
- This produces up to **four distinct preferred configurations**, while retaining the original seed pairing. Never choose based on test metrics or T04-07 loss-ablation results.
- **Comparison caveat:** selecting the best among 20 strategies/moment combinations per head on one fixed validation split can overfit validation, despite using three seeds. The held-out test set should remain unseen until all policies and runs are frozen.
- The current split audit reports source-level reuse (previously 103221 overlap occurrences); source independence must **not** be claimed. A leak-free source-disjoint re-evaluation remains required for strong generalization claims.

## Source artifact inspection

Read-only preflight:

```bash
cd ~/ml-jet-ext
conda activate pytorch
PYTHONPATH=. python tools/t04_10_stage4_freeze.py \
  --configs experiments/t04/config/t04_06_v2 \
  --runs /rs/rs_grp_mljet/ml-jet-ext/training_output/T04/t04_06_v2_production \
  --audit \
  > /rs/rs_grp_mljet/ml-jet-ext/training_output/T04/t04_10_single_head/stage4_audit.json
```

Inspect the audit before freezing:

```bash
python - <<'PY'
import json
p='/rs/rs_grp_mljet/ml-jet-ext/training_output/T04/t04_10_single_head/stage4_audit.json'
d=json.load(open(p))['audit']
for k in ('config_count','complete'):
    print(k,d[k])
for k in ('missing_configs','extra_configs','missing_runs','bad_runs','config_errors'):
    print(k,len(d[k]));print('  first:',d[k][:3])
PY
```

**Do not freeze if `complete != 120`, or there are missing, ambiguous, mismatched, or incomplete runs.** Audit the T04-06 run registry and progress first, or repair the run provenance; do not fabricate scores or use the test set to fill gaps.

## Freeze the 12 CLIP/SigLIP candidate configurations

Only after inspecting a complete audit:

```bash
PYTHONPATH=. python tools/t04_10_stage4_freeze.py \
  --configs experiments/t04/config/t04_06_v2 \
  --runs /rs/rs_grp_mljet/ml-jet-ext/training_output/T04/t04_06_v2_production \
  --output experiments/t04/config/t04_10_stage4_frozen
```

Outputs: `ranking.csv` (40 rows, four selection lists × 20 candidates), `pilot_plan.json` (4 winners, 12 single-head configs, 6 ViT blocked), and 12 YAMLs. The output directory must not already exist. No job is prepared or submitted by this tool.

Then validate the frozen YAMLs with T04-10 preflight (`--check`) and review against the original T04-06 dual-head checkpoint references. Any individual pilot preparation requires a clean tracked Git tree and existing split/source audit.

## Differences from the original Stage 2 candidate generator

Stage 2's `t04_10_generate_matrix.py` takes **one source baseline per architecture**, and was intended only for smoke/candidate setup. Stage 4 does not replace or modify that script. It selects the best **per task** while retaining the matched CE source run for each seed and recording the selected strategy/moments and the actual original dual-head best checkpoint hash.

## Known limits

- Freeze audit checks the captured T04-06 manifest, configuration, original-YAML hash, 80-row validation history, checkpoint existence and content digest, plus matching input hashes **across** the 120 run manifests. Run artifacts themselves are not on GitHub and must be audited on the WSU Grid `/rs` mount.
- It does not verify the original GPU batch's final tensor predictions or cross-source jet-ID independence; an independent leakage and inference audit is still needed.
- It does not guarantee matching wall-clock/GPU profiles or identical stochastic training of single- vs dual-head models; these must be documented in the final comparison.
- ViT parity remains blocked, and no held production job is created or released.
