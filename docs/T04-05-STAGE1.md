# T04-05 Stage 1 — Independent E5 strategy matrix

This package contains a CPU-only configuration generator and two tests. It reads the **already preflight-validated** T04-04 CLIP and SigLIP S2 BF16 pilot YAMLs from `experiments/t04/config/` and emits eight distinct configuration files in `experiments/t04/config/t04_05/`.

No immutable runs or GPU sessions are created at this stage. It refuses to overwrite existing configs.

```bash
cd ~/ml-jet-ext
git switch -c feat/t04-05-mig-feasibility
unzip -o T04-05_stage1_matrix.zip -d .
python -m pytest -q tests/test_t04_05_matrix.py
python -m tools.t04_05_matrix
for f in experiments/t04/config/t04_05/*.yml; do python -m train_utils.t04_preflight --config "$f" --check || exit 1; done
```

The baseline batch size 1 / accumulation 4 is conservative and **not a final production selection**. T04-05 Stage 2 will profile microbatches 2/4/8 and decide whether to use A100 MIG or a larger GPU. All runs use existing aggregate-disjoint data with source audit disclosure. Do not prepare experiments until feasibility settings are selected.
