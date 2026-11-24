# T04-06 Stage 3F.2C1 source-aware schema fix

This patch replaces only `tools/t04_06_patch_v2_schema.py` and adds `tests/test_t04_06_v2_schema_precise.py`.

It recognizes the actual preflight optional fields: `fold`, `split_protocol`, `optimizer`. It adds optional v2 protocol fields, validates them against the epoch budget, and routes an independent `T04-06-v2-plan` through its own verifier. The original plan kind remains supported.

Commands:

```bash
cd ~/ml-jet-ext
unzip -o T04-06_stage3f_phase2c1_precise_schema_fix.zip -d .
python -m tools.t04_06_patch_v2_schema --check
python -m tools.t04_06_patch_v2_schema --apply
python -m pytest -q tests/test_t04_06_v2_schema_precise.py tests/test_t04_06_v2_matrix.py
python -m pytest -q tests/
git diff --check
python -m tools.t04_06_registry verify
python -m tools.t04_06_v2_matrix dry-run --epochs 80
```

Only if all pass, generate the independent plan and verify it:

```bash
python -m tools.t04_06_v2_matrix generate --epochs 80
python -m tools.t04_06_v2_matrix verify
```

No training runs or Slurm jobs are submitted by this patch. Do not rerun `generate` if a v2 plan already exists.
