# T04-02 final acceptance — numerical equivalence and Git cleanup

## What the new test establishes

`test_exact_uninterrupted_vs_resumed_trajectory` compares a single four-epoch CPU trajectory with two sessions (epochs 1–2, then 3–4) under the same initial seed and scripted validation scores. The fixture performs actual PyTorch gradient updates and compares model tensors, optimizer momentum, ReduceLROnPlateau scheduler, RNG, best-epoch/early-stop counters, latest checkpoint epochs, and full epoch CSV rows for exact equality. It also confirms completed session checkpoint SHA-256 values remain unchanged.

This demonstrates deterministic CPU resume equivalence in the tested environment; it does **not** prove bitwise equality between arbitrary CUDA runs or across different GPU/library versions.

## Grid acceptance commands

```bash
python -m pytest -q \
  tests/test_t04_resume_state.py \
  tests/test_t04_sessions.py \
  tests/test_t04_session_training.py \
  tests/test_t04_stage4_acceptance.py

python -m pytest -q \
  tests/test_t04_preflight.py \
  tests/test_t04_01_integration.py \
  tests/test_t03_runtime.py \
  tests/test_t02_moment_training.py \
  tests/test_moment_vlm.py
```

Expected counts given the previously observed baseline: 25 Stage 1–4 tests and 41 regressions; 66 total.

## Non-destructive Git cleanup

Inspect files before staging: `git status --short`. Remove only the temporary `T04-02_stage*.zip` patch archives from the checkout (if no longer needed), not model checkpoints or session files. Ensure bytecode is not tracked using `git ls-files | grep -E '(__pycache__/|\.pyc$)'`.

Stage explicitly:

```bash
git add train_utils/t04_resume_state.py train_utils/t04_sessions.py \
  train_utils/t04_session_training.py train_utils/t04_session_worker.py \
  train_utils/t04_session_execute.py \
  tests/test_t04_resume_state.py tests/test_t04_sessions.py \
  tests/test_t04_session_training.py tests/test_t04_stage4_acceptance.py \
  docs/T04-02-STAGE1.md docs/T04-02-STAGE2.md docs/T04-02-STAGE3.md \
  docs/T04-02-RETENTION.md docs/T04-02-STAGE4-ACCEPTANCE.md \
  docs/T04-02-FINAL-ACCEPTANCE.md slurm/train_t04_session.slurm
```

If a listed file does not exist on the checkout, remove it from the `git add` command; inspect `git status` before committing. Check source snapshot/resume code changes carefully; never stage `training_output/`, Slurm logs, or old pilot outputs.

```bash
git diff --cached --stat
git diff --cached --check
git commit -m "feat(t04): exact resumable sessions and deterministic equivalence (refs #21)"
git push
git status --short
```

Issue #21 may be marked complete after the 66 tests pass and Git is clean, explicitly documenting that GPU session resume and retention were demonstrated operationally, but strict numeric identity was validated by CPU fixture rather than a full four-epoch GPU A/B comparison.
