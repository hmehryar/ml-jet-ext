# T04-02 Stage 4: acceptance and production validation

The new CPU acceptance tests check:
- resumption uses epoch LAST when BEST is earlier, with correct stagnant counter;
- previous completed session checkpoint bytes stay unchanged;
- repair of one missing CSV row after a committed checkpoint, without duplicates;
- corrupt LAST is rejected even if a BEST checkpoint remains intact;
- early-stopped trajectories reject exact resume;
- default `last_best` retains only last and best session checkpoint files.

Run on Grid (alongside earlier Stage 1-3 files):

```bash
python -m pytest -q tests/test_t04_resume_state.py tests/test_t04_sessions.py tests/test_t04_session_training.py tests/test_t04_stage4_acceptance.py
python -m pytest -q tests/test_t04_preflight.py tests/test_t04_01_integration.py tests/test_t03_runtime.py tests/test_t02_moment_training.py tests/test_moment_vlm.py
```

These tests are CPU deterministic fixtures; they do NOT establish exact A100/MIG trajectory equivalence across two independently executed Slurm jobs. Complete the following before closing Issue #21:

1. Fresh prepared run, S1 E5, `T04_CHECKPOINT_RETENTION=last_best`; run two epochs, resume to four. Both jobs must complete with exit status 0.
2. Check `latest_checkpoint.json` epoch 4, sessions 001/002 status completed and epochs 1,2,3,4 exactly once in global metrics.
3. Confirm session-001 saved bytes SHA-256 before and after session-002; each session has at most one `epoch_*.pth` and at most one `best_epoch_*.pth`.
4. Confirm resumed last checkpoint includes complete RNG, optimizer, scheduler, best metric/epoch and stagnant count. If best epoch < last epoch, pointer must reference LAST.
5. Separately run a 4-epoch uninterrupted experiment on identical GPU software/hardware, seed, mini-batch/data loader settings. Compare model tensor values, optimizer states, scheduler states and epoch CSV metrics within documented deterministic/numerical tolerances. The two-run comparison should not be inferred from exit codes alone.
6. Test controlled interruption after committed checkpoint and verify the next session resumes from its last valid pointer without duplicate metrics.
7. Clean up any accidental `train_utils/__pycache__/*.pyc` added by previous ZIPs; do not commit temporary archives or huge checkpoint artifacts.

Important: full bitwise equivalence requires deterministic kernels, consistent CUDA/cuDNN/runtime and stable data-loader worker behavior. Treat independent-job bitwise equality as a testable requirement, not a guarantee. No older T04-01 prepared snapshots should be rewritten.
