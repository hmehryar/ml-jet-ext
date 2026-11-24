# T04-04 Stage 1 — Named AdamW parameter groups

This patch adds an **opt-in independent optimizer module** without modifying the current T04 worker or legacy T03 behavior. The new optimizer is not yet activated in full production sessions; that integration belongs to the next patch.

* Group order: `backbone`, `adapter`, `heads`.
* `heads` includes both alpha and Q0 classification heads.
* Frozen backbone tensors are omitted, and no parameter may appear twice or be omitted.
* AdamW is default with explicit nonzero weight decay; legacy Adam is explicitly selectable with zero decay.
* All three rates may differ. Groups receive stable names for LR logging and checkpoint inspection.
* Optimizer state round-trip is tested using a small CPU-only model.

Run:

```bash
unzip -o T04-04_stage1_optimizer_groups.zip -d .
python -m pytest -q tests/test_t04_optimizers.py
```

**Next:** wire this builder into `t04_session_worker.py`, add a controlled configuration field for optimizer selection/weight decay without invalidating old YAMLs, test resume with optimizer groups, then implement precision/accumulation.
