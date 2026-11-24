# T04-06 Stage 3D — One-run controlled integration gate

## Scope
One production identity by default: `clip_s2_e5_seed42`. The 120 YAML
configurations, Stage 1 planning registry, Stage 2 runtime registry, and prior
T04-05 smoke evidence are unchanged. The new path submits **at most one held**
job and does not release it automatically. No automatic retry or resume.

## CPU-only acceptance
```bash
python -m pytest -q tests/test_t04_06_gate.py
python -m pytest -q tests/
bash -n slurm/t04_06_single.slurm
python -m tools.t04_06_gate dry-run
python -m tools.t04_06_registry verify
```
The dry-run verifies the existing config and dataset provenance; no run or Slurm
job is created. This run is configured to train from fresh to `min_epochs`
(typically 12, but read the actual YAML). **Do not use target=1** if
`min_epochs > 1`.

## Live gate (ONLY after explicit decision to allocate GPU)
1. Inspect quota/QoS/time and the one-CPU loading plan. The 120 frozen configs
   may still specify 2 DataLoader workers; do not edit them in-place.
2. `python -m tools.t04_06_gate prepare-one` (prepares one immutable run).
3. `python -m tools.t04_06_gate submit-held` (submits only **held**, then
   records Slurm job identity and a durable intent; no job begins).
4. Verify `python -m tools.t04_06_gate intent`,
   `python -m tools.t04_06_registry verify`,
   `scontrol show job JOB_ID` and `squeue -j JOB_ID`.
5. The operator must explicitly authorize and manually run
   `scontrol release JOB_ID`. No other job may target this prepared run.
6. After a successful minimum-epoch segment, inspect Slurm, checkpoint SHA,
   CSV history, session status and telemetry. **Do not** mark the registry
   completed if the configured horizon is unfinished. Stage 3B currently
   returns `none` for this valid partial state; this must be resolved before
   building an automatic continuation launcher.
7. A later resume must bind parent checkpoint SHA, require a closed session,
   and use a new audited attempt. The Stage 3C inspector intentionally blocks
   active or unfinished session state.

## Safety properties and limitations
- Held submission is serialized by an intent lock and fenced before `sbatch`.
- Ambiguous `sbatch` outcomes retain the intent. Never re-submit automatically.
- The acceptance task runs existing `tools.t04_04_pilot execute`; frozen source
  and verified checkpoint publisher remain unchanged.
- Live Slurm behavior, registry reconciliation, resume ownership and explicit
  interruption-finalization are **not** verified by the CPU test suite.
- No 8-job or 120-job production rollout should start until those acceptance
  gates pass.
- Aggregate-disjoint source-overlap disclosure is retained; this is not a
  source-independent performance evaluation.
