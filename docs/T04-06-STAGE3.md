# T04-06 Stage 3 — guarded Slurm integration (first E5 seed-42 gate)

**Design:** This is the **first controlled eight-job gate**, not the 120-job launcher. No resume or automatic requeue. The completed T04-05 smoke experiments are not selected, read or overwritten. Every T04-06 job uses a new planned run path.

The launcher reads existing `registry.runtime.json` and deterministic configs without regenerating them. `dry-run` only validates planning identities and hashes; it does not reserve runs or call Slurm. `prepare` builds eight frozen source snapshots, with per-run preflight and verification; existing immutable prepared runs are adopted only when identical. `submit` requires all eight prepared runs, acquires a single submission lock, writes a durable intent **before** sbatch, submits a held array, records eight indexed Slurm job IDs in the operational registry and releases the array. It never retries a submission after uncertainty. On any failure, **STOP** and inspect `stage3.e5.seed42.intent.json`, Slurm, and the registry. Do not delete the fence or rerun submit automatically.

Important limits: the Stage-2 registry does not yet have a `claiming` state. The on-disk intent fence is the fail-closed bridge across `sbatch` and registry transitions. The first array has a max concurrency of four (not the 16-slot cluster maximum), but it does **not** independently verify current GPU capacity or QoS entitlement. The `sinfo` partition gate and Slurm's own allocation checks are preliminary; check `scontrol show partition ecscp`, `sacctmgr show qos express` and cluster policy before production. The array's default `12:00:00` must be justified for up to 80 epochs (longer runs could hit the time limit). The `T04-06` worker only trains on train/validation data; T04 preflight verifies test split hashes and source-audit disclosure without evaluating test labels.

**No live monitoring/reconciliation yet.** A separate Stage 3B patch must implement scheduler-to-registry observation, final checkpoint evidence, failure handling, and resume/requeue policy. Do not mark live integration accepted solely because jobs launch. Do not launch the eight 80-epoch jobs until Stage 3B is available and live audited.

```bash
cd ~/ml-jet-ext
python -m pytest -q tests/test_t04_06_launcher.py
bash -n slurm/t04_06_array.slurm
python -m tools.t04_06_launcher dry-run
```

After review and scheduler readiness, commands are available but **do not run prematurely**:

```bash
python -m tools.t04_06_launcher prepare
python -m tools.t04_06_launcher submit --concurrency 4
python -m tools.t04_06_launcher intent
```

`prepare` is intentionally distinct from `dry-run` and modifies run directories. `submit` is intentionally non-idempotent; duplicate requests refuse the intent fence. If the submission is held because registry recording failed, cancel/reconcile the held job through an audited recovery rather than manually releasing it.

Registry commit provenance originates from Stage-2 bootstrap (`git rev-parse HEAD` at that time), whereas prepared runs record their actual source snapshot. The later full implementation must reconcile them explicitly. Split protocol: `aggregate_disjoint` with source overlap disclosed, never source-independent.
