# T04-06 Stage 2 — Operational registry

This is a CPU-only **operational sidecar** to Stage 1's immutable planning registry. It does not submit Slurm jobs, prepare experiments or alter frozen source snapshots. Its `transition` API is a controlled state-store primitive, **not evidence** that Slurm really submitted/ran a job, nor that a checkpoint hash matches a file. Stage 3 must validate external evidence and coordinate Slurm submissions before recording transitions.

## Install and initialize exactly once

```bash
unzip -o T04-06_stage2_operational_registry.zip -d .
python -m pytest -q tests/test_t04_06_registry.py
python -m tools.t04_06_registry bootstrap --commit "$(git rev-parse HEAD)"
python -m tools.t04_06_registry verify
```

Bootstrapping verifies the existing Stage 1 matrix; do **not** regenerate 120 configs or overwrite `registry.json`. `registry.runtime.json` is untracked operational state, and a `.lock` file serializes writes. Keep both private runtime registries out of Git; preserve backups of the registry after every submission batch.

## State machine

`planned -> submitted -> running -> completed`, `submitted/running -> failed`, `completed/failed -> resumed -> running -> completed/failed`. Every submitted/resumed attempt receives an ID and Slurm job identity; only the latest attempt can change state. Duplicate job identities and duplicate active attempts are rejected. Completion requires a caller-provided SHA-256 checkpoint fingerprint. Resume requires a caller-provided parent checkpoint SHA-256.

All writes acquire a POSIX `flock` and replace the JSON atomically with fsync. Input plan hash and config hashes are checked before transitions. Attempts and events are retained. The state store does not inspect Slurm or validate checkpoint contents. It is not yet safe to call `transition` as a production submitter. Stage 3 will implement pending-submission claims, job reservation semantics, Slurm query/reconciliation, and schedule-aware idempotency. **Do not run 120 jobs from this patch.**

## Verification

```bash
python -m tools.t04_06_registry summary
python -m pytest -q tests/
git diff --check
```

The workflow continues to use `aggregate_disjoint` split disclosure and frozen T04 manifests. The T04-05 smoke runs remain separate.
