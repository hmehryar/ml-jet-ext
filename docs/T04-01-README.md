# T04-01: production preflight (stage-one implementation)

This is a **CPU-only standalone preparation CLI**, intentionally separate from the existing `t04_runner.py`. It does **not** launch training, modify T03, or overwrite any completed T04 pilots.

1. Copy `train_utils/t04_preflight.py` and `tests/test_t04_preflight.py` into the repository.
2. Run `python -m pytest -q tests/test_t04_preflight.py` in the `pytorch` environment.
3. Add a production YAML with all required fields described in `T04Config`.
4. Supply the independently produced **verified source-overlap audit JSON**. Its contract is:

```json
{
  "global_cross_split_overlap": 0,
  "per_configuration_cross_split_overlap": {"<each of 35 actual configuration IDs>": 0},
  "source_id_method": "description of true source event identity extraction",
  "split_sha256": {
    "train.csv": "sha256 of train.csv",
    "val.csv": "sha256 of val.csv",
    "test.csv": "sha256 of test.csv"
  }
}
```

**Do not fabricate this audit file.** If the existing audit has a different layout, write a verified adapter that exports the required fields from the actual audit evidence. The preflight only checks the reported zero-overlap results and that they match the supplied split hashes; it does not independently recompute underlying jet membership.

Use an immutable full Hugging Face checkpoint revision commit SHA. Do not supply `main`.

```bash
python -m train_utils.t04_preflight --config experiments/t04/config/production.yml --check
python -m train_utils.t04_preflight --config experiments/t04/config/production.yml --prepare-only --run-id pilot-001
python -m train_utils.t04_preflight --verify /absolute/path/to/prepared/run
```

**Important:** This is Stage 1 of Issue T04-01, not a completed end-to-end T04 runtime. The current T04 runner does not consume the snapshot. T04-01 still requires: binding the prepared snapshot to the execution engine, fail-closed runtime verification on the compute node, recording actual GPU/MIG telemetry and training parameter counts, and a complete audit adapter for the existing provenance report. Do not mark the GitHub issue closed yet.
