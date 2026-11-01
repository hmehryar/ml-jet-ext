# T04-03 Stage 2: session-worker strategy integration

The `t04_session_worker.py` now resolves S1–S4 from the *immutable* `config.resolved.json`, instantiates the matching model before constructing the optimizer, checks exact optimizer membership, and records the strategy and trainable counts in session runtime metadata.

- S1/S2/S3: load pinned pretrained vision checkpoint; S1 frozen, S2 final two transformer layers and final norm/pooling, S3 entire backbone.
- S4: use pinned checkpoint **configuration** only to instantiate the same-size backbone with fresh random weights; do not load pretrained weights.
- All four policies continue to use float32, cross entropy, batch-step Adam and a single common learning rate. Distinct learning-rate parameter groups, AMP and accumulation remain T04-04.
- Existing T04-02 completed sessions are immutable. Prepare a new run after installing these changes. Do not modify a completed snapshot or switch strategy during resume.
- **GPU feasibility not yet established**: S2/S3/S4 may exceed a 10 GB A100 MIG slice when training optimizer and activations. Stage 3 should test actual model instantiation, parameter counts, gradient paths and checkpoint restore before running production jobs.

Run `python -m pytest -q tests/test_t04_strategy_session_worker.py` then T04-02/T03 regressions. The change does not modify legacy `t04_runner.py`.
