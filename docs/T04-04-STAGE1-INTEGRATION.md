# T04-04 Stage 1B: opt-in AdamW integration

Run `python tools/apply_t04_04_optimizer_integration.py` from the repository root after installing Stage 1. The installer requires exact T04-03 source anchors and is idempotent. No completed prepared run is modified.

New T04 YAML fields:

```yaml
optimizer:
  type: adamw
  weight_decay: 0.01
learning_rates:
  backbone: 0.00001
  adapter: 0.0001
  heads: 0.0001
```

Without `optimizer`, legacy single-rate Adam remains intact, preserving existing T04-03 configurations. With `optimizer`, deterministic group order is `backbone`, `adapter`, `heads`, skipping empty groups (S1 backbone). Optimizer group ordering is stable across resumes. Existing T04-02 scheduler state is preserved. A new prepared run is required for the new optimization behavior.

Stage 1B does not yet implement AMP, accumulation, telemetry CSV columns, or alter T03. Those remain separate stages. All 3 groups can have unequal learning rates with the opt-in optimizer.
