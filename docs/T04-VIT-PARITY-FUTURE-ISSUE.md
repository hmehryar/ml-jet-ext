# Proposed follow-on issue: ViT-Tiny Transformer Parity Benchmark

**Dependency:** T04-06, T04-07 and T04-10; **separate from #30**.

## Goal

Make historical JetBench-transferred ViT-Tiny comparable, as far as experimental
protocol permits, to CLIP ViT-B/16 and SigLIP ViT-B/16. Architectural parameter
count and pretraining distribution still differ and must be reported.

## Work packages

- [ ] Audit historical ViT-Tiny initialization, checkpoint hash and datasets.
- [ ] Implement ViT adaptation S1 frozen backbone, S2 partial unfreeze, S3
  fully trainable pretrained backbone, S4 scratch; specify an objectively
  analogous S2 layer count/fraction despite unequal backbone depths.
- [ ] Extend moment-based E1–E5 experiments over all four strategies and seeds
  42/123/2026, with matching 80-epoch fixed-budget protocol where feasible.
- [ ] Implement same CE / CE+ordinal-CDF / CE+physical-MAE (and any additional
  fixed loss variants selected for T04-07) across all four strategies, without
  changing T04-07 CLIP/SigLIP production runs.
- [ ] Match split CSV SHA256s, signed-moment normalization, validation-only
  selection, optimizer/scheduler protocol and BF16 handling.
- [ ] Report both per-architecture best and protocol-controlled matched
  comparisons. Include parameter count, FLOPs, throughput and latency.
- [ ] Preserve immutable run manifests and held Slurm jobs.

**Important:** ViT-Tiny vs CLIP/SigLIP ViT-Base compares different model sizes,
initialization/pretraining data, and architecture configurations. Label this a
transformer-family benchmark, not a strict controlled model-architecture causal
comparison. A strict architecture comparison requires additional scale- and
pretraining-matched models.
