# T04-05 Stage 2 — Accepted 2026-11-02

- 24/24 **unique** E5 BF16 feasibility combinations have `status: pass` JSON evidence: two encoders × S1–S4 × batches 2, 4, 8.
- CUDA memory was measured using PyTorch allocator counters, rather than Slurm MaxRSS. Maximum observed reserved was **2064 MiB** for SigLIP S3/S4 batch 8.
- Initial CLIP/SigLIP S2 batch 2 standalone probes: Slurm **40941112**, **40941113** (both successful).
- Two duplicate submissions in the 24-job sweep rejected existing report filenames as intended: **40941117_0** (CLIP S2 bs2) and **40941129_0** (SigLIP S2 bs2), exit code `2:0`. The latter log explicitly prints `report already exists`; the former has identical mapping and failure timing, but its log should be separately retained. These are not OOM results, and the unique combinations' passing JSON reports came from earlier jobs.
- Selected provisional E5 smoke setting: BF16, AdamW, microbatch 8, accumulation 1, seed 42, A100 MIG 1g.10gb. Requires full-epoch confirmation; short-probe throughput is not directly comparable with full-epoch throughput.
- Dataset is **aggregate_disjoint**, with source-overlap audit disclosed; no source-disjoint claim.
- Preserve the T04-04 pilots as separate infrastructure evidence.

To post Stage 2 acceptance to GitHub Issue #24, copy these findings into the issue discussion. Do not close the issue until eight E5 runs pass validation.
