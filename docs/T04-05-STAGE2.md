# T04-05 Stage 2: Isolated A100 MIG feasibility probes

Each Slurm job executes a **real BF16 forward/backward/AdamW step** on dataset samples for one of eight model/strategy configurations and one microbatch size (2, 4, or 8). Outputs are unique, append-only JSON files and Slurm logs; no immutable production experiment is prepared or modified. Job failures are classified as `oom` or `error`; only `pass` counts as feasible.

Use `--steps 8` for the screening gate, then benchmark complete epochs for selected profiles. Eight steps give *step throughput*, not steady-state epoch throughput; do not label the result epoch throughput. CUDA memory is the maximum of allocated/reserved values seen during the loop and includes model parameters/optimizer state. Compare reserved memory to `cuda_total_bytes` with safety headroom.

Run in a separate allocation for every batch/config to release CUDA memory after failure. These probes do not establish accuracy, checkpoint/resume or early stopping. Those are checked in the subsequent eight short immutable runs.

The dataset split is `aggregate_disjoint`, not independent source-disjoint. Retain the source-overlap disclosure in reports.
