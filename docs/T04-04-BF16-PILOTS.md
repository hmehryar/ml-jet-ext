# T04-04 BF16 CLIP/SigLIP immutable pilot

Config values preserve existing E5 and aggregate-disjoint split provenance; do not claim source independence. Validate the audit path and checkpoint revisions against the existing successful pilots.

Run `python -m pytest -q tests/test_t04_preflight.py tests/test_t04_sessions.py tests/test_t04_session_training.py tests/test_t04_04_telemetry.py` before preparing new snapshots.

For each of clip and siglip (replace MODEL):

```bash
MODEL=clip
CONFIG=experiments/t04/config/t04_04_${MODEL}_s2_bf16_pilot.yml
python -m train_utils.t04_preflight --config "$CONFIG" --check
python tools/t04_04_pilot.py prepare --config "$CONFIG" --run-id "t04-04-${MODEL}-s2-bf16-pilot-01"
# Path is output_root/model/S2/E5/seed-42/run_id.  Use an absolute path:
RUN="$(pwd)/training_output/T04/production/${MODEL}/S2/E5/seed-42/t04-04-${MODEL}-s2-bf16-pilot-01"
python -m train_utils.t04_preflight --verify "$RUN"
# Submit only one session at a time. Wait for the fresh job to complete before submitting resume.
sbatch --export=ALL,T04_PILOT_RUN="$RUN",T04_PILOT_MODE=fresh,T04_PILOT_TARGET=1 slurm/t04_04_bf16_pilot.slurm
# AFTER sacct shows COMPLETED exit 0: check epoch 1 and latest_checkpoint.json
sbatch --export=ALL,T04_PILOT_RUN="$RUN",T04_PILOT_MODE=resume,T04_PILOT_TARGET=2 slurm/t04_04_bf16_pilot.slurm
# AFTER completion:
python tools/t04_04_verify_pilot.py --run "$RUN"
```

Repeat with MODEL=siglip only after the CLIP run passes. Fresh and resume Slurm jobs cannot overlap for a run.

Do not rerun a failed `prepare` with the same run-id; reservations are never reused. Do not rerun `fresh` after an interrupted job; inspect latest checkpoint and session status first. The worker may stop early depending on configured policy. Any test-set evaluation is out of scope.

**Important:** Immutable source snapshots copy train_utils, models, and data. They do **not** copy tools. The pilot orchestration script executes outside the frozen source, but the actual training worker executes from the read-only source snapshot using PYTHONPATH pointed to the snapshot. Production model code is sourced from that snapshot only.
