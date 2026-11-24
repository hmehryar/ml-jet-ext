# T04-04 Synthetic GPU acceptance

Runs four deterministic CUDA cases on the **same 11-sample dual-head fixture** with three optimizer windows of 4,4,3 samples: FP32 reference microbatch=4/accum=1; FP32 microbatch=1/accum=4; BF16 and FP16 microbatch=1/accum=4. No dataset or HF checkpoint is required. No preflight or immutable-run snapshot is created.

Checks sample count, window count, finite train/validation loss, finite and changed parameters, BF16 without a scaler, FP16 with an enabled scaler and state roundtrip, FP32 final parameter equivalence and AMP parameter discrepancy sanity. Reports peak allocated and reserved GPU memory. GPU outputs are not guaranteed bitwise deterministic. The FP16 `optimizer_steps_attempted` count is **not** a guarantee every update was applied if GradScaler skipped an overflow.

## Commands

```bash
cd ~/ml-jet-ext
unzip -o T04-04_stage2_gpu_acceptance.zip -d .
python -m py_compile tools/t04_04_gpu_acceptance.py
sbatch slurm/t04_04_gpu_acceptance.slurm
# After completion:
cat slurm-<jobid>.out
cat slurm-<jobid>.err
cat training_output/T04/gpu_acceptance/t04_04_<jobid>.json
```

Run only once on a node with CUDA; direct interactive execution works via `python -m tools.t04_04_gpu_acceptance --output /tmp/t04_gpu_acceptance.json`. No changes to existing T03/T04 code are made by this patch. Dataset-backed fresh/resume and full-model BF16/FP16 acceptance are separate later gates.
