# T03 launch guide — pretrained ViT-Tiny cumulative M1–M5

This package preserves the legacy `train.ipynb` and historical Slurm launcher. It adds `train_t03.ipynb` (same conceptual stages), `train_t03.py` (already exported from notebook), five-configuration generator, and `slurm/train_t03_moments.slurm`.

All experiments use a 3-channel ViT-Tiny with an M1-to-three-channel trainable projection, transfer 300/300 compatible backbone weights from the historical JetBench checkpoint and initialize **fresh** 7-class/5-class heads. Heads, adapter and backbone are trained jointly. The T02 moment loop records separate alpha/Q0 accuracies, macro-F1 and joint exact-match and holds back the test set.  Dataset split sizes: 11,200/1,400/1,400.

## Deploy

1. Copy ZIP to `/wsu/home/gy/gy40/gy4065/ml-jet-ext`, `unzip -l ...`, then `unzip ...` in repo root.
2. Run `python tools/generate_t03_configs.py` to generate five independent YAMLs under `experiments/exp_03_cumulative_moment_ablation/config/`.
3. `python -m jupyter nbconvert --to python train_t03.ipynb --output train_t03` (run only once before submission). An equivalent `train_t03.py` is also in ZIP.
4. Test pilot using a **separate directory**: `python tools/generate_t03_configs.py --root experiments/exp_03_pilot --epochs 1`; `PYTHONPATH="$PWD" python train_t03.py --config experiments/exp_03_pilot/config/E2_m1_m2_pretrained.yml` on an allocated GPU node. This scans full train/validation sets.
5. `mkdir -p logs` before submission. Then submit each of the five YAMLs via `sbatch slurm/train_t03_moments.slurm experiments/exp_03_cumulative_moment_ablation/config/E1_m1_pretrained.yml` etc.
6. `squeue -u "$USER"`, `tail -f logs/t03_JOBID.out` and `tail -f logs/t03_JOBID.err` for monitoring.

## Care points

- `train.ipynb` and the original `slurm/train_ml_jet.slurm` are **not modified**.
- Environment is `/wsu/home/gy/gy40/gy4065/.conda/envs/pytorch/bin/python`. If your allocation uses a different Python, edit PYTHON_BIN in new Slurm file.
- Slurm partition/GRES are based on supplied previous script; resource requests/time caps must be verified on cluster before submitting. No automatic QOS is requested.
- T03 notebook expects `models/jetbench_transfer.py`, plus T02 `config.py`, `data.loader`, `models.model_vit` and `train_utils.moment_training` from prior successful integration.
- Model transfer is validated by `transfer_backbone()` checking coverage and exact loaded tensors; training fails if coverage below 99%.
- The 3-channel backbone uses the historical pretrained 224x224 positional embeddings. Adapter initially replicates M1 and zeroes the other moments until learned.
- The checkpoint's historical optimizer state is *not* loaded, and fresh heads and optimizer are created for every experiment.
- The **training output path** is derived by `get_config` by appending a run tag to the YAML `output_dir`; results will be nested under the per-experiment base directory.
- If a run folder already contains checkpoints/CSV, script refuses to overwrite. Choose a new output directory or implement proper resume.
- The training loop chooses best checkpoint/early stopping based on **joint validation accuracy**, not validation loss. Current scheduler is `ReduceLROnPlateau(mode='max')` on joint validation accuracy.
- One complete GPU training epoch has been verified for T02 but this exact notebook+Slurm T03 end-to-end procedure has **not yet been executed on user's cluster**. Run the pilot first.
- Do not interpret first-epoch chance-level accuracy as model failure. Compare converged validation scores across runs and eventually repeat with multiple seeds.
- Source-jet overlap remains a dataset limitation despite aggregate-disjoint splits.
