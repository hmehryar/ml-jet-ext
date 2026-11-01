# T04-03 Stage 3: real Hugging Face model validation

Each invocation loads a real CLIP or SigLIP vision encoder, not FakeVision.
The validator checks exact trainability masks and optimizer membership for
S1–S4, model/transformer structure, and gradient propagation at chosen E1–E5
channel count. S4 constructs randomly initialized vision weights; HF config
may still be retrieved, but pretrained *model weights* are not loaded.

## Example

```bash
python tools/t04_validate_real_models.py --model siglip --strategy S2 --channels 5 --device cpu --revision 7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed
```

Run each model/strategy separately to avoid keeping multiple 90M-parameter
models in RAM. Default backward pass is expensive on CPU. Do not use login
node for extensive backward testing: use an interactive compute allocation.

**Warning:** Stage 3 is real-model inspection only, not GPU feasibility for
S2–S4. That is Stage 4/T04-05. Any runtime failure must be investigated; no
report should be marked PASS unless the real model executed successfully.

After model+strategy checks at E5, validate E1–E4 at least with `--skip-backward`
(per model/strategy), or use `--channels` with backward to exercise every case.
For S3 and S4, CPU backprop can be slow and require substantial RAM.

## Real-checkpoint lineage

Always supply the 40-hex pinned commit revision recorded in the matching
experiment YAML. For CLIP, retrieve the revision separately with
`HfApi().model_info('openai/clip-vit-base-patch16').sha`. E5 covers full
signed M3/M5 inputs; the adapter and bilinear interpolation do not clip them.

The validator will not touch any existing prepared experiment or dataset.
