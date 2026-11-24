# T04-03 — Stage 1: model adaptation policies

Adds S1 frozen pretrained, S2 pretrained final-two-block fine-tuning, S3 full pretrained fine-tuning, and S4 architecture-matched randomly initialized weights to CLIP and SigLIP factories.

- The default/legacy `freeze_backbone=True` remains S1; `False` remains S3.
- S2 includes the last two transformer blocks and post-layer normalization; SigLIP attention pooling head is also trainable if present.
- S4 reads only the pinned model **configuration** (not weights), then constructs a random vision backbone. Use `pretrained=False, strategy='S4'`.
- Adapter is a learned `Conv2d(C,3,1)`, followed by bilinear resize 32→224, with no sign clipping. Gradient path through frozen backbone is intentionally retained.
- Trainability masks are applied **before** optimizer creation.
- This stage does NOT wire S2–S4 into prepared session execution; the T04-02 worker still fails closed. That work is Stage 2.

Test with `python -m pytest -q tests/test_t04_strategies.py` and the existing T03/T04 suite. Local tests use a small fake vision backbone; run HF architecture integration tests on Grid before declaring full acceptance.
