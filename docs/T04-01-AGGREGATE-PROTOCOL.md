# T04-01: Explicit aggregate-disjoint evaluation protocol

The published split membership is aggregate-disjoint, but the independently executed source-ID audit found **103,221** source IDs spanning train/validation/test. This is documented overlap, not a verification error. Do not state that the split is source-disjoint or that aggregate predictions are statistically independent.

For this benchmark, set `split_protocol: aggregate_disjoint`, while still supplying `source_audit_report` pointing to the existing `source_disjointness.FAILED.json` file (the name denotes failure of a *source-disjointness* assertion, not invalid evidence). The preflight will verify the published split hashes, audit split hashes, and all 35 per-configuration overlap counts and record the actual nonzero overlap in the immutable run manifest. A `source_disjoint` protocol continues to require zero overlap.

Do not change the dataset, split assignments, or the audit output. Model comparisons on these fixed splits remain possible, but generalization claims must disclose source reuse and its possible bias. Treat the held-out test as aggregate-disjoint only, and consider independent simulation seeds for a future robustness set.

## Update production YAML

```yaml
split_protocol: aggregate_disjoint
source_audit_report: /wsu/home/gy/gy40/gy4065/ml-jet-ext/training_output/T04/audits/source_disjointness.FAILED.json
```

## Verify

```bash
python -m pytest -q tests/test_t04_preflight.py tests/test_t04_01_integration.py
python -m train_utils.t04_preflight --config experiments/t04/config/production/siglip_s1_e5.yml --check
```

This patch does not change model training or T03 code.
