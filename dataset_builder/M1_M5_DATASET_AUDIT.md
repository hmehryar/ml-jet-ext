# ML-JET-Ext M1–M5 Dataset Audit

**Tracking issue:** [#17 — T01: Audit and Validate M1–M5 Moment-Based Dataset](https://github.com/hmehryar/ml-jet-ext/issues/17)  
**Status:** Dataset, metadata, tensor, and split checks completed; corrected builder is **provisionally accepted** pending an HPC regression run.  
**Audit evidence:** HPC command outputs collected during T01; statistics below refer to the **normalized** M1–M5 dataset.

## 1. Scope and dataset contract

ML-JET-Ext represents each physics configuration as 500-jet aggregate images on a `32 × 32` pseudorapidity–azimuth grid. Each saved `.npy` tensor contains **five channels in order M1–M5**:

| Channel | Quantity | Definition | Expected normalized range |
|---|---|---|---|
| M1 | Pixelwise mean | `mean(X)` | `[0, 1]` |
| M2 | Second central moment / variance | `mean((X - M1)^2)` | `[0, 1]` |
| M3 | Third central moment | `mean((X - M1)^3)` | `[-1, 1]` |
| M4 | Fourth central moment | `mean((X - M1)^4)` | `[0, 1]` |
| M5 | Fifth central moment | `mean((X - M1)^5)` | `[-1, 1]` |

The odd central moments are **not standardized skewness or fifth standardized moments**. The normalized tensors use the fixed log / signed-log transforms defined in `04_build_agg500_m1to5_dataset.py`, with reference scale `110^k` for moment order `k` and epsilon `1e-8`. Inference and training must **not normalize these tensors a second time**.

| Property | Verified specification |
|---|---|
| Physics grid | 7 `alpha` values × 5 `q0` values = **35 configurations** |
| `alpha` values | `0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50` |
| `q0` values (GeV) | `1.0, 1.5, 2.0, 2.5, 3.0` |
| Aggregate size | **500 distinct source jets** per saved sample |
| Aggregates per configuration | **400** |
| Total saved aggregate tensors | **14,000** |
| Tensor shape | `(5, 32, 32)` |
| Tensor dtype | `float32` |
| File format | `.npy` |

### Filesystem layout

```text
pTHat50-250_eCM5020_c0005_pT90-110_R0p4_agg500_m1-5_normalized/
├── c0005_alpha0p20_q_1p0/
│   ├── agg500_event_0000000.npy
│   └── ...
├── ... (35 configuration directories)
├── metadata/
│   ├── c0005_alpha0p20_q_1p0_agg500_metadata.csv
│   └── ... (35 cleaned CSV files)
├── metadata_backup/           # If preserved by the operator
└── splits/
    ├── train.csv
    ├── val.csv
    ├── test.csv
    ├── split_manifest.json
    └── five_fold/
        ├── fold_0_train.csv
        ├── fold_0_val.csv
        └── ... fold_4_train.csv / fold_4_val.csv
```

Metadata fields are `agg_file_path`, `source_file_paths`, `alpha`, `q0`, and `config`. `agg_file_path` is **relative to the normalized dataset root**; `source_file_paths` references the separate original unfolded single-jet dataset. Dataset code should resolve these roots independently.

## 2. Metadata integrity and cleanup

The initial audit of the 35 configuration CSV files reported:

| Metric | Count |
|---|---:|
| Original metadata rows | **33,844** |
| Unique `agg_file_path` values | **14,000** |
| Extra exact duplicate rows | **19,844** |
| Repeated paths with differing metadata | **0** |
| Distinct 500-jet source-membership groups | **14,000** |
| Different output paths for the same source-membership group | **0** |
| Invalid groups (not 500 distinct paths) | **0** |
| Physical aggregate `.npy` files | **14,000** |
| Missing referenced `.npy` files | **0** |
| Unreferenced `.npy` files | **0** |

Duplicate CSV rows were removed using `07_deduplicate_metadata.py`, initially written into `metadata2/` and then adopted as the canonical `metadata/`. This retained **400 unique records per configuration**, totaling **14,000**. No aggregate `.npy` files required regeneration for this cleanup. The original metadata was intended to be retained in `metadata_backup/`.

The original moment builder contained two metadata risks: (1) appending the same output row twice on the new-computation path, and (2) reading source metadata from the destination directory before overwriting that same CSV. A corrected implementation has been prepared to read the **original `agg500/metadata/`** as input, append one row per distinct aggregate, and make reruns idempotent. **The corrected builder was provisionally accepted for T01; its end-to-end HPC regression has not been reported as passed.** Do not interpret the clean existing dataset as proof of that regression.

## 3. Tensor validation: entire dataset

`08_validate_moment_tensors.py` scanned **all 14,000 files**, using 16 CPU workers (18 seconds reported on the HPC node; approximately 768 files/second).

| Validation | Failures |
|---|---:|
| Invalid shape | **0** |
| Incorrect dtype | **0** |
| NaN / infinity | **0** |
| Out-of-range normalized values | **0** |
| Unreadable files | **0** |

Observed global channel ranges:

| Moment | Minimum | Maximum |
|---|---:|---:|
| M1 | 0.00000000 | 0.92213577 |
| M2 | 0.00000000 | 0.88131428 |
| M3 | 0.00000000 | 0.86346483 |
| M4 | 0.00000000 | 0.86834055 |
| M5 | 0.00000000 | 0.87188089 |

These observations demonstrate range compliance, not by themselves the physical correctness of moment values.

### Odd-moment sign checks

Across the entire stored normalized dataset:

| Statistic | M3 | M5 |
|---|---:|---:|
| Negative pixels | **0** | **0** |
| Positive pixels | **7,808,316** | **7,808,120** |
| Zero pixels | **6,527,684** | **6,527,880** |

`09_test_moment_sign_preservation.py` independently tested the signed-log transform using synthetic positive and negative inputs:

- M3 normalized `-100` to `-0.7080058455` and `+100` to `+0.7080058455`.
- M5 normalized `-1000` to `-0.6041644812` and `+1000` to `+0.6041644812`.
- Zero and finite-value assertions passed.

Thus the normalization function **preserves signs**. Nonnegative odd central moments in the saved dataset should be described as an **empirical observation**, not assumed to be true for all possible jet distributions.

## 4. Source-to-aggregate numerical consistency

`10_verify_moment_consistency.py` randomly sampled **one aggregate from each of 35 physics configurations**, reading **17,500 source-jet images** and recomputing all five central moments with the builder's PyTorch functions and normalization on an available NVIDIA A100 MIG instance.

**Result: 35/35 comparisons passed, zero failures.** The test used `atol=1e-5` and `rtol=1e-4`.

| Moment | Maximum absolute error | Mean absolute error |
|---|---:|---:|
| M1 | `1.1920928955e-07` | `1.9457989505e-10` |
| M2 | `1.1920928955e-07` | `3.6190613173e-09` |
| M3 | `5.9604644775e-08` | `3.1337549444e-09` |
| M4 | `5.9604644775e-08` | `1.0024190747e-09` |
| M5 | `1.1920928955e-07` | `5.6981142344e-10` |

For those 35 recomputed aggregates, the raw M3 and M5 each contained zero negative pixels and had a minimum of zero. This corroborates the saved outputs **for the sampled aggregates only**; the whole dataset was checked for format/ranges, not independently reconstructed from source jets.

## 5. Source-jet overlap: limitation of this benchmark

`11_generate_dataset_splits.py` detected **326,163 source-jet reuse occurrences** when scanning within physics configurations. Each 500-jet aggregate has distinct membership, and no two aggregate *groups* were identical, but **a given source jet can contribute to multiple aggregates**.

The subsequent connected-component analysis demonstrated that some configurations are completely aggregate-source-disjoint (400 singleton components), while others contain a single connected component spanning all 400 aggregates. The reuse figure counts repeated source-path encounters; it is **not** a count of duplicate aggregate samples or of overlapping pairs of aggregates.

**Consequence:** The fixed splits below are **aggregate-disjoint but not guaranteed source-jet-disjoint**. Reused underlying jets introduce dependencies between samples and could lead to optimistic evaluation estimates. Dataset uniqueness alone does not remove that concern. This limitation should be reported in the paper; avoid claiming source-level independence. An alternative source-disjoint evaluation would require re-partitioning source jets before aggregation and may require new aggregated arrays.

Only **per-configuration source overlap** was audited with the split generator; a global cross-configuration overlap audit was not reported.

## 6. Reproducible train/validation/test and cross-validation splits

`11_generate_dataset_splits.py` was run with:

```bash
python dataset_builder/11_generate_dataset_splits.py \
  --dataset-root "$DATASET_ROOT" \
  --seed 42 \
  --folds 5 \
  --overlap-scope per-config \
  --allow-source-overlap
```

The `--allow-source-overlap` flag was an **explicit, documented decision** to preserve the existing aggregates, not evidence that the underlying jets are independent.

| Primary partition | Total | Per configuration |
|---|---:|---:|
| Train | **11,200** | **320** |
| Validation | **1,400** | **40** |
| Held-out test | **1,400** | **40** |
| Total | **14,000** | **400** |

Five-fold cross-validation is defined **only on the 12,600-sample development pool (`train ∪ val`)**. Each of the five folds uses 10,080 training samples and 2,520 validation samples (72 validation samples per physics configuration); all 1,400 held-out test samples are excluded from every fold.

A separate, read-only verification passed all of the following:

- Primary partition sizes and path uniqueness.
- Pairwise aggregate-path disjointness across train/validation/test.
- Coverage of all 14,000 aggregates.
- Within-fold train/validation disjointness, with no test-path overlap.
- All five validation folds jointly cover the development pool exactly once.

**All representation and architecture ablations must use these same CSV partitions.** Keep `split_manifest.json` with the data and retain its recorded parameters and checksums.

## 7. Integration contract for T02

The new moment-aware PyTorch `Dataset` / `DataLoader` should:

1. Read pre-generated `splits/train.csv`, `splits/val.csv`, `splits/test.csv`, or the corresponding `five_fold/` CSV. **Never resplit automatically during model training.**
2. Resolve `agg_file_path` relative to the M1–M5 normalized dataset root.
3. Load each `(5,32,32)` normalized `float32` array lazily, with `np.load(..., allow_pickle=False)`.
4. Select requested moment indices without recomputation: e.g. `[1]`, `[1,2]`, `[1,2,3]`, `[1,2,3,4]`, `[1,2,3,4,5]`. Map one-based moment IDs to zero-based tensor indices.
5. Return an image tensor of shape `(C,32,32)` and labels mapped deterministically from physical values: `alpha` → `0..6`, `q0` → `0..4`.
6. Keep DataLoader-side tensors on CPU; transfer to GPU in the training loop with appropriate pinned-memory/nonblocking options.
7. Avoid on-the-fly aggregation and **avoid a second normalization** of pre-normalized arrays.
8. Preserve legacy loaders and three-head training behavior; the new dataset contains **only `alpha` and `q0`** labels, so its training/evaluation path must not require an energy-loss target.
9. Expose the chosen channel count to the model input adapter / patch embedding. The existing `models/model_vit.py` uses a standard `timm` ViT rather than ViT-CoMer; the dedicated ViT-CoMer integration is a separate model task.
10. Validate that every ablation uses identical record identities and train/test partitions and that only the intended input moments differ.

The existing repository's legacy loader, label contract, model, and training-loop compatibility issues were reviewed conceptually during T01, based on a provided repository snapshot. Actual end-to-end regression testing on the latest HPC checkout remains part of T02.

## 8. Reproduction commands

Assume `DATASET_ROOT` points to the normalized `..._agg500_m1-5_normalized` directory:

```bash
# Full array-format/range validation (14,000 tensors)
python dataset_builder/08_validate_moment_tensors.py \
  --root "$DATASET_ROOT" --workers 16

# Sign-preservation unit test (no dataset I/O)
python dataset_builder/09_test_moment_sign_preservation.py

# Sampled end-to-end numerical verification
python dataset_builder/10_verify_moment_consistency.py \
  --dataset-root "$DATASET_ROOT" \
  --unfolded-root "$BASE_ROOT/pTHat50-250_eCM5020_c0005_pT90-110_R0p4" \
  --samples-per-config 1 --device cuda

# Read-only connected-component analysis
python dataset_builder/12_analyze_source_overlap.py \
  --dataset-root "$DATASET_ROOT"
```

The metadata deduplication operation (`07_deduplicate_metadata.py`) was already performed. Do **not** rerun it against the canonical directory without checking its source/destination assumptions and existing output directories. Likewise, split generation has already completed; do **not** overwrite published split CSVs casually.

## 9. Final disposition and open follow-ups

### Established by tests

- Dataset comprises 35 balanced configurations and 14,000 distinct saved aggregate groups.
- Cleaned metadata comprises 35 × 400 unique aggregate references and no exact duplicate rows.
- All aggregate files exist and pass full-dataset shape, dtype, finiteness, and normalized-range validation.
- Signed normalization works on synthetic negative/positive inputs.
- Thirty-five sampled aggregates match recomputation from source jets to approximately float32 precision.
- Fixed 80/10/10 and five-fold CSV partitions pass aggregate-level exclusivity tests.

### Qualifications / remaining engineering work

- **Builder regression:** Corrected `04_build_agg500_m1to5_dataset.py` was assumed operational for T01; a production-HPC smoke test and idempotency rerun were not evidenced in the conversation. Record this explicitly rather than claiming it passed.
- **Source overlap:** Source-jet reuse between otherwise distinct aggregates is known and expressly allowed in the saved benchmark splits. The resulting validation/test estimates are not source-disjoint estimates.
- **Compatibility:** New moment-loader integration and legacy training regression tests belong to T02.
- **Paper claims:** Distinguish central moments from standardized skewness/kurtosis and state explicitly whether an experiment uses joint 35-way classification, factorized 7-/5-way heads, or both; do not imply a 70-way energy-loss task unless energy-loss labels are actually present.

**T01 issue #17 can be closed once this report is committed and the team accepts the documented builder regression caveat.**
