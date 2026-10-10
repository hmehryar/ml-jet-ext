
from pathlib import Path
from collections import Counter
import ast
import pandas as pd

# Run from the root of the M1-M5 dataset.
root = Path(".")
metadata_dir = root / "metadata"


def parse_paths(value):
    text = str(value).strip()

    if text.startswith("["):
        return [str(p) for p in ast.literal_eval(text)]

    if "|" in text:
        return [p.strip() for p in text.split("|") if p.strip()]

    return [text]


totals = Counter()

for csv_file in sorted(metadata_dir.glob("*.csv")):
    df = pd.read_csv(csv_file)
    config = csv_file.name.removesuffix("_agg500_metadata.csv")

    # Distinct records and output filenames
    unique_paths = df["agg_file_path"].nunique()
    exact_duplicates = df.duplicated().sum()

    # Canonicalize groups: jet order does not matter.
    signatures = df["source_file_paths"].map(
        lambda x: tuple(sorted(parse_paths(x)))
    )

    unique_groups = signatures.nunique()

    # Multiple filenames referring to identical 500-jet groups.
    group_to_paths = pd.DataFrame({
        "signature": signatures,
        "path": df["agg_file_path"],
    })

    paths_per_group = group_to_paths.groupby(
        "signature"
    )["path"].nunique()

    repeated_groups = int((paths_per_group > 1).sum())

    # Check physical files.
    config_dir = root / config
    disk_paths = {
        str(p.relative_to(root))
        for p in config_dir.glob("*.npy")
    }

    metadata_paths = set(df["agg_file_path"].astype(str))

    missing = metadata_paths - disk_paths
    unreferenced = disk_paths - metadata_paths

    # Check group membership for repeated source paths.
    invalid_groups = sum(
        len(group) != 500 or len(set(group)) != 500
        for group in signatures
    )

    print(f"\n{config}")
    print(f"  Metadata rows:          {len(df)}")
    print(f"  Exact duplicate rows:   {exact_duplicates}")
    print(f"  Unique output paths:    {unique_paths}")
    print(f"  Unique 500-jet groups:  {unique_groups}")
    print(f"  Groups with >1 path:    {repeated_groups}")
    print(f"  Invalid 500-jet groups: {invalid_groups}")
    print(f"  Physical NPY files:     {len(disk_paths)}")
    print(f"  Missing NPY files:      {len(missing)}")
    print(f"  Unreferenced NPY files: {len(unreferenced)}")

    totals["rows"] += len(df)
    totals["exact_duplicates"] += exact_duplicates
    totals["unique_paths"] += unique_paths
    totals["unique_groups"] += unique_groups
    totals["repeated_groups"] += repeated_groups
    totals["invalid_groups"] += invalid_groups
    totals["disk_files"] += len(disk_paths)
    totals["missing"] += len(missing)
    totals["unreferenced"] += len(unreferenced)

print("\n========== DATASET TOTALS ==========")

for key, value in totals.items():
    print(f"{key:<22}: {value}")
