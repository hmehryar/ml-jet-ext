from pathlib import Path
import pandas as pd

csv_files = sorted(Path(".").glob("*.csv"))

print(f"\nFound {len(csv_files)} CSV files\n")

print(
    f"{'Configuration':<38}"
    f"{'Total':>8}"
    f"{'Unique':>10}"
    f"{'Exact Dup':>12}"
    f"{'Path Dup':>12}"
)
print("-" * 80)

total_rows = 0
total_unique = 0
total_exact_duplicates = 0
total_path_duplicates = 0

for csv_file in csv_files:
    df = pd.read_csv(csv_file)

    if "agg_file_path" not in df.columns:
        print(f"SKIP: {csv_file.name} - missing agg_file_path")
        continue

    rows = len(df)
    unique = df["agg_file_path"].nunique()
    exact_duplicates = df.duplicated().sum()
    path_duplicates = df.duplicated(
        subset=["agg_file_path"]
    ).sum()

    total_rows += rows
    total_unique += unique
    total_exact_duplicates += exact_duplicates
    total_path_duplicates += path_duplicates

    print(
        f"{csv_file.stem:<38}"
        f"{rows:>8}"
        f"{unique:>10}"
        f"{exact_duplicates:>12}"
        f"{path_duplicates:>12}"
    )

    if path_duplicates > 0:
        duplicated = df[
            df.duplicated(
                subset=["agg_file_path"], keep=False
            )
        ]

        print("  Example duplicated paths:")
        for path in duplicated["agg_file_path"].unique()[:3]:
            print(f"    {path}")

        if path_duplicates != exact_duplicates:
            print("  WARNING: Duplicate paths have different metadata!")

print("-" * 80)
print(
    f"{'TOTAL':<38}"
    f"{total_rows:>8}"
    f"{total_unique:>10}"
    f"{total_exact_duplicates:>12}"
    f"{total_path_duplicates:>12}"
)

print("\nSummary:")
print(f"  Total CSV files:          {len(csv_files)}")
print(f"  Total metadata rows:      {total_rows}")
print(f"  Unique aggregate paths:   {total_unique}")
print(f"  Exact duplicate rows:     {total_exact_duplicates}")
print(f"  Duplicate path rows:      {total_path_duplicates}")

if total_path_duplicates == 0:
    print("\nRESULT: No duplicate aggregate paths found.")
else:
    print("\nRESULT: Duplicate aggregate paths detected.")