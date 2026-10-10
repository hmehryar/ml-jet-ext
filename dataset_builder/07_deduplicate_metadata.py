
#!/usr/bin/env python3
"""
Deduplicate M1-M5 aggregate metadata.

Reads CSVs from metadata/ and writes cleaned copies to metadata2/.
Preserves all columns and exactly one row per unique agg_file_path.
Does not modify the original CSVs or NPY files.
"""

from pathlib import Path
import argparse
import pandas as pd


def deduplicate_metadata(root: Path, expected_per_config=400):
    source_dir = root / "metadata"
    output_dir = root / "metadata2"

    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)

    csv_files = sorted(
        source_dir.glob("*_agg500_metadata.csv")
    )

    if len(csv_files) != 35:
        raise ValueError(
            f"Expected 35 metadata CSVs, found {len(csv_files)}"
        )

    # Validate everything before creating outputs.
    cleaned_files = []
    total_original = 0
    total_cleaned = 0

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        required = {
            "agg_file_path",
            "source_file_paths",
            "alpha",
            "q0",
            "config",
        }

        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"{csv_file.name}: missing columns {missing}"
            )

        # Reject repeated output paths with conflicting metadata.
        duplicates = df[df.duplicated(
            subset=["agg_file_path"], keep=False
        )]

        if not duplicates.empty:
            conflicts = (
                duplicates.groupby("agg_file_path", sort=False)
                .nunique(dropna=False)
            )
            conflicts = conflicts.drop(
                columns=["agg_file_path"], errors="ignore"
            )

            if (conflicts > 1).any().any():
                raise ValueError(
                    f"{csv_file.name}: conflicting metadata "
                    "for repeated aggregate paths"
                )

        # Retain exactly one occurrence of each aggregate.
        cleaned = df.drop_duplicates(
            subset=["agg_file_path"],
            keep="first",
        ).reset_index(drop=True)

        if len(cleaned) != expected_per_config:
            raise ValueError(
                f"{csv_file.name}: expected "
                f"{expected_per_config} unique aggregates, "
                f"found {len(cleaned)}"
            )

        # Ensure referenced arrays exist.
        for rel_path in cleaned["agg_file_path"]:
            path = root / str(rel_path)
            if not path.is_file():
                raise FileNotFoundError(path)

        cleaned_files.append((csv_file.name, cleaned))
        total_original += len(df)
        total_cleaned += len(cleaned)

        print(
            f"{csv_file.stem}: "
            f"{len(df)} -> {len(cleaned)} "
            f"(removed {len(df) - len(cleaned)})"
        )

    # Only create new metadata after validation succeeds.
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"{output_dir} is not empty. "
            "Refusing to overwrite existing files."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, cleaned in cleaned_files:
        cleaned.to_csv(output_dir / filename, index=False)

    print("\n========== SUMMARY ==========")
    print(f"Configurations:        {len(cleaned_files)}")
    print(f"Original CSV rows:      {total_original}")
    print(f"Cleaned CSV rows:       {total_cleaned}")
    print(f"Removed duplicate rows:{total_original-total_cleaned:>8}")
    print(f"Output directory:       {output_dir}")
    print("\nOriginal metadata and NPY files were not modified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root of the M1-M5 dataset",
    )
    args = parser.parse_args()

    deduplicate_metadata(args.root)
