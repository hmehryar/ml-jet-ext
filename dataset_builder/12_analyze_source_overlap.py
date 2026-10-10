
#!/usr/bin/env python3
"""
Analyze source-jet overlap between aggregated images.

Read-only: does not modify metadata or NPY files.
"""

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd


def parse_paths(value):
    if not isinstance(value, str):
        raise ValueError("Invalid source_file_paths value")

    return [p.strip() for p in value.split("|") if p.strip()]


def analyze_config(csv_file):
    df = pd.read_csv(csv_file)

    n = len(df)
    parent = list(range(n))
    sizes = [1] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)

        if a == b:
            return

        if sizes[a] < sizes[b]:
            a, b = b, a

        parent[b] = a
        sizes[a] += sizes[b]

    first_seen = {}
    repeated_occurrences = 0

    for i, row in enumerate(df.itertuples(index=False)):
        # Use the actual CSV column, not its tuple position.
        paths = parse_paths(
            getattr(row, "source_file_paths")
        )

        if len(paths) != 500 or len(set(paths)) != 500:
            raise ValueError(
                f"{csv_file.name}: invalid aggregate row {i}"
            )

        for source in paths:
            if source in first_seen:
                repeated_occurrences += 1
                union(i, first_seen[source])
            else:
                first_seen[source] = i

    components = Counter(find(i) for i in range(n))
    component_sizes = sorted(
        components.values(), reverse=True
    )

    return {
        "config": csv_file.stem,
        "aggregates": n,
        "components": len(component_sizes),
        "largest_component": component_sizes[0],
        "component_sizes": component_sizes,
        "source_reuses": repeated_occurrences,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    metadata_dir = args.dataset_root / "metadata"
    csv_files = sorted(
        metadata_dir.glob("*_agg500_metadata.csv")
    )

    if len(csv_files) != 35:
        raise ValueError(
            f"Expected 35 CSV files, found {len(csv_files)}"
        )

    print(
        f"{'Configuration':<35} "
        f"{'Components':>11} "
        f"{'Largest':>9} "
        f"{'Reuses':>9}"
    )
    print("-" * 70)

    for csv_file in csv_files:
        result = analyze_config(csv_file)

        print(
            f"{result['config']:<35} "
            f"{result['components']:>11} "
            f"{result['largest_component']:>9} "
            f"{result['source_reuses']:>9}"
        )

        print(
            "  Largest component sizes:",
            result["component_sizes"][:10]
        )


if __name__ == "__main__":
    main()
