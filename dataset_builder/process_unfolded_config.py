#!/usr/bin/env python3
import argparse
import csv
import math
import os
import re
from pathlib import Path

import numpy as np


# BIN_COUNT_DEFAULT = 32
PHI_RANGE_DEFAULT = (-math.pi, math.pi)
# PHI_RANGE_DEFAULT = (-0.5, 0.5)
ETA_RANGE_DEFAULT = (-2.8, 2.8)
BIN_COUNT_DEFAULT = 32
# PHI_RANGE_DEFAULT = (0.0, 2.0 * math.pi)
# ETA_RANGE_DEFAULT = (-0.5, 0.5)


JETMOD_PATTERN = re.compile(r"jetmod_(\d+)_final_state_hadrons\.dat")


def parse_alpha_q0_from_config_name(config_name: str):
    """
    Example config_name: c0005_alpha0p20_q_1p0_unfolded
    alpha0p20  -> 0.20
    q_1p0      -> 1.0
    """
    m = re.search(r"alpha0p(\d+)_q_(\d+p\d+)", config_name)
    if not m:
        raise ValueError(f"Could not parse alpha/q0 from config name '{config_name}'")

    alpha_str = m.group(1)       # e.g. "20"
    q0_str = m.group(2)          # e.g. "1p0"

    alpha = float("0." + alpha_str)   # -> 0.20
    q0 = float(q0_str.replace("p", "."))  # -> 1.0

    return alpha, q0


def iter_events_from_dat(dat_path: Path):
    """
    Generator over events in one .dat file.

    - Assumes first line is header "index    particle_id   ...".
    - Events are separated by a line where all six entries are zero.
    - Yields np.ndarray with shape (N_particles, 3) and columns:
        [:,0] = phi
        [:,1] = eta
        [:,2] = pT
    """
    events = []
    current_event = []

    with dat_path.open("r") as f:
        # Skip header line
        header = next(f, None)

        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 6:
                continue

            # Sentinel line? (all six entries are zeros)
            try:
                vals = [float(p) for p in parts[:6]]
            except ValueError:
                # If something unexpected, just skip this line
                continue

            if all(abs(v) < 1e-12 for v in vals):
                # End of current event
                if current_event:
                    arr = np.array(current_event, dtype=np.float32)
                    yield arr
                    current_event = []
                continue

            # Regular particle line:
            # index, particle_id, E, pT, phi, eta
            pT = vals[3]
            phi = vals[4]
            eta = vals[5]
            current_event.append((phi, eta, pT))

    # In case there's data without a trailing sentinel
    if current_event:
        arr = np.array(current_event, dtype=np.float32)
        yield arr


def process_dat_file(
    dat_path: Path,
    output_root: Path,
    config_name: str,
    alpha: float,
    q0: float,
    bin_count: int,
    phi_range,
    eta_range,
    global_event_counter: int,
    csv_rows: list,
):
    """
    Process one .dat file:
      - Read all events
      - Make hist2d images (phi, eta) weighted by pT
      - Save as .npy in config/seed_X
      - Append metadata rows to csv_rows

    Returns updated global_event_counter.
    """
    m = JETMOD_PATTERN.match(dat_path.name)
    if not m:
        # Not a jetmod_* file; skip
        return global_event_counter

    seed = int(m.group(1))
    label_dir = output_root / config_name / f"seed_{seed:03d}"
    label_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over events in this .dat
    local_idx = 0
    for event_item in iter_events_from_dat(dat_path):
        if event_item.size == 0:
            continue

        # event_item columns: (phi, eta, pT)
        # For histogram: event_v[:,0]=phi, event_v[:,1]=eta, weights=pT
        phi_vals = event_item[:, 0]
        phi_vals = np.mod(phi_vals + math.pi, 2.0 * math.pi) - math.pi  # Wrap to [-pi, pi]
        eta_vals = event_item[:, 1]
        pT_vals = event_item[:, 2]

        counts, xedges, yedges = np.histogram2d(
            phi_vals,
            eta_vals,
            bins=bin_count,
            range=[phi_range, eta_range],
            weights=pT_vals,
        )

        image = counts.astype(np.float32)

        # File name within this seed directory
        event_fname = f"event_{local_idx:07d}.npy"
        event_path = label_dir / event_fname
        np.save(event_path, image)

        # Relative path from output_root (for CSV)
        rel_path = event_path.relative_to(output_root)

        csv_rows.append(
            {
                "file_paths": str(rel_path),
                "alpha": alpha,
                "q0": q0,
                "seed": seed,
            }
        )

        global_event_counter += 1
        local_idx += 1

        # Progress print every 1000 events (global)
        if global_event_counter % 1000 == 0:
            print(f"Saved {global_event_counter} event...")

    return global_event_counter


def process_config_dir(config_dir: Path, output_root: Path, bin_count: int):
    config_name = config_dir.name
    alpha, q0 = parse_alpha_q0_from_config_name(config_name)

    print(f"[INFO] Processing config: {config_name} (alpha={alpha}, q0={q0})")
    print(f"[INFO] Input dir: {config_dir}")
    print(f"[INFO] Output root: {output_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    csv_rows = []
    global_event_counter = 0

    # Process all jetmod_* .dat files in this config dir
    # turn 0 to *
    for dat_path in sorted(config_dir.glob("jetmod_0_final_state_hadrons.dat")):
        print(f"[INFO]   Reading {dat_path.name}")
        global_event_counter = process_dat_file(
            dat_path=dat_path,
            output_root=output_root,
            config_name=config_name,
            alpha=alpha,
            q0=q0,
            bin_count=bin_count,
            phi_range=PHI_RANGE_DEFAULT,
            eta_range=ETA_RANGE_DEFAULT,
            global_event_counter=global_event_counter,
            csv_rows=csv_rows,
        )

    # Write per-config CSV
    csv_path = metadata_dir / f"{config_name}_events.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_paths", "alpha", "q0", "seed"])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"[INFO] Finished config {config_name}.")
    print(f"[INFO]   Total events saved: {global_event_counter}")
    print(f"[INFO]   Metadata CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Process one *_unfolded jet config into per-event .npy images."
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        required=True,
        help="Path to a single *_unfolded config directory.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Root directory where processed .npy files and metadata CSVs are stored.",
    )
    parser.add_argument(
        "--bin-count",
        type=int,
        default=BIN_COUNT_DEFAULT,
        help="Number of bins for both phi and eta (default: 32).",
    )

    args = parser.parse_args()

    config_dir = Path(args.config_dir).resolve()
    output_root = Path(args.output_root).resolve()

    if not config_dir.is_dir():
        raise SystemExit(f"Config dir does not exist or is not a directory: {config_dir}")

    process_config_dir(config_dir, output_root, args.bin_count)


if __name__ == "__main__":
    main()

# python process_unfolded_config.py \
#   --config-dir /rs/rs_grp_mljet/ml-jet-ext/simulation_results_pTHat50-250_eCM5020/c0005_alpha0p20_q_1p0_unfolded \
#   --output-root /rs/rs_grp_mljet/ml-jet-ext/mljet_agg_pTHat50-250_eCM5020