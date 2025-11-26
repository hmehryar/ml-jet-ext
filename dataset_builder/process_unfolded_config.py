#!/usr/bin/env python3
import argparse
import csv
import math
import os
import re
from pathlib import Path

import numpy as np


PHI_RANGE_DEFAULT = (-0.5, 0.5)
BIN_COUNT_DEFAULT = 32
ETA_RANGE_DEFAULT = (-0.5, 0.5)


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


def iter_jets_from_dat(dat_path: Path):
    """
    Generator over jets in one .dat file.

    - Assumes first line is header "index    particle_id   ...".
    - Events are separated by a line where all six entries are zero.
    - Yields np.ndarray with shape (N_particles, 3) and columns:
        [:,0] = phi
        [:,1] = eta
        [:,2] = pT
    """
    jets = []
    current_jet = []

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
                # End of current jet
                if current_jet:
                    arr = np.array(current_jet, dtype=np.float32)
                    yield arr
                    current_jet = []
                continue

            # Regular particle line:
            # index, particle_id, E, pT, phi, eta
            pT = vals[3]
            phi = vals[4]
            eta = vals[5]
            current_jet.append((phi, eta, pT))

    # In case there's data without a trailing sentinel
    if current_jet:
        arr = np.array(current_jet, dtype=np.float32)
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
    seed_stats_writer=None,
    seed_stats_rows=None,
):
    """
    Process one .dat file:
      - Read all jets
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

    # All per-jet rows for this seed will be stored here first
    seed_rows = []

    # --- per-seed accumulators ---
    phi_min_seed = math.inf
    phi_max_seed = -math.inf
    eta_min_seed = math.inf
    eta_max_seed = -math.inf
    pT_min_seed  = math.inf
    pT_max_seed  = -math.inf

    # Iterate over jets in this .dat
    local_idx = 0
    for event_item in iter_jets_from_dat(dat_path):
        if event_item.size == 0:
            continue

        # event_item columns: (phi, eta, pT)
        # For histogram: event_v[:,0]=phi, event_v[:,1]=eta, weights=pT
        phi_vals = event_item[:, 0]
        phi_vals = np.mod(phi_vals + math.pi, 2.0 * math.pi) - math.pi  # Wrap to [-pi, pi]
        eta_vals = event_item[:, 1]
        pT_vals = event_item[:, 2]

        # --- per-jet min/max for phi, eta, pT ---
        if phi_vals.size > 0:
            phi_min = float(phi_vals.min())
            phi_max = float(phi_vals.max())
        else:
            phi_min = float("nan")
            phi_max = float("nan")

        if eta_vals.size > 0:
            eta_min = float(eta_vals.min())
            eta_max = float(eta_vals.max())
        else:
            eta_min = float("nan")
            eta_max = float("nan")

        if pT_vals.size > 0:
            pT_min = float(pT_vals.min())
            pT_max = float(pT_vals.max())
        else:
            pT_min = float("nan")
            pT_max = float("nan")

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


        # build row for this jet and keep it local to this seed
        seed_rows.append(
            {
                "file_paths": str(rel_path),
                "alpha": alpha,
                "q0": q0,
                "seed": seed,
                "phi_min": phi_min,
                "phi_max": phi_max,
                "eta_min": eta_min,
                "eta_max": eta_max,
                "pT_min": pT_min,
                "pT_max": pT_max,
            }
        )

        global_event_counter += 1
        local_idx += 1

        # Progress print every 1000 jets (global)
        if global_event_counter % 1000 == 0:
            print(f"Saved {global_event_counter} jet...")

    # After finishing all jets of this seed
    if local_idx > 0:
        # compute per-seed min/max from the per-jet rows
        phi_min_seed = min(row["phi_min"] for row in seed_rows)
        phi_max_seed = max(row["phi_max"] for row in seed_rows)
        eta_min_seed = min(row["eta_min"] for row in seed_rows)
        eta_max_seed = max(row["eta_max"] for row in seed_rows)
        pT_min_seed  = min(row["pT_min"]  for row in seed_rows)
        pT_max_seed  = max(row["pT_max"]  for row in seed_rows)

        # relative path to this seed directory from output_root
        seed_dir_rel = str(label_dir.relative_to(output_root))

        # build a single dict for this seed
        seed_stats_row = {
            "seed_dir": seed_dir_rel,
            "seed": seed,
            "alpha": alpha,
            "q0": q0,
            "n_events": local_idx,
            "phi_min": phi_min_seed,
            "phi_max": phi_max_seed,
            "eta_min": eta_min_seed,
            "eta_max": eta_max_seed,
            "pT_min": pT_min_seed,
            "pT_max": pT_max_seed,
        }
        if seed_stats_writer is not None:
            seed_stats_writer.writerow(seed_stats_row)

        # also keep a copy in memory for global stats
        if seed_stats_rows is not None:
            seed_stats_rows.append(seed_stats_row)

        print(
            f"[INFO]   Seed {seed:03d}: finished with {local_idx} jets \n"
            f"\t(phi [{phi_min_seed:.4g}, {phi_max_seed:.4g}], \n"
            f"\teta [{eta_min_seed:.4g}, {eta_max_seed:.4g}], \n"
            f"\tpT [{pT_min_seed:.4g}, {pT_max_seed:.4g}]) \n"
        )

        # finally append this seed's rows to the global CSV list
        csv_rows.extend(seed_rows)

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
    seed_stats_rows = []   # NEW: keep per-seed rows in memory

    # # Write per-seed stats CSV
    seed_csv_path = metadata_dir / f"{config_name}_seed_stats.csv"
    with seed_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed_dir",
                "seed",
                "alpha",
                "q0",
                "n_events",
                "phi_min",
                "phi_max",
                "eta_min",
                "eta_max",
                "pT_min",
                "pT_max",
            ],
        )
        writer.writeheader()
        # writer.writerows(seed_stats.values())

        # Process all jetmod_* .dat files in this config dir
        # turn [0,1,2] to *
        for dat_path in sorted(config_dir.glob("jetmod_*_final_state_hadrons.dat")):
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
                seed_stats_writer=writer,
                seed_stats_rows=seed_stats_rows, 
            )
        if seed_stats_rows:
            total_events = sum(r["n_events"] for r in seed_stats_rows)
            phi_min_global = min(r["phi_min"] for r in seed_stats_rows)
            phi_max_global = max(r["phi_max"] for r in seed_stats_rows)
            eta_min_global = min(r["eta_min"] for r in seed_stats_rows)
            eta_max_global = max(r["eta_max"] for r in seed_stats_rows)
            pT_min_global  = min(r["pT_min"]  for r in seed_stats_rows)
            pT_max_global  = max(r["pT_max"]  for r in seed_stats_rows)

            writer.writerow(
                {
                    "seed_dir": config_name,  # as you requested
                    "seed": "",                   # empty
                    "alpha": alpha,
                    "q0": q0,
                    "n_events": total_events,
                    "phi_min": phi_min_global,
                    "phi_max": phi_max_global,
                    "eta_min": eta_min_global,
                    "eta_max": eta_max_global,
                    "pT_min": pT_min_global,
                    "pT_max": pT_max_global,
                }
            )

    # Write per-config CSV
    csv_path = metadata_dir / f"{config_name}_jets.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file_paths",
                "alpha",
                "q0",
                "seed",
                "phi_min",
                "phi_max",
                "eta_min",
                "eta_max",
                "pT_min",
                "pT_max",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    
    print(f"[INFO] Finished config {config_name}.")
    print(f"[INFO]   Total jets saved: {global_event_counter}")
    print(f"[INFO]   Metadata CSV (per-jet): {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Process one *_unfolded jet config into per-jet .npy images."
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