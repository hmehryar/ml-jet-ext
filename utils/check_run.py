#!/usr/bin/env python
# coding: utf-8

# In[4]:


# ML-JET — Job Result Checker (step-by-step)
from pathlib import Path
import pandas as pd


# In[23]:


from pathlib import Path
from dataclasses import dataclass
import pandas as pd
from typing import List, Dict, Any

@dataclass
class RunScanConfig:
    run_dir: Path
    prefix: str = "jetmod_"
    suffix: str = "_final_state_hadrons.dat"
    tail_bytes: int = 16384
    max_files: int = None  # if set (e.g. 5), only first 5 files are scanned

class RunScanner:
    def __init__(self, cfg: RunScanConfig):
        self.cfg = cfg
        self.df = pd.DataFrame()

    def non_empty(self, p: Path) -> bool:
        try:
            return p.is_file() and p.stat().st_size > 0
        except Exception:
            return False

    def last_line_fast(self, path: Path) -> str:
        try:
            with open(path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                if size == 0:
                    return ""
                read_len = min(size, self.cfg.tail_bytes)
                f.seek(size - read_len)
                buf = f.read(read_len)
            parts = buf.splitlines()
            return parts[-1].decode(errors="replace") if parts else ""
        except Exception:
            return ""

    def is_complete(self, path: Path) -> bool:
        if not path.exists() or not self.non_empty(path):
            return False
        line = self.last_line_fast(path)
        return ("sigmaGen" in line) and ("sigmaErr" in line)

    # def scan(self) -> pd.DataFrame:
    #     files = sorted(self.cfg.run_dir.glob(f"{self.cfg.prefix}*{self.cfg.suffix}"))
    #     if self.cfg.max_files:
    #         files = files[:self.cfg.max_files]

    #     rows: List[Dict[str, Any]] = []
    #     for f in files:
    #         name = f.name
    #         k_str = name[len(self.cfg.prefix):-len(self.cfg.suffix)]
    #         try:
    #             k = int(k_str)
    #         except ValueError:
    #             k = None
    #         rows.append({
    #             "k": k,
    #             "file": str(f),
    #             "exists": f.exists(),
    #             "nonempty": self.non_empty(f),
    #             "complete": self.is_complete(f),
    #         })
    #     self.df = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    #     self.save_results()
    #     return self.df
    # import pandas as pd

    # def add_missing_indices(self, df: pd.DataFrame, idx_col: str = "idx",
    #                         start: int = 0, end: int = 199) -> pd.DataFrame:
    #     """
    #     Ensure rows for every index in [start, end]. For indices not present,
    #     append rows with exists=False, nonempty=False, complete=False.
    #     Keeps existing rows/values intact and returns a sorted DataFrame.
    #     """
    #     if idx_col not in df.columns:
    #         raise ValueError(f"DataFrame must contain '{idx_col}' column.")

    #     df = self.df.copy()
    #     # Work with the index set to idx_col for easy alignment
    #     df = df.set_index(idx_col, drop=False)

    #     expected = pd.Index(range(start, end + 1), name=idx_col)
    #     missing = expected.difference(df.index)

    #     # Make sure the 3 flag columns exist; if not, create (don’t override existing)
    #     for col in ("exists", "nonempty", "complete"):
    #         if col not in df.columns:
    #             df[col] = pd.NA  # preserve unknowns for existing rows

    #     # Build rows for missing indices with required defaults
    #     if len(missing) > 0:
    #         missing_df = pd.DataFrame({
    #             idx_col: missing,
    #             "exists": False,
    #             "nonempty": False,
    #             "complete": False,
    #         }).set_index(idx_col, drop=False)

    #         # Ensure missing_df has any extra columns your df has (as NA)
    #         for col in df.columns:
    #             if col not in missing_df.columns:
    #                 missing_df[col] = pd.NA

    #         df = pd.concat([df, missing_df], axis=0, sort=False)

    #     # Reindex to sorted 0..199 order; keep existing values intact
    #     df = df.reindex(expected)

    #     # Final tidy up
    #     self.df = df.reset_index(drop=True)

    #     return self.df

    def scan(self) -> pd.DataFrame:
        """
        Loop over expected jetmod_0..jetmod_199 files and record:
        - exists: file is present on disk
        - nonempty: size > 0 (only meaningful if exists)
        - complete: last line contains the completion markers (only if exists)
        If a file is missing, record exists=False, nonempty=False, complete=False.
        """
        # Respect max_files for smoke tests (0..max_files-1), else full 0..199
        last_k = 199 if self.cfg.max_files is None else min(199, self.cfg.max_files - 1)

        rows = []
        for k in range(0, last_k + 1):
            fname = f"{self.cfg.prefix}{k}{self.cfg.suffix}"
            p = self.cfg.run_dir / fname

            exists = p.is_file()
            nonempty = self.non_empty(p) if exists else False
            complete = self.is_complete(p) if exists else False  # checks for proper ending

            rows.append({
                "k": k,
                "file": str(p),
                "exists": exists,
                "nonempty": nonempty,
                "complete": complete,
            })

        self.df = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
        self.save_results()
        return self.df

    def save_results(self):
        if self.df is None or self.df.empty:
            print("[WARN] No data to save.")
            return
        out_path = self.cfg.run_dir / "scan_summary.csv"
        self.df.to_csv(out_path, index=False)
        print(f"[INFO] Saved scan results to: {out_path}")
    
    def get_problem_jobs(self) -> list[int]:
        """
        Return a list of k values for jobs that are missing, empty, or incomplete.
        """
        if self.df is None:
            self.scan()
            self.add_missing_indices(self.df, idx_col="k", start=0, end=199)

        if self.df is None or self.df.empty:
            return []

        bad_rows = self.df[
            (~self.df["exists"]) | (~self.df["nonempty"]) | (~self.df["complete"])
        ]
        ks = [int(k) for k in bad_rows["k"].dropna().tolist()]
        return ks

    def save_problem_jobs(self, filename: str = "problem_jobs.csv"):
        """
        Save the list of bad job numbers as a single comma-separated line.
        """
        ks = self.get_problem_jobs()
        out_path = self.cfg.run_dir / filename
        with open(out_path, "w") as f:
            f.write(",".join(map(str, ks)))
        print(f"[INFO] Saved {len(ks)} problem jobs to {out_path}")
        return ks


# In[ ]:


# === MAIN ENTRYPOINT ===
# Allows running this notebook as a standalone script with arguments
# Example:
# !python -m nbconvert --to notebook --execute check_run.ipynb --args --run_dir /path/to/run --max_files 10

import argparse
import sys

def main(run_dir: str = None, max_files: int = None):
    # Allow both CLI args and direct call
    # Only parse argv if *our* flag is present; otherwise expect direct args.
    if run_dir is None and "--run_dir" in sys.argv:
        parser = argparse.ArgumentParser(description="Scan ML-JET run directory")
        parser.add_argument("--run_dir", type=str, required=True, help="Path to run directory")
        parser.add_argument("--max_files", type=int, default=None, help="Number of files to scan (None=all)")
        
        args, _ = parser.parse_known_args()   # don't crash on Jupyter's extra flags
        run_dir = args.run_dir
        max_files = args.max_files
    
    if run_dir is None:
        print("[INFO] Notebook mode: call main('/path/to/run', 5) or include --run_dir on CLI.")
        return []

    run_dir_path = Path(run_dir)
    cfg = RunScanConfig(run_dir=run_dir_path, max_files=max_files)
    scanner = RunScanner(cfg)

    print(f"[INFO] Scanning {run_dir_path} (max_files={max_files})")
    scanner.scan()
    
    bad_jobs = scanner.get_problem_jobs()
    scanner.save_problem_jobs()
    print(f"[DONE] Found {len(bad_jobs)} problem jobs:", bad_jobs)
    return bad_jobs




# In[ ]:


# === Batch runner over all 35 run directories ===
from pathlib import Path

if __name__ == "__main__":
    # Define base path and run directories
    BASE = Path("/rs/rs_grp_mljet/ml-jet-ext/simulation_results_pTHat50-250_eCM5020/")
    #BASE = Path("/rs/rs_grp_mljet/ml-jet-ext/simulation_results_pTHat100-110_eCM5020/")

    RUN_DIRS = [
        # "c0005_alpha0p20_q_1p0",
        # "c0005_alpha0p20_q_1p5",
        # "c0005_alpha0p20_q_2p0",
        # "c0005_alpha0p20_q_2p5",
        # "c0005_alpha0p20_q_3p0",
        # "c0005_alpha0p25_q_1p0",
        # "c0005_alpha0p25_q_1p5",
        # "c0005_alpha0p25_q_2p0",
        # "c0005_alpha0p25_q_2p5", 
        # "c0005_alpha0p25_q_3p0",
        # "c0005_alpha0p30_q_1p0", 
        # "c0005_alpha0p30_q_1p5",
        # "c0005_alpha0p30_q_2p0",
        # "c0005_alpha0p30_q_2p5",
        # "c0005_alpha0p30_q_3p0",
        "c0005_alpha0p35_q_1p0",
        "c0005_alpha0p35_q_1p5",
        "c0005_alpha0p35_q_2p0",
        "c0005_alpha0p35_q_2p5", 
        "c0005_alpha0p35_q_3p0",
        "c0005_alpha0p40_q_1p0", 
        "c0005_alpha0p40_q_1p5",
        "c0005_alpha0p40_q_2p0",
        "c0005_alpha0p40_q_2p5", 
        "c0005_alpha0p40_q_3p0",
        "c0005_alpha0p45_q_1p0", 
        "c0005_alpha0p45_q_1p5", 
        # "c0005_alpha0p45_q_2p0",
        # "c0005_alpha0p45_q_2p5", 
        # "c0005_alpha0p45_q_3p0",
        # "c0005_alpha0p50_q_1p0",
        # "c0005_alpha0p50_q_1p5", 
        # "c0005_alpha0p50_q_2p0",
        # "c0005_alpha0p50_q_2p5", 
        # "c0005_alpha0p50_q_3p0",
    ]

    for name in RUN_DIRS:
        run_path = BASE / name
        print(f"\n=== Scanning: {run_path} ===")
        try:
            # None => scan ALL files; set an int to limit per-dir if you want a quick smoke test
            main(str(run_path), None)
        except Exception as e:
            print(f"[ERROR] {run_path}: {e}")


# In[ ]:


# main("/home/arsalan/wsu_grid_rs_grp_mljet/simulation_results/c0005_alpha0p20_q_1p0", 5)

