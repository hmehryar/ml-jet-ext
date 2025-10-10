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

    def scan(self) -> pd.DataFrame:
        files = sorted(self.cfg.run_dir.glob(f"{self.cfg.prefix}*{self.cfg.suffix}"))
        if self.cfg.max_files:
            files = files[:self.cfg.max_files]

        rows: List[Dict[str, Any]] = []
        for f in files:
            name = f.name
            k_str = name[len(self.cfg.prefix):-len(self.cfg.suffix)]
            try:
                k = int(k_str)
            except ValueError:
                k = None
            rows.append({
                "k": k,
                "file": str(f),
                "exists": f.exists(),
                "nonempty": self.non_empty(f),
                "complete": self.is_complete(f),
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
    BASE = Path("/rs/rs_grp_mljet/ml-jet-ext/simulation_results/")
    RUN_DIRS = [
        "c0005_alpha0p20_q_1p0", "c0005_alpha0p20_q_1p5", "c0005_alpha0p20_q_2p0",
        "c0005_alpha0p20_q_2p5", "c0005_alpha0p20_q_3p0",
        "c0005_alpha0p25_q_1p0", "c0005_alpha0p25_q_1p5", "c0005_alpha0p25_q_2p0",
        "c0005_alpha0p25_q_2p5", "c0005_alpha0p25_q_3p0",
        "c0005_alpha0p30_q_1p0", "c0005_alpha0p30_q_1p5", "c0005_alpha0p30_q_2p0",
        "c0005_alpha0p30_q_2p5", "c0005_alpha0p30_q_3p0",
        "c0005_alpha0p35_q_1p0", "c0005_alpha0p35_q_1p5", "c0005_alpha0p35_q_2p0",
        "c0005_alpha0p35_q_2p5", "c0005_alpha0p35_q_3p0",
        "c0005_alpha0p40_q_1p0", "c0005_alpha0p40_q_1p5", "c0005_alpha0p40_q_2p0",
        "c0005_alpha0p40_q_2p5", "c0005_alpha0p40_q_3p0",
        "c0005_alpha0p45_q_1p0", "c0005_alpha0p45_q_1p5", "c0005_alpha0p45_q_2p0",
        "c0005_alpha0p45_q_2p5", "c0005_alpha0p45_q_3p0",
        "c0005_alpha0p50_q_1p0", "c0005_alpha0p50_q_1p5", "c0005_alpha0p50_q_2p0",
        "c0005_alpha0p50_q_2p5", "c0005_alpha0p50_q_3p0",
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

