"""Audit and safely migrate T04-06 Slurm logs using registry job IDs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile


REGISTRY = Path(
    "training_output/T04/t04_06_v2/registry.runtime.json"
)

DEFAULT_ROOT = Path(
    "/mnt/Research/arsalan/"
    "wsu_grid_rs_grp_mljet/ml-jet-ext"
)

JOB_PATTERN = re.compile(r"-(\d+)$")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--move", action="store_true")
    parser.add_argument(
        "--root", type=Path, default=DEFAULT_ROOT
    )
    args = parser.parse_args()

    repo = Path.cwd().resolve()
    root = args.root.resolve()
    dest = root / "slurm_logs/T04/t04_06_v2"

    if not root.is_dir():
        raise RuntimeError(
            f"Research storage not mounted/accesssible: {root}"
        )

    registry = json.loads((repo / REGISTRY).read_text())
    entries = registry["entries"]

    if len(entries) != 120:
        raise RuntimeError("Expected 120 experiment entries")

    if any(e["status"] != "completed" for e in entries.values()):
        raise RuntimeError("Not all experiments completed")

    jobs = {}
    for identity, entry in entries.items():
        for attempt in entry["attempts"]:
            job = str(attempt["slurm_job_id"])
            if job in jobs:
                raise RuntimeError(f"Duplicate Slurm job ID: {job}")
            jobs[job] = identity

    found = {}
    unrelated = []

    for file in sorted(repo.iterdir()):
        if not file.is_file() or file.suffix not in (".out", ".err"):
            continue

        match = JOB_PATTERN.search(file.stem)
        job = match.group(1) if match else None

        if job in jobs:
            found.setdefault(job, []).append(file)
        else:
            unrelated.append(file)

    files = [
        p for group in found.values() for p in group
    ]

    print("=== T04-06 SLURM LOG AUDIT ===")
    print("Completed experiments:", len(entries))
    print("Registered Slurm attempts:", len(jobs))
    print("Attempts with logs:", len(found))
    print("Matched log files:", len(files))
    print("Unrelated root logs (excluded):", len(unrelated))
    print("Total log bytes:", sum(p.stat().st_size for p in files))
    print("Destination:", dest)

    missing = []

    for job, identity in sorted(jobs.items()):
        names = {p.suffix for p in found.get(job, [])}
        absent = {".out", ".err"} - names

        print(
            identity,
            "job=" + job,
            "logs=" + ",".join(sorted(names)) if names else "NO_LOGS"
        )

        if absent:
            missing.append((identity, job, sorted(absent)))

    print("\n=== MISSING LOG REPORT ===")
    print("Attempts with missing log types:", len(missing))

    for identity, job, suffixes in missing:
        print("MISSING", identity, job, ",".join(suffixes))

    if not args.move:
        print("\nAUDIT ONLY — no files moved")
        print("Use --move to migrate the matched logs")
        return

    if not files:
        raise RuntimeError("No matching logs to move")

    dest.mkdir(parents=True, exist_ok=True)

    # Preflight every destination before moving any file.
    for src in files:
        target = dest / src.name
        if target.exists() and sha256(src) != sha256(target):
            raise RuntimeError(
                f"Destination collision with different contents: {target}"
            )

    moved = 0
    already_present = 0

    for src in files:
        target = dest / src.name

        if target.exists():
            already_present += 1
        else:
            fd, temporary = tempfile.mkstemp(
                prefix=".t04-log-", dir=str(dest)
            )
            os.close(fd)
            temporary = Path(temporary)

            try:
                shutil.copy2(src, temporary)

                if sha256(src) != sha256(temporary):
                    raise RuntimeError(
                        f"Checksum mismatch: {src}"
                    )

                os.replace(temporary, target)
            finally:
                if temporary.exists():
                    temporary.unlink()

            moved += 1

        # Remove source only after content is verified at destination.
        if sha256(src) != sha256(target):
            raise RuntimeError(
                f"Final verification failed: {src}"
            )

        src.unlink()

    print("\nMIGRATION COMPLETE")
    print("Files newly copied:", moved)
    print("Files already present:", already_present)
    print("Source files removed after verification:", len(files))
    print("Destination:", dest)


if __name__ == "__main__":
    main()
