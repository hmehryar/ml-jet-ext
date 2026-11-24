#!/usr/bin/env python3
"""Export audited T04-06 validation metrics from existing run artifacts."""
import csv
import json
import math
import os
import statistics
from collections import defaultdict
from itertools import product
from pathlib import Path

ROOT = Path(os.environ["T04_06_ROOT"]).resolve()
OUT = Path("results/t04_06/manuscript")
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ("clip", "siglip")
STRATEGIES = ("S1", "S2", "S3", "S4")
MOMENTS = ("E1", "E2", "E3", "E4", "E5")
SEEDS = (42, 123, 2026)
METRICS = (
    "val_alpha_accuracy",
    "val_q0_accuracy",
    "val_joint_accuracy",
)

records = []
problems = []
multi_session = []

for model, strategy, moment, seed in product(
    MODELS, STRATEGIES, MOMENTS, SEEDS
):
    parent = ROOT / model / strategy / moment / f"seed-{seed}"
    candidates = [p for p in parent.iterdir() if p.is_dir()] if parent.exists() else []

    if len(candidates) != 1:
        problems.append(
            f"{model}/{strategy}/{moment}/{seed}: "
            f"expected one run directory, found {len(candidates)}"
        )
        continue

    run = candidates[0]
    sessions = sorted((run / "sessions").glob("session-*"))
    epoch_rows = {}
    results = []

    if len(sessions) > 1:
        multi_session.append(f"{run.relative_to(ROOT)}: {len(sessions)} sessions")

    for session in sessions:
        result_path = session / "result.json"
        csv_path = session / "moment_epoch_metrics.csv"

        if not result_path.exists() or not csv_path.exists():
            problems.append(f"{run}: missing result or metric CSV in {session.name}")
            continue

        try:
            result = json.loads(result_path.read_text())
            results.append((int(result["epoch"]), result, session))

            with csv_path.open(newline="") as f:
                rows = list(csv.DictReader(f))

            for row in rows:
                epoch = int(row["epoch"])
                if epoch in epoch_rows:
                    problems.append(f"{run}: duplicate epoch {epoch}")
                else:
                    epoch_rows[epoch] = (row, session, csv_path)
        except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
            problems.append(f"{run}/{session.name}: {exc}")

    if not results or not epoch_rows:
        problems.append(f"{run}: no usable results")
        continue

    # Use the final completed session, not an intermediate resume result.
    final_epoch, final_result, final_session = max(results, key=lambda x: x[0])
    selected_epoch = int(final_result["best_epoch"])
    selected = epoch_rows.get(selected_epoch)

    if selected is None:
        problems.append(f"{run}: best epoch {selected_epoch} absent from CSVs")
        continue

    selected_row, selected_session, selected_csv = selected

    try:
        values = {
            metric: float(selected_row[metric])
            for metric in METRICS
        }
        all_joint = [
            (float(row["val_joint_accuracy"]), epoch)
            for epoch, (row, _, _) in epoch_rows.items()
        ]
        global_max = max(v for v, _ in all_joint)
        earliest_max_epoch = min(
            epoch for value, epoch in all_joint
            if abs(value - global_max) < 1e-10
        )

        if len(epoch_rows) != 80 or set(epoch_rows) != set(range(1, 81)):
            problems.append(
                f"{run}: expected epochs 1..80, got {len(epoch_rows)} unique epochs"
            )

        if final_epoch != 80:
            problems.append(f"{run}: final session ends at epoch {final_epoch}")

        if abs(values["val_joint_accuracy"] - float(final_result["best_metric"])) > 1e-8:
            problems.append(f"{run}: best_metric disagrees with selected CSV row")

        if selected_epoch != earliest_max_epoch:
            problems.append(
                f"{run}: selected epoch {selected_epoch}, "
                f"earliest observed maximum at {earliest_max_epoch}"
            )

        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values.values()):
            problems.append(f"{run}: invalid accuracy range")

    except (KeyError, ValueError) as exc:
        problems.append(f"{run}: metric validation failed: {exc}")
        continue

    records.append({
        "model": model,
        "strategy": strategy,
        "moments": moment,
        "seed": seed,
        "best_epoch": selected_epoch,
        "final_epoch": final_epoch,
        "sessions": len(sessions),
        "alpha_accuracy_pct": 100 * values["val_alpha_accuracy"],
        "q0_accuracy_pct": 100 * values["val_q0_accuracy"],
        "joint_accuracy_pct": 100 * values["val_joint_accuracy"],
        "run_path": str(run),
        "selected_metric_csv": str(selected_csv),
        "final_result_json": str(final_session / "result.json"),
        "selected_epoch_session": selected_session.name,
    })

expected = set(product(MODELS, STRATEGIES, MOMENTS, SEEDS))
observed = {
    (r["model"], r["strategy"], r["moments"], r["seed"])
    for r in records
}
if observed != expected or len(records) != 120:
    problems.append(
        f"Incomplete inventory: {len(records)}/120; "
        f"missing={sorted(expected - observed)}"
    )

grouped = defaultdict(list)
for r in records:
    grouped[(r["model"], r["strategy"], r["moments"])].append(r)

summary = []
for (model, strategy, moment), runs in sorted(grouped.items()):
    if sorted(r["seed"] for r in runs) != sorted(SEEDS):
        problems.append(f"{model}/{strategy}/{moment}: incomplete seeds")
        continue

    row = {
        "model": model,
        "strategy": strategy,
        "moments": moment,
        "n_seeds": len(runs),
    }
    for metric in ("alpha", "q0", "joint"):
        vals = [r[f"{metric}_accuracy_pct"] for r in runs]
        row[f"{metric}_mean_pct"] = statistics.mean(vals)
        row[f"{metric}_sd_pct"] = statistics.stdev(vals)
    summary.append(row)

# Compare against rounded manuscript values.
for key, mean_ref, sd_ref in [
    (("clip", "S4", "E3"), 19.40, 1.34),
    (("siglip", "S4", "E4"), 17.50, 1.73),
]:
    matches = [
        r for r in summary
        if (r["model"], r["strategy"], r["moments"]) == key
    ]
    if len(matches) != 1:
        problems.append(f"Missing manuscript reference: {key}")
        continue
    r = matches[0]
    if abs(r["joint_mean_pct"] - mean_ref) > 0.015 or \
       abs(r["joint_sd_pct"] - sd_ref) > 0.015:
        problems.append(f"Manuscript value mismatch: {key}: {r}")

audit = [
    "# T04-06 manuscript extraction audit",
    f"- Production root: `{ROOT}`",
    f"- Recovered distinct runs: {len(records)}",
    f"- Recovered configurations: {len(summary)}",
    f"- Multi-session runs: {len(multi_session)}",
    "- Selection: final result.json best_epoch; metrics at that epoch",
    "- Accuracies converted from fractions to percentages",
    "- Variability: sample standard deviation across three seeds",
    "- No training, checkpoint loading, or inference was performed",
    "",
    "## Multi-session runs",
    *[f"- {entry}" for entry in multi_session],
    "",
    "## Validation problems",
    *([f"- {p}" for p in problems] if problems else ["- None"]),
]
(OUT / "extraction_audit.md").write_text("\n".join(audit) + "\n")

if problems:
    print("\n".join(audit))
    raise SystemExit("FAILED audit: outputs withheld; inspect extraction_audit.md")

for filename, data in [
    ("t04_06_all_120_runs.csv", records),
    ("t04_06_summary_40_configs.csv", summary),
]:
    path = OUT / filename
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    print("Wrote:", path)

print("\nAUDIT PASS")
print(f"Runs: {len(records)}")
print(f"Configurations: {len(summary)}")
print(f"Multi-session runs: {len(multi_session)}")
print("Manuscript reference checks: PASS")
