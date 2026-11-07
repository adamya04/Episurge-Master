#!/usr/bin/env python3
"""
EpiSurge - ordered pipeline driver.

The correct execution order has lived only in notes, and getting it wrong
fails in ways that are quiet rather than loud: a stage run before its input
exists reads a stale file from the previous run and reports plausible
numbers from the wrong data.

Each stage declares the files it needs and the files it produces. A stage
whose inputs are missing is skipped with a reason rather than left to fail
mid-way through.

Run (from the repository root, inside the venv):

    python src/tools/run_pipeline.py --list
    python src/tools/run_pipeline.py --from qc_filter
    python src/tools/run_pipeline.py --only analysis
    python src/tools/run_pipeline.py --dry-run

Ingestion stages are NOT run by default: they need credentials, take hours,
and are resumable on their own. Use --include-ingestion to add them.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# (name, script, group, [required inputs], [outputs])
# Paths are relative to the repository root.
STAGES = [
    ("fetch_ncbi", "src/ingestion/fetch_ncbi.py", "ingestion",
     [], ["data/raw/ncbi"]),
    ("fetch_fluview", "src/ingestion/fetch_fluview.py", "ingestion",
     [], ["data/raw/fluview/fluview.csv"]),
    ("fetch_fluview_vintages", "src/ingestion/fetch_fluview_vintages.py", "ingestion",
     ["data/raw/fluview/fluview.csv"], ["data/raw/fluview/fluview_vintages.csv"]),
    ("fetch_census", "src/ingestion/fetch_census.py", "ingestion",
     [], ["data/raw/census"]),
    ("fetch_gadm", "src/ingestion/fetch_gadm.py", "ingestion",
     [], ["data/raw/gadm"]),
    ("fetch_era5", "src/ingestion/fetch_era5.py", "ingestion",
     [], ["data/raw/era5"]),

    ("qc_filter", "src/preprocessing/qc_filter.py", "preprocessing",
     ["data/raw/ncbi"], ["data/processed/qc_summary.json"]),
    ("map_region_epiweek", "src/preprocessing/map_region_epiweek.py", "preprocessing",
     ["data/processed/qc_summary.json"], ["data/processed/mapping_summary.json"]),
    ("fetch_submission_dates", "src/ingestion/fetch_submission_dates.py", "preprocessing",
     ["data/processed/mapping_summary.json"], ["data/processed/submission_dates.json"]),
    ("unzip_era5", "src/preprocessing/unzip_era5.py", "preprocessing",
     ["data/raw/era5"], ["data/raw/era5/extracted"]),
    ("era5_to_region_week", "src/preprocessing/era5_to_region_week.py", "preprocessing",
     ["data/raw/era5/extracted", "data/raw/gadm", "data/raw/census"],
     ["data/processed/climate_region_week.csv"]),

    ("build_oracle_composition", "src/features/build_oracle_composition.py", "features",
     ["data/processed/submission_dates.json"], ["data/features/composition_oracle.csv"]),
    ("build_asof_composition", "src/features/build_asof_composition.py", "features",
     ["data/processed/submission_dates.json"],
     ["data/features/asof_composition_L8-16.csv"]),
    ("build_asof_ili", "src/features/build_asof_ili.py", "features",
     ["data/raw/fluview/fluview_vintages.csv"], ["data/features/asof_ili.csv"]),

    ("build_master_table", "src/preprocessing/build_master_table.py", "join",
     ["data/features/composition_oracle.csv",
      "data/features/asof_composition_L8-16.csv",
      "data/features/asof_ili.csv",
      "data/processed/climate_region_week.csv"],
     ["data/processed/master_region_week.csv"]),
    ("validate_all_sources", "src/preprocessing/validate_all_sources.py", "join",
     ["data/processed/master_region_week.csv"], []),

    ("E1_delay_coverage", "src/analysis/delay_coverage_analysis.py", "analysis",
     ["data/processed/master_region_week.csv"],
     ["data/reports/coverage_by_phase.csv"]),
    ("E2_baselines", "src/analysis/baselines.py", "analysis",
     ["data/processed/master_region_week.csv"],
     ["data/reports/baseline_results.csv"]),
]

GROUPS = ["ingestion", "preprocessing", "features", "join", "analysis"]


def exists(rel):
    return (ROOT / rel).exists()


def run(stage, dry):
    name, script, _, inputs, outputs = stage
    missing = [p for p in inputs if not exists(p)]
    if missing:
        print(f"  [SKIP] {name}: missing input(s) {missing}")
        return "skipped"
    if dry:
        print(f"  [DRY ] {name}: would run {script}")
        return "dry"

    print(f"\n{'=' * 70}\n  {name}  ({script})\n{'=' * 70}")
    t0 = time.time()
    r = subprocess.run([sys.executable, script], cwd=ROOT)
    dt = time.time() - t0
    if r.returncode != 0:
        print(f"\n  [FAIL] {name} exited {r.returncode} after {dt:.0f}s")
        return "failed"
    produced = [p for p in outputs if exists(p)]
    print(f"\n  [ OK ] {name} in {dt:.0f}s"
          + (f"  -> {', '.join(produced)}" if produced else ""))
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="print stages and exit")
    ap.add_argument("--dry-run", action="store_true", help="show what would run")
    ap.add_argument("--from", dest="start", help="begin at this stage name")
    ap.add_argument("--only", help=f"run one group: {', '.join(GROUPS)}")
    ap.add_argument("--include-ingestion", action="store_true",
                    help="include ingestion stages (slow, needs credentials)")
    args = ap.parse_args()

    stages = list(STAGES)
    if args.only:
        if args.only not in GROUPS:
            sys.exit(f"--only must be one of {GROUPS}")
        stages = [s for s in stages if s[2] == args.only]
    elif not args.include_ingestion:
        stages = [s for s in stages if s[2] != "ingestion"]

    if args.start:
        names = [s[0] for s in stages]
        if args.start not in names:
            sys.exit(f"Unknown stage {args.start!r}. Known: {names}")
        stages = stages[names.index(args.start):]

    if args.list:
        print("EpiSurge pipeline stages\n")
        for name, script, group, inputs, outputs in STAGES:
            ready = "ready" if all(exists(p) for p in inputs) else "blocked"
            print(f"  [{group:<13}] {name:<26} {ready:>7}   {script}")
        print("\nIngestion is excluded by default (slow, needs credentials).")
        return

    print("EpiSurge pipeline")
    print(f"  root:   {ROOT}")
    print(f"  stages: {len(stages)}")
    if not args.include_ingestion and not args.only:
        print("  note:   ingestion excluded (use --include-ingestion)")

    results = {}
    for stage in stages:
        results[stage[0]] = run(stage, args.dry_run)
        if results[stage[0]] == "failed":
            print("\nStopping: a stage failed. Fix it before continuing - "
                  "later stages would read stale files and report "
                  "plausible numbers from the wrong data.")
            sys.exit(1)

    print(f"\n{'=' * 70}\nSummary")
    for k, v in results.items():
        print(f"  {v:<8} {k}")
    if any(v == "skipped" for v in results.values()):
        print("\nSkipped stages were missing inputs. Run the ingestion stages "
              "or an earlier group first.")


if __name__ == "__main__":
    main()