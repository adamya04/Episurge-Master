#!/usr/bin/env python3
"""
EpiSurge - Phase 1: As-of (vintage-correct) composition features.

At forecast origin T, a forecaster can use any sequence with
submission_date <= T. Because median collection->submission delay is
~76 days, those sequences mostly describe collection weeks 9-16 back.

So the operationally honest genomic feature is:
    composition over collection weeks [T-lag_max, T-lag_min],
    restricted to sequences submitted on or before T.

Composition estimates are shrunk toward the pooled national composition
(empirical Bayes). A proportion from n sequences has standard error
sqrt(p(1-p)/n) - at n=5, p=0.5 that is +/-22pp. A hard validity cutoff is
therefore arbitrary wherever it is placed, and discarding low-count cells
both loses real signal and biases the retained sample toward high-
sequencing regions (which correlate with population and surveillance
capacity). Shrinkage handles that uncertainty continuously instead.

No cell is filtered out. Sequence count is carried as a feature so the
model can weight evidence strength itself.

Output: data/features/asof_composition_L{min}-{max}.csv
        data/reports/asof_coverage_summary.csv
"""

import json
import re
import csv
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

from epiweeks import Week

PROCESSED_DIR = Path("data/processed")
FEATURES_DIR = Path("data/features")
REPORTS_DIR = Path("data/reports")
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from config import SEASON_LABELS as SEASONS
SUBTYPE_PATTERN = re.compile(r'\((H\d+N\d+)\)')

# Retained windows only (see DECISIONS.md): L1-8 dropped at 12.4% coverage,
# L9-20 and L12-24 dominated by L8-16 on both coverage and freshness.
CANDIDATE_WINDOWS = [(4, 12), (8, 16)]

# Forecast-origin weeks are defined by the season calendar, not by where
# sequences happen to exist.
from config import SEASON_WINDOWS



# Shrinkage strength: weight given to the pooled (national) composition.
# k=5 means a cell with 5 sequences sits halfway between its own estimate
# and the national mean. Chosen to match the scale at which binomial
# sampling error becomes comparable to between-region variation.
SHRINKAGE_K = 5

# Retained only as a reporting threshold, not a filter - no cell is dropped.
MIN_SEQS_FOR_REPORT = 5


def extract_subtype(desc):
    m = SUBTYPE_PATTERN.search(desc)
    if not m:
        return "unknown"
    s = m.group(1).upper()
    return s if s in ("H1N1", "H3N2") else "other"


def parse_submitted(s):
    try:
        return datetime.strptime(s.strip(), "%d-%b-%Y").date()
    except ValueError:
        return None


def epiweek_start(y, w):
    """Sunday-start date of MMWR week (y, w)."""
    return Week(y, w).startdate()


def load_records():
    with open(PROCESSED_DIR / "submission_dates.json") as f:
        sub_dates = json.load(f)

    out = []
    # Publication windows overlap by 24 months, so the same accession can
    # appear in two raw season files. Deduplicate or it is counted twice.
    seen = set()
    duplicates = 0
    for season in SEASONS:
        with open(PROCESSED_DIR / f"mapped_{season}.json") as f:
            for r in json.load(f):
                if r["accession"] in seen:
                    duplicates += 1
                    continue
                seen.add(r["accession"])
                sub = parse_submitted(sub_dates.get(r["accession"], ""))
                if sub is None:
                    continue
                coll = datetime.fromisoformat(r["collection_date_parsed"]).date()
                out.append({
                    "region": r["hhs_region"],
                    "season": season,
                    "collection_date": coll,
                    "submission_date": sub,
                    "subtype": extract_subtype(r["description"]),
                })
    if duplicates:
        print(f"  deduplicated {duplicates} accessions seen in more than one season file")
    return out


def build_origin_grid(records):
    """
    Every forecast-origin week in the season windows, whether or not any
    sequence was collected that week. An origin with no collections that
    week can still have published sequences in its trailing window.
    """
    regions = sorted({r["region"] for r in records})
    weeks = []
    for start, end in SEASON_WINDOWS:
        w = Week(start // 100, start % 100)
        while w.year * 100 + w.week <= end:
            weeks.append((w.year, w.week))
            w = w + 1
    return regions, weeks


def main():
    records = load_records()
    print(f"Loaded {len(records)} records")

    regions, all_weeks = build_origin_grid(records)
    print(f"Regions: {len(regions)}   Candidate origin weeks: {len(all_weeks)}")

    coverage_rows = []

    for lag_min, lag_max in CANDIDATE_WINDOWS:
        grouped = defaultdict(lambda: defaultdict(int))

        for region in regions:
            for (oy, ow) in all_weeks:
                origin_date = epiweek_start(oy, ow)
                win_start = origin_date - timedelta(weeks=lag_max)
                win_end = origin_date - timedelta(weeks=lag_min)

                for r in records:
                    if r["region"] != region:
                        continue
                    if not (win_start <= r["collection_date"] <= win_end):
                        continue
                    if r["submission_date"] > origin_date:   # not yet public
                        continue
                    grouped[(region, oy, ow)][r["subtype"]] += 1

        # pooled (national) composition per origin week, for shrinkage
        pooled = {}
        for (region, oy, ow), c in grouped.items():
            key = (oy, ow)
            agg = pooled.setdefault(key, {"H1N1": 0, "H3N2": 0, "other": 0})
            for k_ in agg:
                agg[k_] += c.get(k_, 0)

        above = sum(1 for c in grouped.values()
                    if sum(c.values()) >= MIN_SEQS_FOR_REPORT)
        total_cells = len(regions) * len(all_weeks)
        total_seqs = sum(sum(c.values()) for c in grouped.values())

        print(f"  window L{lag_min}-{lag_max:<2}w : "
              f"populated={len(grouped):>4}  "
              f">={MIN_SEQS_FOR_REPORT} seqs={above:>4} "
              f"({100*above/total_cells:.1f}% of grid)  seqs={total_seqs}")

        coverage_rows.append({
            "lag_min": lag_min, "lag_max": lag_max,
            "populated_cells": len(grouped), "cells_above_threshold": above,
            "grid_cells": total_cells,
            "pct_above_threshold": round(100 * above / total_cells, 2),
            "total_sequences": total_seqs,
        })

        out = FEATURES_DIR / f"asof_composition_L{lag_min}-{lag_max}.csv"
        write_composition(out, grouped, pooled)

    with open(REPORTS_DIR / "asof_coverage_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(coverage_rows[0].keys()))
        w.writeheader()
        w.writerows(coverage_rows)

    print(f"\nCoverage summary -> data\\reports\\asof_coverage_summary.csv")
    print("Note: no cells are filtered. Threshold is reporting only;")
    print("low-count cells are shrunk toward the national composition.")


def write_composition(path, grouped, pooled):
    """
    Writes raw counts, raw proportions, and shrunk proportions.

    Shrinkage: p_shrunk = (n*p_obs + k*p_pooled) / (n + k)
    Low-count cells are pulled toward the national composition for that
    week rather than being trusted at face value or discarded. No cell is
    filtered out - sequence count is carried as a feature so the model can
    weight evidence strength itself.
    """
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hhs_region", "origin_year", "origin_week", "origin_epiweek_int",
                    "n_h1n1", "n_h3n2", "n_other", "n_unknown", "n_total",
                    "prop_h1n1", "prop_h3n2", "prop_other",
                    "shrunk_h1n1", "shrunk_h3n2", "shrunk_other",
                    "meets_report_threshold"])

        for (region, oy, ow), c in sorted(grouped.items()):
            h1, h3 = c.get("H1N1", 0), c.get("H3N2", 0)
            oth, unk = c.get("other", 0), c.get("unknown", 0)
            n = h1 + h3 + oth
            tot = n + unk

            pool = pooled.get((oy, ow), {"H1N1": 0, "H3N2": 0, "other": 0})
            pool_n = sum(pool.values())
            if pool_n > 0:
                pp = [pool["H1N1"] / pool_n, pool["H3N2"] / pool_n, pool["other"] / pool_n]
            else:
                pp = [1 / 3, 1 / 3, 1 / 3]     # uninformative prior

            if n > 0:
                obs = [h1 / n, h3 / n, oth / n]
                raw = [round(x, 4) for x in obs]
            else:
                obs = pp
                raw = ["", "", ""]

            denom = n + SHRINKAGE_K
            shrunk = [round((n * o + SHRINKAGE_K * p) / denom, 4)
                      for o, p in zip(obs, pp)]

            w.writerow([region, oy, ow, oy * 100 + ow,
                        h1, h3, oth, unk, tot,
                        raw[0], raw[1], raw[2],
                        shrunk[0], shrunk[1], shrunk[2],
                        int(tot >= MIN_SEQS_FOR_REPORT)])


if __name__ == "__main__":
    main()