#!/usr/bin/env python3
"""
EpiSurge - Oracle composition vectors per (hhs_region, epiweek).

ORACLE = all sequences indexed by COLLECTION week, ignoring publication
delay. This is what retrospective studies implicitly assume is available.
It is the upper-bound comparison against which as-of composition
(build_asof_composition.py) is measured. That gap is contribution N5.

Oracle is NOT a usable forecasting feature - it leaks future information.
It exists solely as the ceiling in the N5 comparison.

Also emits the collection -> submission delay distribution, a Phase 1
deliverable per the Master Plan.

Outputs:
  data/features/composition_oracle.csv
  data/reports/delay_distribution.csv
"""

import csv
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from epiweeks import Week

PROCESSED = Path("data/processed")
FEATURES = Path("data/features")
REPORTS = Path("data/reports")
FEATURES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from config import SEASON_LABELS as SEASONS, ORIGINAL_SEASONS


SUBTYPE_PATTERN = re.compile(r"\((H\d+N\d+)\)")


def extract_subtype(desc):
    m = SUBTYPE_PATTERN.search(desc)
    if not m:
        return "unknown"
    s = m.group(1).upper()
    return s if s in ("H1N1", "H3N2") else "other"


def parse_submitted(s):
    try:
        return datetime.strptime(s.strip(), "%d-%b-%Y").date()
    except (ValueError, AttributeError):
        return None


def main():
    with open(PROCESSED / "submission_dates.json") as f:
        sub_dates = json.load(f)

    records = []
    subtype_counts = defaultdict(int)
    no_submission = 0
    # Publication windows overlap by 24 months (see config.FETCH_TAIL_MONTHS),
    # so the same accession can appear in two raw season files and would
    # otherwise be counted twice in the composition. First season wins.
    seen = set()
    duplicates = 0

    for season in SEASONS:
        with open(PROCESSED / f"mapped_{season}.json") as f:
            for r in json.load(f):
                if r["accession"] in seen:
                    duplicates += 1
                    continue
                seen.add(r["accession"])
                sub = parse_submitted(sub_dates.get(r["accession"], ""))
                if sub is None:
                    no_submission += 1
                    continue
                coll = datetime.fromisoformat(r["collection_date_parsed"]).date()
                w = Week.fromdate(coll)
                subtype = extract_subtype(r["description"])
                subtype_counts[subtype] += 1
                records.append({
                    "accession": r["accession"],
                    "season": season,
                    "region": r["hhs_region"],
                    "collection_date": coll,
                    "submission_date": sub,
                    "epiweek_int": w.year * 100 + w.week,
                    "epiweek": f"{w.year}-W{w.week:02d}",
                    "delay_days": (sub - coll).days,
                    "subtype": subtype,
                })

    print(f"Records: {len(records)}  (dropped, no submission date: {no_submission}; duplicate accessions across overlapping fetch windows: {duplicates})")
    print(f"Subtypes: {dict(subtype_counts)}")

    # ── delay distribution ──────────────────────────────────────
    delays = sorted(r["delay_days"] for r in records)
    negative = sum(1 for d in delays if d < 0)
    pct = lambda p: delays[int(len(delays) * p)]

    print(f"\nCollection -> submission delay (days):")
    print(f"  negative (data issue): {negative}")
    print(f"  p10={pct(0.10)}  p25={pct(0.25)}  median={pct(0.50)}  "
          f"p75={pct(0.75)}  p90={pct(0.90)}")
    print(f"  median weeks: {pct(0.50)/7:.1f}")
    if negative:
        print(f"  [WARN] {negative} records submitted before collection date")

    with open(REPORTS / "delay_distribution.csv", "w", newline="") as f:
        w_ = csv.writer(f)
        w_.writerow(["accession", "season", "hhs_region", "collection_date",
                     "submission_date", "delay_days"])
        for r in records:
            w_.writerow([r["accession"], r["season"], r["region"],
                         r["collection_date"], r["submission_date"],
                         r["delay_days"]])

    # ── oracle composition ──────────────────────────────────────
    grouped = defaultdict(lambda: defaultdict(int))
    for r in records:
        key = (r["region"], r["epiweek_int"], r["epiweek"], r["season"])
        grouped[key][r["subtype"]] += 1

    out = FEATURES / "composition_oracle.csv"
    with open(out, "w", newline="") as f:
        w_ = csv.writer(f)
        w_.writerow(["hhs_region", "epiweek_int", "epiweek", "season",
                     "n_h1n1", "n_h3n2", "n_other", "n_unknown", "n_total",
                     "prop_h1n1", "prop_h3n2", "prop_other"])
        for (region, ew_int, ew_str, season), c in sorted(grouped.items()):
            h1, h3 = c.get("H1N1", 0), c.get("H3N2", 0)
            oth, unk = c.get("other", 0), c.get("unknown", 0)
            denom = h1 + h3 + oth
            w_.writerow([region, ew_int, ew_str, season,
                         h1, h3, oth, unk, denom + unk,
                         round(h1 / denom, 4) if denom else "",
                         round(h3 / denom, 4) if denom else "",
                         round(oth / denom, 4) if denom else ""])

    print(f"\nOracle rows: {len(grouped)}  ({len(SEASONS)} seasons)")
    # 734 was the validated row count on the original three seasons. It is
    # not a defect for this to differ once seasons are added or the fetch
    # window changes - only a silent DROP would be. Report against the
    # three-season baseline only when the study is still three seasons.
    if set(SEASONS) == set(ORIGINAL_SEASONS) and len(grouped) != 734:
        print("  [WARN] three-season run but row count differs from the "
              "validated baseline of 734")
    print(f"Saved -> {out}")
    print(f"Saved -> {REPORTS / 'delay_distribution.csv'}")


if __name__ == "__main__":
    main()