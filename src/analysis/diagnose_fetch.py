#!/usr/bin/env python3
"""
EpiSurge - two diagnostics after the corrected fetch.

1. CENSORING, tested properly.
   verify_fetch_window.py flagged "STILL CENSORED" using a crude rule:
   max observed delay within 85% of the window bound. That rule cannot
   distinguish real truncation from a long tail that occasionally touches
   the bound, and any finite window produces the latter.

   The signature that actually matters is systematic: under truncation,
   samples collected later in the season have less time to be published,
   so MEDIAN delay declines monotonically with collection month. Under the
   old window that was unmistakable (2023-24 medians: 113, 84, 76, 72, 45,
   37, 41, 14 by month). A flat or noisy profile means the window is wide
   enough for inference, even if a few tail observations sit near the edge.

   Also reports tail mass beyond 1, 1.5 and 2 years, which is what a wider
   window would actually recover.

2. STATE PARSING.
   2015-16 mapped only 21.6% of QC-passed records against 89-97% in other
   seasons, with 2,799 NO_STATE_PARSED. This prints the geo_loc_name
   strings that failed, by season, so the cause is visible rather than
   guessed at.

Run (from the repository root):
    python src/analysis/diagnose_fetch.py
"""

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "preprocessing"))

from config import SEASON_LABELS                         # noqa: E402
from map_region_epiweek import extract_state             # noqa: E402

PROCESSED = Path("data/processed")
REPORTS = Path("data/reports")


def parse_date(s):
    s = (s or "").strip()
    for fmt in ("%d-%b-%Y", "%b-%Y", "%Y", "%Y-%m-%d", "%Y-%m"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def censoring():
    print("-" * 70)
    print("1. CENSORING - does MEDIAN delay decline with collection month?")
    print("-" * 70)

    p = REPORTS / "delay_distribution.csv"
    if not p.exists():
        print(f"  missing {p} - run verify_fetch_window.py first")
        return
    d = pd.read_csv(p, parse_dates=["collection_date", "submission_date"])
    d["month"] = d["collection_date"].dt.month

    # season-relative month order: Oct..Dec then Jan..May
    order = [10, 11, 12, 1, 2, 3, 4, 5]
    print("\n  Median delay (days) by collection month, in season order:")
    tab = (d[d.month.isin(order)]
           .pivot_table(index="month", columns="season",
                        values="delay_days", aggfunc="median")
           .reindex(order))
    print(tab.round(0).to_string())

    print("\n  Spearman correlation between season-month order and median delay")
    print("  (strongly negative => still truncated; near zero => clean):")
    from scipy.stats import spearmanr
    flagged = []
    for season in tab.columns:
        col = tab[season].dropna()
        if len(col) < 4:
            print(f"    {season}: too few months")
            continue
        pos = [order.index(m) for m in col.index]
        rho, pv = spearmanr(pos, col.values)
        verdict = "TRUNCATED" if (rho < -0.75 and pv < 0.05) else "clean"
        if verdict == "TRUNCATED":
            flagged.append(season)
        print(f"    {season}: rho={rho:+.2f}  p={pv:.3f}   {verdict}")

    print("\n  Tail mass - what a wider window could still recover:")
    for cut, lab in [(365, "> 1 year"), (548, "> 1.5 years"), (730, "> 2 years")]:
        pct = 100 * (d.delay_days > cut).mean()
        print(f"    {lab:<12} {pct:5.2f}%  ({int((d.delay_days > cut).sum())} records)")

    print(f"\n  Overall median delay: {d.delay_days.median():.0f} days "
          f"({d.delay_days.median()/7:.1f} weeks)")
    print(f"  VERDICT: {'refetch needed for ' + ', '.join(flagged) if flagged else 'window is adequate; no refetch'}")


def parsing():
    print("\n" + "-" * 70)
    print("2. STATE PARSING - which geo_loc_name strings fail, and why")
    print("-" * 70)

    rows = []
    for label in SEASON_LABELS:
        p = PROCESSED / f"qc_{label}.json"
        if not p.exists():
            continue
        recs = json.loads(p.read_text())
        failed = Counter()
        n_ok = 0
        for r in recs:
            g = r.get("geo_loc_name", "")
            if extract_state(g):
                n_ok += 1
            else:
                failed[g.strip()] += 1
        rows.append({"season": label, "qc_records": len(recs),
                     "state_parsed": n_ok,
                     "pct": round(100 * n_ok / len(recs), 1) if recs else 0,
                     "distinct_failures": len(failed)})
        if failed:
            print(f"\n  {label}: {sum(failed.values())} failures, "
                  f"{len(failed)} distinct strings. Top 10:")
            for s, n in failed.most_common(10):
                print(f"      {n:>6}  {s!r}")

    print("\n  Summary:")
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    print("EpiSurge - post-fetch diagnostics")
    print("=" * 70)
    censoring()
    parsing()


if __name__ == "__main__":
    main()