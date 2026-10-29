#!/usr/bin/env python3
"""
EpiSurge - verify the corrected NCBI fetch window removed the censoring.

The old fetch searched publication dates over Oct-May while QC filtered
collection dates over the same months, so anything published after 31 May
was never retrieved. Two symptoms followed:

  A. April/May collections were near zero, in seasons whose evaluation
     window runs to mid-May.
  B. Maximum observed delay tracked days-remaining-to-window-close, month
     by month - the signature of right-censoring.

Stage 1 tests symptom A and needs only the raw fetch.
Stage 2 tests symptom B and needs data/processed/submission_dates.json,
so it is skipped until fetch_submission_dates.py has run.

Run (from the repository root):
    python src/analysis/verify_fetch_window.py
"""

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import SEASON_LABELS, fetch_window, verify_no_censoring   # noqa: E402

RAW = Path("data/raw/ncbi")
OLD = Path("data/raw/ncbi_OLD_censored")
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


def stage1():
    print("-" * 66)
    print("STAGE 1  Record counts and collection-month coverage (raw fetch)")
    print("-" * 66)

    rows, month_tab = [], {}
    for label in SEASON_LABELS:
        p = RAW / f"ncbi_{label}.json"
        if not p.exists():
            print(f"  [missing] {p}")
            continue
        recs = json.loads(p.read_text())

        months = Counter()
        parsed = 0
        for r in recs:
            d = parse_date(r.get("collection_date", ""))
            if d:
                parsed += 1
                months[d.strftime("%m")] += 1
        month_tab[label] = months

        old_n = None
        op = OLD / f"ncbi_{label}.json"
        if op.exists():
            old_n = len(json.loads(op.read_text()))

        rows.append({
            "season": label, "records": len(recs),
            "parseable_dates": parsed,
            "old_records": old_n,
            "ratio_vs_old": round(len(recs) / old_n, 2) if old_n else None,
            "apr": months.get("04", 0), "may": months.get("05", 0),
        })

    t = pd.DataFrame(rows)
    print(t.to_string(index=False))

    print("\n  Collection month distribution (all months, raw records):")
    md = pd.DataFrame(month_tab).fillna(0).astype(int).sort_index()
    print(md.to_string())

    print("\n  Symptom A - April/May collections present?")
    bad = []
    for _, r in t.iterrows():
        late = r["apr"] + r["may"]
        verdict = "OK" if late >= 20 else "STILL SPARSE"
        if late < 20:
            bad.append(r["season"])
        print(f"    {r['season']}: Apr {r['apr']:>5}  May {r['may']:>5}   {verdict}")
    return bad


def stage2():
    print("\n" + "-" * 66)
    print("STAGE 2  Censoring signature in the delay distribution")
    print("-" * 66)

    sub_p = PROCESSED / "submission_dates.json"
    if not sub_p.exists():
        print("  Skipped - run fetch_submission_dates.py first, then re-run this.")
        return None

    sub = json.loads(sub_p.read_text())
    rows = []
    for label in SEASON_LABELS:
        mp = PROCESSED / f"mapped_{label}.json"
        if not mp.exists():
            continue
        for r in json.loads(mp.read_text()):
            s = sub.get(r["accession"], "")
            sd = parse_date(s)
            cd = parse_date(r.get("collection_date_parsed", "")[:10]) \
                or parse_date(r.get("collection_date", ""))
            if sd and cd:
                rows.append({"season": label, "collection_date": cd,
                             "submission_date": sd, "delay_days": (sd - cd).days})
    if not rows:
        print("  No records with both dates - nothing to test.")
        return None

    d = pd.DataFrame(rows)
    print(f"  n = {len(d)}   delay range {d.delay_days.min()} .. {d.delay_days.max()}")
    print(f"  median delay = {d.delay_days.median():.0f} days "
          f"({d.delay_days.median()/7:.1f} weeks)")
    print("\n  Median delay by season:")
    print(d.groupby("season").delay_days.agg(["size", "median", "max"]).to_string())

    ok, rep = verify_no_censoring(d)
    print("\n  Max delay as % of the window bound (censoring if near 100):")
    print(rep.to_string(index=False))

    REPORTS.mkdir(parents=True, exist_ok=True)
    rep.to_csv(REPORTS / "fetch_window_verification.csv", index=False)
    d.to_csv(REPORTS / "delay_distribution.csv", index=False)

    print(f"\n  Symptom B: {'CLEARED' if ok else 'STILL CENSORED'}")
    if not ok:
        print("  Raise FETCH_TAIL_MONTHS in src/config.py and refetch the")
        print("  affected seasons with --force.")
    return ok


def main():
    print("EpiSurge - fetch window verification")
    print("=" * 66)
    bad = stage1()
    ok2 = stage2()

    print("\n" + "=" * 66)
    if bad:
        print(f"  Symptom A unresolved for: {', '.join(bad)}")
    else:
        print("  Symptom A cleared - late-season collections are present.")
    if ok2 is None:
        print("  Symptom B not yet testable - run fetch_submission_dates.py.")
    elif ok2:
        print("  Symptom B cleared - delay distribution is not truncated.")
    else:
        print("  Symptom B UNRESOLVED - widen the window and refetch.")


if __name__ == "__main__":
    main()