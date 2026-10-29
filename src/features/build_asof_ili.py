#!/usr/bin/env python3
"""
EpiSurge - As-of ILI lag features.

At forecast origin T, a forecaster sees issue T. Under the F8 decision we
use lag >= 1 only, so the available history is epiweeks T-1 ... T-8 as
PUBLISHED AT ISSUE T - not the later revised values.

Implementation note: we filter on Delphi's own `lag` column rather than
computing T-k ourselves. Epiweek arithmetic across year boundaries is a
classic off-by-one source (202301 minus 1 is 202252 or 202253 depending
on the year), and Delphi has already done it correctly.

Also emits the final revised values for the same (region, epiweek) pairs,
so the backfill correction can be quantified per cell.

Output: data/features/asof_ili.csv
"""

import sys
from pathlib import Path

import pandas as pd
from epiweeks import Week

RAW = Path("data/raw/fluview")
FEATURES = Path("data/features")
FEATURES.mkdir(parents=True, exist_ok=True)

OUT = FEATURES / "asof_ili.csv"

SEASON_WINDOWS = [(201927, 202026), (202227, 202326), (202327, 202426)]
MAX_LAG = 8          # Master Plan: surveillance block = ILI at lags 1-8
MIN_LAG = 1          # F8 decision


def origin_weeks():
    out = []
    for start, end in SEASON_WINDOWS:
        w = Week(start // 100, start % 100)
        while w.year * 100 + w.week <= end:
            out.append(w.year * 100 + w.week)
            w = w + 1
    return out


def main():
    print("Building as-of ILI lag features\n")

    v = pd.read_csv(RAW / "fluview_vintages.csv")
    v["hhs_region"] = v["region"].str.replace("hhs", "", regex=False).astype(int)
    print(f"  Vintage rows: {len(v)}")

    origins = origin_weeks()
    print(f"  Forecast origins: {len(origins)}")

    # keep only the lags we need, from issues that are forecast origins
    usable = v[(v["lag"] >= MIN_LAG) &
               (v["lag"] <= MAX_LAG) &
               (v["issue"].isin(origins))].copy()
    print(f"  Usable rows (lag {MIN_LAG}-{MAX_LAG}, issue in origins): {len(usable)}")

    # one row per (region, origin), one column per lag
    wide = usable.pivot_table(
        index=["hhs_region", "issue"],
        columns="lag",
        values="wili",
        aggfunc="first",
    )
    wide.columns = [f"ili_asof_lag{int(c)}" for c in wide.columns]
    wide = wide.reset_index().rename(columns={"issue": "epiweek"})

    expected_rows = len(origins) * 10
    print(f"\n  Origin-region rows: {len(wide)} (expected {expected_rows})")
    if len(wide) != expected_rows:
        print(f"  [WARN] missing {expected_rows - len(wide)} origin-region pairs")

    lag_cols = [f"ili_asof_lag{k}" for k in range(MIN_LAG, MAX_LAG + 1)]
    missing_cols = [c for c in lag_cols if c not in wide.columns]
    if missing_cols:
        sys.exit(f"Missing lag columns entirely: {missing_cols}")

    print(f"\n  Completeness per lag:")
    for c in lag_cols:
        n_missing = wide[c].isna().sum()
        print(f"    {c:<18} missing {n_missing:>4} / {len(wide)}")

    # quantify the backfill correction at lag 1, the most revised point
    final = pd.read_csv(RAW / "fluview.csv")
    final["hhs_region"] = final["region"].str.replace("hhs", "", regex=False).astype(int)
    final = final[["hhs_region", "epiweek", "wili"]].rename(columns={"wili": "wili_final"})

    lag1 = usable[usable["lag"] == 1][["hhs_region", "epiweek", "issue", "wili"]]
    lag1 = lag1.rename(columns={"wili": "wili_asof"})
    cmp = lag1.merge(final, on=["hhs_region", "epiweek"], how="inner")
    cmp["delta"] = (cmp["wili_final"] - cmp["wili_asof"]).abs()

    print(f"\n  Backfill at lag 1 ({len(cmp)} cells):")
    print(f"    mean |revision|:   {cmp['delta'].mean():.4f} pct-points")
    print(f"    median:            {cmp['delta'].median():.4f}")
    print(f"    p90:               {cmp['delta'].quantile(0.90):.4f}")
    print(f"    max:               {cmp['delta'].max():.4f}")
    print(f"    revised by >0.1:   {(cmp['delta'] > 0.1).mean()*100:.1f}%")

    wide = wide.sort_values(["hhs_region", "epiweek"]).reset_index(drop=True)
    wide.to_csv(OUT, index=False)
    print(f"\n  Saved -> {OUT}")
    print(f"  Shape: {wide.shape[0]} rows x {wide.shape[1]} columns")


if __name__ == "__main__":
    main()