#!/usr/bin/env python3
"""
EpiSurge - Phase 1, pipeline step 10: master region-week table.

One row per (hhs_region, epiweek). FluView is the base panel.

Holds JOINED RAW VALUES ONLY - no lags beyond the as-of ILI block already
computed, no derived features. Feature engineering happens downstream so
feature choices can change without re-running the join.

Columns by block:
  target      wili_final          final revised wILI (FluSight convention)
  surveillance ili_asof_lag1..8   as published at issue T, lag >= 1 (F8)
  genomic     oracle_*            N5 ceiling - LEAKS, not a usable feature
              asof_L4_12_*        usable genomic features
              asof_L8_16_*
  climate     temp_c, abs_humidity
  static      population
  flags       in_core_season, has_genomic_*

Season windows run MMWR week 27 -> week 26 so lead-in weeks exist.
in_core_season marks weeks 40-20, the evaluation window.

Output: data/processed/master_region_week.csv
"""

import sys
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
FEATURES = Path("data/features")

OUT = PROCESSED / "master_region_week.csv"

import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from config import SEASONS as _CFG_SEASONS


SEASONS = [(lbl, a, b) for lbl, (_, _, a, b) in _CFG_SEASONS.items()]

ASOF_WINDOWS = ["L4-12", "L8-16"]
LAG_COLS = [f"ili_asof_lag{k}" for k in range(1, 9)]


def assign_season(ew):
    for label, start, end in SEASONS:
        if start <= ew <= end:
            return label
    return None


def is_core_season(ew):
    week = ew % 100
    return week >= 40 or week <= 20


def load_fluview_final():
    df = pd.read_csv(RAW / "fluview" / "fluview.csv")
    df["hhs_region"] = df["region"].str.replace("hhs", "", regex=False).astype(int)
    df["epiweek"] = df["epiweek"].astype(int)
    df = df[["hhs_region", "epiweek", "wili", "num_patients", "num_providers"]]
    return df.rename(columns={"wili": "wili_final"})


def load_asof_ili():
    df = pd.read_csv(FEATURES / "asof_ili.csv")
    return df[["hhs_region", "epiweek"] + LAG_COLS]


def load_population():
    df = pd.read_csv(RAW / "census" / "hhs_region_population.csv")
    return df[["hhs_region", "population"]]


def load_climate():
    df = pd.read_csv(PROCESSED / "climate_region_week.csv")
    return df.rename(columns={"n_days": "climate_n_days"})


def load_oracle():
    df = pd.read_csv(FEATURES / "composition_oracle.csv")
    # file carries both epiweek_int (key) and epiweek (display string) -
    # drop the string before renaming, or the rename collides
    df = df.drop(columns=["epiweek"]).rename(columns={"epiweek_int": "epiweek"})
    cols = ["n_total", "prop_h1n1", "prop_h3n2", "prop_other"]
    df = df[["hhs_region", "epiweek"] + cols]
    return df.rename(columns={c: f"oracle_{c}" for c in cols})


def load_asof_composition(window):
    path = FEATURES / f"asof_composition_{window}.csv"
    if not path.exists():
        sys.exit(f"Missing {path} - run build_asof_composition.py first")
    df = pd.read_csv(path)
    df = df.rename(columns={"origin_epiweek_int": "epiweek"})
    # shrunk_* are the model features; prop_* retained for comparison;
    # n_total carries evidence strength so the model can weight it itself
    cols = ["n_total",
            "prop_h1n1", "prop_h3n2", "prop_other",
            "shrunk_h1n1", "shrunk_h3n2", "shrunk_other",
            "meets_report_threshold"]
    df = df[["hhs_region", "epiweek"] + cols]
    tag = window.replace("-", "_")
    return df.rename(columns={c: f"asof_{tag}_{c}" for c in cols})


def main():
    print("Building master region-week table\n")

    master = load_fluview_final()
    master["season"] = master["epiweek"].apply(assign_season)
    master = master.dropna(subset=["season"]).copy()
    master["in_core_season"] = master["epiweek"].apply(is_core_season)
    print(f"  FluView rows in season windows: {len(master)}")

    before = len(master)
    master = master.merge(load_asof_ili(), on=["hhs_region", "epiweek"], how="left")
    master = master.merge(load_population(), on=["hhs_region"], how="left")
    master = master.merge(load_climate(), on=["hhs_region", "epiweek"], how="left")
    master = master.merge(load_oracle(), on=["hhs_region", "epiweek"], how="left")
    for w in ASOF_WINDOWS:
        master = master.merge(load_asof_composition(w),
                              on=["hhs_region", "epiweek"], how="left")

    if len(master) != before:
        sys.exit(f"Join changed row count {before} -> {len(master)} "
                 f"- duplicate keys in a joined source")

    # coverage flags (H3): absent sequences are NOT zero signal
    master["has_genomic_oracle"] = master["oracle_n_total"].fillna(0) > 0
    for w in ASOF_WINDOWS:
        tag = w.replace("-", "_")
        # F11: coverage now means "any sequences present", not "passes a
        # count threshold" - low-count cells are shrunk, not discarded
        master[f"has_genomic_asof_{tag}"] = master[f"asof_{tag}_n_total"].fillna(0) > 0
    master["has_surveillance"] = master[LAG_COLS].notna().all(axis=1)

    master = master.sort_values(["hhs_region", "epiweek"]).reset_index(drop=True)

    # ── validation ──────────────────────────────────────────────
    failures = []

    print("\n  Rows per season:")
    for label, grp in master.groupby("season"):
        print(f"    {label}: {len(grp):>4} rows ({grp['in_core_season'].sum()} core)")

    core = master[master["in_core_season"]]
    print(f"\n  Core-season rows: {len(core)}")

    for name, col in [("target wili_final", "wili_final"),
                      ("population", "population"),
                      ("climate abs_humidity", "abs_humidity")]:
        n = core[col].isna().sum()
        print(f"    missing {name:<22} {n}")
        if n:
            failures.append(f"{n} core rows missing {name}")

    n_surv = (~core["has_surveillance"]).sum()
    print(f"    missing surveillance lags  {n_surv}")
    if n_surv:
        failures.append(f"{n_surv} core rows missing surveillance lags")

    partial = (core["climate_n_days"] < 7).sum()
    print(f"    partial climate weeks      {partial}")

    print(f"\n  Genomic coverage (core season, n={len(core)}):")
    print(f"    oracle  : {core['has_genomic_oracle'].sum():>4} "
          f"({100*core['has_genomic_oracle'].mean():.1f}%)")
    for w in ASOF_WINDOWS:
        tag = w.replace("-", "_")
        col = f"has_genomic_asof_{tag}"
        print(f"    as-of {w:<5}: {core[col].sum():>4} "
              f"({100*core[col].mean():.1f}%)")

    total_oracle = master["has_genomic_oracle"].sum()
    # 734 was the validated count on the original three seasons; quoting it
    # against a seven-season run was misleading rather than informative.
    print(f"\n  Oracle cells, all rows: {total_oracle} "
          f"across {len(SEASONS)} seasons")

    # modelling-ready rows: everything except genomic present
    ready = core[core["has_surveillance"] &
                 core["wili_final"].notna() &
                 core["abs_humidity"].notna()]
    print(f"\n  Modelling-ready core rows: {len(ready)} / {len(core)}")
    for w in ASOF_WINDOWS:
        tag = w.replace("-", "_")
        n = ready[f"has_genomic_asof_{tag}"].sum()
        print(f"    with as-of genomic {w:<5}: {n} ({100*n/len(ready):.1f}%)")

    master.to_csv(OUT, index=False)
    print(f"\n  Saved -> {OUT}")
    print(f"  Shape: {master.shape[0]} rows x {master.shape[1]} columns")

    if failures:
        print("\n  FAILURES:")
        for f in failures:
            print(f"    [FAIL] {f}")
        sys.exit(1)
    print("\n  All checks passed.")


if __name__ == "__main__":
    main()