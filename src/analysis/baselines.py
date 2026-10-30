#!/usr/bin/env python3
"""
EpiSurge - Phase 2 / E2: baseline forecasts B0-B2 across six horizons.

Establishes the bar every later model must clear. Nothing here uses genomic
data; that comparison is E3. The point of this script is to find out how
much of regional wILI is already explained by case history and seasonality
alone, horizon by horizon, so that any genomic gain later is measured
against an honest reference rather than a weak one.

Why this matters more than it looks. Measured on the master table, wILI
autocorrelation decays from 0.93 at one week to 0.12 at six weeks and
-0.17 at eight. At long horizons persistence is not merely weak, it is
anti-correlated: the value eight weeks after a peak tends to move opposite
to the value at the peak. So "beats persistence at h=8" is close to
meaningless. The climatological baseline is the one that matters out there,
and it is reported separately for exactly that reason.

Models
    B0  persistence      forecast = last PUBLISHED wILI at origin T, i.e.
                         ili_asof_lag1. Not the final revised value - a
                         forecaster at T does not have it.
    B1  climatological   mean wILI for this (region, MMWR week) across
                         strictly prior seasons in the training set. With
                         three seasons this rests on one or two observations
                         per cell; n_seasons is reported with every result.
    B2  gradient boosting  on as-of lags 1-8 + region + position in season.

Splits - rolling origin, two folds (DECISIONS.md 2026-10-29):
    fold 1   train 2019-20              -> test 2022-23
    fold 2   train 2019-20 + 2022-23    -> test 2023-24
This replaces Master Plan 3.4, whose single calibration block sat in the
early-season zero-genomic-coverage region while test sat in the late-season
full-coverage region.

Metrics are MAE and RMSE only. No WIS here: these are point forecasts, and
manufacturing intervals from training residuals would not be comparable to
the conformal intervals applied uniformly in Phase 5. WIS enters there.

All models are scored on an identical row set per (fold, horizon) so every
comparison is paired, as Master Plan 7.4 requires.

Inputs:
    data/processed/master_region_week.csv

Outputs:
    data/reports/baseline_results.csv        every model x fold x horizon x stratum
    data/reports/baseline_summary.csv        pooled across folds, headline table
    figures/fig_baseline_by_horizon.png
    figures/fig_baseline_skill_vs_persistence.png

Run (from the repository root, inside the venv):
    python src/analysis/baselines.py

Requires scikit-learn and matplotlib:
    pip install scikit-learn matplotlib
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor

import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from config import SEASON_LABELS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROCESSED = Path("data/processed")
REPORTS = Path("data/reports")
FIGURES = Path("figures")

MASTER = PROCESSED / "master_region_week.csv"

HORIZONS = [1, 2, 3, 4, 6, 8]
SEEDS = [0, 1, 2, 3, 4]              # Master Plan 7.4: 5 seeds, report mean +/- std
LAG_COLS = [f"ili_asof_lag{k}" for k in range(1, 9)]

# Rolling-origin folds, generated from config: expanding window, each test
# season strictly later than every training season in its fold. Seven
# seasons give six folds.
#
# E2 on the original three seasons produced two folds that reversed each
# other (climatological skill at h=8: -8.9% then +56.7%), because the
# seasons peak at MMWR weeks 06, 47 and 52 and a climatology built from
# one season lands on the wrong week in the next. Pooled-across-fold
# numbers are therefore NOT reported; per-fold results are the output.
FOLDS = [
    (f"fold{i}", SEASON_LABELS[:i], SEASON_LABELS[i])
    for i in range(1, len(SEASON_LABELS))
]

# B2 gets position-in-season alongside lags and region. Master Plan 7.1
# specifies "lags + region", but seasonality is the dominant structure in
# this target and withholding it would produce an artificially weak
# baseline - precisely the failure mode 9.3 warns about.
B2_FEATURES = LAG_COLS + ["hhs_region", "season_week_idx"]

# Implementation note: the plan names XGBoost. scikit-learn's
# HistGradientBoostingRegressor is the same algorithm family (histogram-
# binned gradient-boosted trees), is already a transitive dependency, and
# avoids an extra Windows build. Logged in DECISIONS.md.
B2_PARAMS = dict(max_iter=300, learning_rate=0.05, max_depth=3,
                 min_samples_leaf=20, l2_regularization=1.0,
                 early_stopping=False)

COLOURS = {"B0": "#888888", "B1": "#e8a33d", "B2": "#1f4e79"}
NAMES = {"B0": "B0 persistence", "B1": "B1 climatological",
         "B2": "B2 gradient boosting"}


# ──────────────────────────────────────────────────────────────────────

def load():
    if not MASTER.exists():
        sys.exit(f"Missing {MASTER} - run build_master_table.py first")
    df = pd.read_csv(MASTER)
    core = df[df["in_core_season"] == True].copy()        # noqa: E712

    core["week"] = core["epiweek"] % 100
    core["season_week_idx"] = (
        core.groupby("season")["epiweek"].rank(method="dense").astype(int) - 1
    )

    # peak-relative position, same definition as E1, for stratification
    peak_idx = {}
    for season, grp in core.groupby("season"):
        nat = grp.groupby("epiweek")["wili_final"].mean()
        pe = int(nat.idxmax())
        peak_idx[season] = int(grp.loc[grp["epiweek"] == pe, "season_week_idx"].iloc[0])
    core["weeks_from_peak"] = core["season_week_idx"] - core["season"].map(peak_idx)
    core["peak_side"] = np.where(core["weeks_from_peak"] < 0, "pre-peak", "peak or after")

    core["has_genomic"] = core["has_genomic_asof_L8_16"].astype(bool)
    return core.sort_values(["season", "hhs_region", "season_week_idx"]).reset_index(drop=True)


def build_targets(core, h):
    """
    Attach the horizon-h target: wILI at season_week_idx + h, same region,
    same season. Indexing within the season rather than by epiweek
    arithmetic avoids the year-boundary off-by-one (202301 - 1 is 202252 or
    202253 depending on the year).
    """
    key = core[["season", "hhs_region", "season_week_idx", "wili_final"]].copy()
    key["season_week_idx"] -= h
    key = key.rename(columns={"wili_final": "y_target"})
    out = core.merge(key, on=["season", "hhs_region", "season_week_idx"], how="left")
    assert len(out) == len(core), "target join changed row count"
    return out


def usable(df):
    """Origins where every model can forecast and the target exists."""
    return df[df["y_target"].notna()
              & df[LAG_COLS].notna().all(axis=1)
              & df["wili_final"].notna()].copy()


# ──────────────────────────────────────────────────────────────────────
# models
# ──────────────────────────────────────────────────────────────────────

def b0_persistence(test):
    """Last published value at origin T. Deterministic, no fitting."""
    return test["ili_asof_lag1"].values


def b1_climatology(train_all, test):
    """
    Mean wILI per (region, MMWR week) over the training seasons only.

    Built from EVERY training-season row, not only those carrying a
    horizon-h target. A climatology is a statement about the observed value
    in a calendar week; restricting it to rows that happen to have a target
    would truncate the last h weeks of each training season and leave the
    corresponding test origins with no prediction, which then silently
    unpairs the model comparison.

    Returns predictions and the number of prior seasons behind each one, so
    that a cell resting on a single observation is never silently presented
    as a climatology.
    """
    src = train_all[train_all["wili_final"].notna()]
    clim = (src.groupby(["hhs_region", "week"])
               .agg(clim_mean=("wili_final", "mean"),
                    n_seasons=("season", "nunique"))
               .reset_index())
    # the target week is h ahead of the origin week
    tgt = test[["hhs_region", "target_week"]].rename(columns={"target_week": "week"})
    merged = tgt.merge(clim, on=["hhs_region", "week"], how="left")
    return merged["clim_mean"].values, merged["n_seasons"].values


def b2_gbm(train, test, seed):
    m = HistGradientBoostingRegressor(random_state=seed, **B2_PARAMS)
    m.fit(train[B2_FEATURES].values, train["y_target"].values)
    return m.predict(test[B2_FEATURES].values)


# ──────────────────────────────────────────────────────────────────────

def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


STRATA = [
    ("all", lambda d: np.ones(len(d), dtype=bool)),
    ("pre-peak", lambda d: (d["peak_side"] == "pre-peak").values),
    ("peak or after", lambda d: (d["peak_side"] == "peak or after").values),
    ("genomic-covered", lambda d: d["has_genomic"].values),
]


def score(rows, fold, h, model, seed, test, preds):
    y = test["y_target"].values
    for name, fn in STRATA:
        m = fn(test)
        if m.sum() < 5:
            continue
        rows.append({
            "fold": fold, "horizon": h, "model": model, "seed": seed,
            "stratum": name, "n": int(m.sum()),
            "mae": round(mae(y[m], preds[m]), 4),
            "rmse": round(rmse(y[m], preds[m]), 4),
            "mean_target": round(float(np.mean(y[m])), 4),
        })


def assert_no_leakage(train, test, fold):
    if train.empty or test.empty:
        sys.exit(f"{fold}: empty split")
    if train["epiweek"].max() >= test["epiweek"].min():
        sys.exit(f"{fold}: train max epiweek {train['epiweek'].max()} "
                 f">= test min {test['epiweek'].min()} - temporal leakage")
    if set(train["season"]) & set(test["season"]):
        sys.exit(f"{fold}: a season appears in both train and test")
    if "wili_final" in B2_FEATURES:
        sys.exit("wili_final must never be a feature - it is the target")


# ──────────────────────────────────────────────────────────────────────

def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    print("EpiSurge E2 - baselines B0-B2")
    print("=" * 70)

    core = load()
    print(f"\nCore-season rows: {len(core)}")
    print("Folds (rolling origin):")
    for name, tr, te in FOLDS:
        print(f"  {name}: train {', '.join(tr)}  ->  test {te}")

    rows = []
    clim_support = []

    for h in HORIZONS:
        d = build_targets(core, h)
        d["target_week"] = d.groupby(["season", "hhs_region"])["week"].shift(-h)

        for fold, train_seasons, test_season in FOLDS:
            train_all = d[d["season"].isin(train_seasons)]      # climatology source
            train = usable(train_all)                           # B2 fitting set
            test = usable(d[d["season"] == test_season])
            if train.empty or test.empty:
                print(f"  [skip] {fold} h={h}: no usable rows")
                continue
            assert_no_leakage(train, test, fold)

            # B1 first: it defines the evaluation set. Every model is then
            # scored on exactly these rows so comparisons are paired
            # (Master Plan 7.4).
            p1, nseas = b1_climatology(train_all, test)
            ok = ~np.isnan(p1)
            if ok.sum() < 5:
                print(f"  [skip] {fold} h={h}: climatology undefined for all rows")
                continue
            test_p = test[ok].copy()

            clim_support.append({
                "fold": fold, "horizon": h,
                "n_test": int(len(test)), "n_evaluated": int(ok.sum()),
                "min_prior_seasons": int(np.nanmin(nseas[ok])),
                "max_prior_seasons": int(np.nanmax(nseas[ok])),
            })

            score(rows, fold, h, "B0", 0, test_p, b0_persistence(test_p))
            score(rows, fold, h, "B1", 0, test_p, p1[ok])
            for seed in SEEDS:
                score(rows, fold, h, "B2", seed, test_p, b2_gbm(train, test_p, seed))

    res = pd.DataFrame(rows)
    if res.empty:
        sys.exit("No results produced")
    res.to_csv(REPORTS / "baseline_results.csv", index=False)

    # ── summary: average over seeds, then pool folds by weighted mean ──
    per_fold = (res.groupby(["fold", "horizon", "model", "stratum"])
                   .agg(mae=("mae", "mean"), mae_sd=("mae", "std"),
                        rmse=("rmse", "mean"), n=("n", "first"))
                   .reset_index())
    per_fold["mae_sd"] = per_fold["mae_sd"].fillna(0.0)

    def pooled(g):
        w = g["n"].values
        return pd.Series({
            "mae": float(np.average(g["mae"], weights=w)),
            "rmse": float(np.average(g["rmse"], weights=w)),
            "mae_sd": float(np.average(g["mae_sd"], weights=w)),
            "n": int(w.sum()),
        })

    summary = (per_fold.groupby(["horizon", "model", "stratum"])
                       .apply(pooled, include_groups=False)
                       .reset_index())
    summary.to_csv(REPORTS / "baseline_summary.csv", index=False)

    # ── console tables ────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("TABLE 1  MAE by horizon and model  (all test rows, folds pooled)")
    print("-" * 70)
    a = summary[summary.stratum == "all"]
    piv = a.pivot_table(index="horizon", columns="model", values="mae").round(3)
    piv["n"] = a.groupby("horizon")["n"].first()
    print(piv.to_string())

    print("\n" + "-" * 70)
    print("TABLE 2  Skill vs persistence  (positive = better than B0, %)")
    print("-" * 70)
    base = a[a.model == "B0"].set_index("horizon")["mae"]
    sk = a[a.model != "B0"].copy()
    sk["skill_pct"] = sk.apply(
        lambda r: round(100 * (base[r.horizon] - r.mae) / base[r.horizon], 1), axis=1)
    print(sk.pivot_table(index="horizon", columns="model",
                         values="skill_pct").to_string())
    print("\n  Pooled figures above are a weighted mean ACROSS FOLDS and can")
    print("  describe no individual fold. On the original three seasons the")
    print("  two folds gave -8.9% and +56.7% at h=8, pooling to +18.7%.")
    print("  TABLE 2b is the one to read.")

    # ── per-fold, the reportable view ─────────────────────────────────
    print("\n" + "-" * 70)
    print("TABLE 2b  Skill vs persistence BY FOLD  (%, seeds averaged)")
    print("-" * 70)
    pf = (res[res.stratum == "all"]
          .groupby(["fold", "horizon", "model"])["mae"].mean().reset_index())
    for model in ["B1", "B2"]:
        sub = pf[pf.model.isin(["B0", model])]
        wide = sub.pivot_table(index="fold", columns=["horizon", "model"],
                               values="mae")
        rows = {}
        for fold in sorted(pf.fold.unique()):
            rows[fold] = {}
            for h in HORIZONS:
                try:
                    b0 = wide.loc[fold, (h, "B0")]
                    mm = wide.loc[fold, (h, model)]
                    rows[fold][h] = round(100 * (b0 - mm) / b0, 1)
                except (KeyError, ZeroDivisionError):
                    rows[fold][h] = np.nan
        tab = pd.DataFrame(rows).T
        tab.index.name = f"{NAMES[model]} vs B0"
        print(f"\n  {NAMES[model]} — skill vs persistence, per fold:")
        print(tab.to_string())
        spread = tab.max() - tab.min()
        print(f"    fold-to-fold spread by horizon: "
              f"{', '.join(f'h{h}={spread[h]:.0f}pp' for h in HORIZONS if h in spread)}")
        print(f"    sign consistent across folds: "
              f"{', '.join(f'h{h}=' + ('yes' if (tab[h].dropna() > 0).all() or (tab[h].dropna() < 0).all() else 'NO') for h in HORIZONS if h in tab)}")
    print("\n  A horizon whose sign flips across folds supports no claim.")

    print("\n" + "-" * 70)
    print("TABLE 3  MAE by stratum  (folds pooled, seeds averaged)")
    print("-" * 70)
    for st in ["pre-peak", "peak or after", "genomic-covered"]:
        s = summary[summary.stratum == st]
        if s.empty:
            continue
        print(f"\n  {st}:")
        p = s.pivot_table(index="horizon", columns="model", values="mae").round(3)
        p["n"] = s.groupby("horizon")["n"].first()
        print(p.to_string())

    print("\n" + "-" * 70)
    print("TABLE 4  B2 across 5 seeds (stratum=all)")
    print("-" * 70)
    b2 = summary[(summary.model == "B2") & (summary.stratum == "all")]
    print(b2[["horizon", "mae", "mae_sd"]].round(4).to_string(index=False))
    if float(b2["mae_sd"].max()) == 0.0:
        print("\n  mae_sd is 0 at every horizon because HistGradientBoostingRegressor")
        print("  is deterministic given fixed training data and early_stopping=False;")
        print("  random_state changes nothing here. This is NOT evidence of stability.")
        print("  The 5-seed protocol in Master Plan 7.4 bites on stochastic models")
        print("  (weight init, dropout, batch order) - i.e. the GNN in Phase 3.")

    if clim_support:
        print("\n" + "-" * 70)
        print("TABLE 5  Climatological baseline support (how thin B1 really is)")
        print("-" * 70)
        print(pd.DataFrame(clim_support).to_string(index=False))

    # ── figures ───────────────────────────────────────────────────────
    f1 = fig_by_horizon(summary)
    f2 = fig_skill(a, base)
    print(f"\n  wrote {f1}\n  wrote {f2}")

    # ── checks ────────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("Checks")
    print("-" * 70)
    fails = []
    for h in HORIZONS:
        n_models = a[a.horizon == h].model.nunique()
        if n_models < 3:
            fails.append(f"h={h}: only {n_models} models scored")
    paired = res.groupby(["fold", "horizon", "stratum"])["n"].nunique()
    if (paired > 1).any():
        bad = paired[paired > 1]
        fails.append(f"unpaired evaluation rows in {len(bad)} fold/horizon/stratum cells "
                     f"- B1 drops rows without climatology, check TABLE 5")
    print(f"  models scored at every horizon: {'yes' if not fails else 'no'}")
    print(f"  rows per (fold, horizon, stratum) identical across models: "
          f"{'yes' if (paired <= 1).all() else 'no - see note'}")

    if fails:
        print("\n  NOTES:")
        for f in fails:
            print(f"    [!] {f}")
    print("\nE2 complete. No genomic features used anywhere in this script.")


def fig_by_horizon(summary):
    fig, ax = plt.subplots(figsize=(8, 5))
    a = summary[summary.stratum == "all"]
    for m in ["B0", "B1", "B2"]:
        s = a[a.model == m].sort_values("horizon")
        if s.empty:
            continue
        ax.plot(s.horizon, s.mae, marker="o", linewidth=2.2,
                color=COLOURS[m], label=NAMES[m])
        if m == "B2":
            ax.fill_between(s.horizon, s.mae - s.mae_sd, s.mae + s.mae_sd,
                            color=COLOURS[m], alpha=0.15, linewidth=0)
    ax.set_xlabel("Forecast horizon (weeks)")
    ax.set_ylabel("MAE (wILI percentage points)")
    ax.set_title("Baseline forecast error by horizon", fontsize=12, loc="left")
    ax.set_xticks(HORIZONS)
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = FIGURES / "fig_baseline_by_horizon.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_skill(a, base):
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in ["B1", "B2"]:
        s = a[a.model == m].sort_values("horizon")
        if s.empty:
            continue
        skill = [100 * (base[h] - v) / base[h] for h, v in zip(s.horizon, s.mae)]
        ax.plot(s.horizon, skill, marker="o", linewidth=2.2,
                color=COLOURS[m], label=NAMES[m])
    ax.axhline(0, color="#888888", linewidth=1.3, linestyle="--")
    ax.annotate("persistence (B0)", xy=(HORIZONS[-1], 0), xytext=(6.1, 2),
                fontsize=9, color="#666666")
    ax.set_xlabel("Forecast horizon (weeks)")
    ax.set_ylabel("MAE improvement over persistence (%)")
    ax.set_title("Persistence collapses at long horizons; the climatological "
                 "baseline is the real bar there", fontsize=11, loc="left")
    ax.set_xticks(HORIZONS)
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = FIGURES / "fig_baseline_skill_vs_persistence.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


if __name__ == "__main__":
    main()