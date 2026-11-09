#!/usr/bin/env python3
"""
EpiSurge - Phase 3 / E3: genomic feature ablation.

THE GATE. This is the experiment the project exists to run, and every later
Phase 3 step is conditioned on its outcome (PHASE3_PLAN.md 9, 10).

Research question
-----------------
Does genomic composition add forecasting skill over case history, and how
much of any gain is destroyed by publication delay?

E1 established that delay removes genomic coverage specifically before the
epidemic peak: 97.1% of core-season region-weeks carry collection-dated
composition before the peak against 34.8% under publication-dating, a 62.3
point gap, with 8-11 consecutive onset weeks at exactly zero in every
season. Published work that uses pathogen genomics for incidence
forecasting indexes sequences by COLLECTION date, which credits a forecast
with data that did not exist at its origin. E3 measures what that costs.

Arms
----
Each arm adds a feature block to E2's B2 features (8 as-of ILI lags, region,
season-week index). Everything else is held fixed.

  A0   none                              = B2 exactly, the reference
  A1   GF3 as-of effort only             is any gain just surveillance volume?
  A2   GF1 as-of raw                     deployable composition level
  A2s  GF1 as-of SHRUNK                  what empirical-Bayes shrinkage buys
  A3   GF1 + GF2 as-of                   deployable level + change
  A4   GF1+GF2+GF4+GF5 as-of             DEPLOYABLE, FULL
  A5   GF1 oracle raw                    literature-equivalent level
  A6   GF1+GF2+GF4+GF5 oracle            THE CEILING

The four comparisons that matter:

  A6 - A4    the delay cost            PRIMARY ENDPOINT
  A5 - A0, A6 - A0                     does the oracle carry signal at all
  A4 - A0                              does DEPLOYABLE genomics carry signal
  A4 - A1                              genomics, or sequencing effort

WHY A2s EXISTS, AND WHY THE PRIMARY ARMS USE RAW PROPORTIONS. The master
table carries asof_*_shrunk_* but only oracle_prop_* - there is no shrunk
oracle. PHASE3_PLAN.md 4 originally specified "GF1 as-of" against "GF1
oracle" without pinning the representation, which would have compared
SHRUNK against RAW and confounded the availability contrast with the
shrinkage. A5 and A6 therefore use oracle_prop_*, A2-A4 use
asof_*_prop_*, and A2s measures shrinkage separately against A2. Recorded
as a correction to the plan.

Inference
---------
Five folds is far too few for a t-test across folds. The primary tool is
SIGN CONSISTENCY, the rule already standing in DECISIONS: a horizon whose
sign flips across folds supports no claim.

Secondary: a block bootstrap over REGIONS within the test season. Each test
fold is one season, so the exchangeable blocks are its 10 HHS regions, not
its rows - rows within a region are strongly autocorrelated and resampling
them would give intervals several times too narrow.

Fold 4 (test 2022-23) is the COVID discontinuity and is reported
separately, not averaged in. Pre-COVID seasons peak at MMWR weeks 05-07;
2022-23 peaked at week 47. It broke every baseline in E2 and is expected to
break this.

Seeds
-----
One seed, deliberately. E2 measured mae_sd = 0.0 at every horizon because
HistGradientBoostingRegressor is deterministic given fixed training data
with early_stopping=False. Eight arms x 5 folds x 6 horizons x 5 seeds
would be 1,200 fits for zero additional information. The 5-seed protocol
bites on the GNN in E6, where it is kept.

Shared definitions
------------------
Splits, row usability, target construction, strata and metrics are IMPORTED
from baselines.py rather than reimplemented, so E2 and E3 are provably
evaluated on the same rows under the same fold definitions. If that import
ever breaks, the comparison is void and this script should fail rather than
silently diverge.

Inputs:
    data/processed/master_region_week.csv     (needs the GF2-GF5 columns:
                                               run build_genomic_features.py)

Outputs:
    data/reports/ablation_results.csv      arm x fold x horizon x stratum
    data/reports/ablation_contrasts.csv    every contrast, per fold and horizon
    data/reports/ablation_delay_cost.csv   the primary endpoint, with CIs
    figures/fig_ablation_headline.png
    figures/fig_ablation_arms.png
    figures/fig_ablation_by_fold.png

Run (from the repository root, inside the venv):
    python src/analysis/genomic_ablation.py
    python src/analysis/genomic_ablation.py --window L4_12   # sensitivity
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor

_src = str(Path(__file__).resolve().parents[1])
if _src not in sys.path:
    sys.path.insert(0, _src)
_here = str(Path(__file__).resolve().parent)
if _here not in sys.path:
    sys.path.insert(0, _here)

# shared with E2 on purpose - see "Shared definitions" above
from baselines import (                                       # noqa: E402
    FOLDS, HORIZONS, B2_FEATURES, B2_PARAMS,
    load, build_targets, usable, assert_no_leakage,
)

import plotstyle                                              # noqa: E402

REPORTS = Path("data/reports")
FIGURES = Path("figures")

N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 20261108

PRIMARY = ("A6", "A4")          # the delay cost
PRIMARY_STRATUM = "pre-peak"

COVID_FOLD_TEST = "2022-23"     # reported separately, never averaged in


# ──────────────────────────────────────────────────────────────────────
# arm definitions
# ──────────────────────────────────────────────────────────────────────

def feature_blocks(window):
    """
    Column names per GF family, for the as-of window in use and for oracle.

    `window` is the as-of tag as it appears in the master table, e.g.
    "L8_16". L8-16 is primary (64.4% core coverage); L4-12 is the
    sensitivity check.
    """
    a = f"asof_{window}"
    return {
        ("asof", "GF1_raw"):    [f"{a}_prop_h1n1", f"{a}_prop_h3n2"],
        ("asof", "GF1_shrunk"): [f"{a}_shrunk_h1n1", f"{a}_shrunk_h3n2"],
        ("asof", "GF2"):        [f"{a}_js_div_4w", f"{a}_d_prop_h3n2_4w"],
        ("asof", "GF3"):        [f"{a}_n_total", f"{a}_seq_per_100k",
                                 f"{a}_n_total_log_ratio"],
        ("asof", "GF4"):        [f"{a}_js_vs_national", f"{a}_js_vs_others"],
        ("asof", "GF5"):        [f"{a}_frac_offdominant"],
        ("oracle", "GF1_raw"):  ["oracle_prop_h1n1", "oracle_prop_h3n2"],
        ("oracle", "GF2"):      ["oracle_js_div_4w", "oracle_d_prop_h3n2_4w"],
        ("oracle", "GF4"):      ["oracle_js_vs_national", "oracle_js_vs_others"],
        ("oracle", "GF5"):      ["oracle_frac_offdominant"],
    }


# `prop_other` is excluded everywhere: composition is binary in practice
# (H3N2 vs H1N1) so the three proportions sum to 1 and the third is
# redundant, while `other` is additionally near-constant at zero. Two
# columns carry the whole simplex.
ARM_SPEC = {
    "A0":  ("reference",  []),
    "A1":  ("deployable", [("asof", "GF3")]),
    "A2":  ("deployable", [("asof", "GF1_raw")]),
    "A2s": ("deployable", [("asof", "GF1_shrunk")]),
    "A3":  ("deployable", [("asof", "GF1_raw"), ("asof", "GF2")]),
    "A4":  ("deployable", [("asof", "GF1_raw"), ("asof", "GF2"),
                           ("asof", "GF4"), ("asof", "GF5")]),
    "A5":  ("oracle",     [("oracle", "GF1_raw")]),
    "A6":  ("oracle",     [("oracle", "GF1_raw"), ("oracle", "GF2"),
                           ("oracle", "GF4"), ("oracle", "GF5")]),
}

ARM_ORDER = ["A0", "A1", "A2", "A2s", "A3", "A4", "A5", "A6"]

ARM_LABEL = {
    "A0": "A0  case history only",
    "A1": "A1  + sequencing effort",
    "A2": "A2  + as-of composition",
    "A2s": "A2s + as-of composition (shrunk)",
    "A3": "A3  + as-of level & change",
    "A4": "A4  + as-of, all families",
    "A5": "A5  + oracle composition",
    "A6": "A6  + oracle, all families",
}

CONTRASTS = [
    ("A6", "A4", "delay cost (PRIMARY)"),
    ("A4", "A0", "deployable genomics vs case history"),
    ("A6", "A0", "oracle genomics vs case history"),
    ("A5", "A0", "oracle level vs case history"),
    ("A2", "A0", "as-of level vs case history"),
    ("A1", "A0", "sequencing effort vs case history"),
    ("A4", "A1", "genomics beyond effort"),
    ("A2s", "A2", "what shrinkage buys"),
]


def arm_features(arm, blocks):
    cols = list(B2_FEATURES)
    for key in ARM_SPEC[arm][1]:
        cols.extend(blocks[key])
    return cols


# ──────────────────────────────────────────────────────────────────────

def fit_predict(train, test, cols, seed=0):
    m = HistGradientBoostingRegressor(random_state=seed, **B2_PARAMS)
    m.fit(train[cols].values, train["y_target"].values)
    return m.predict(test[cols].values)


def block_bootstrap_delta(err_a, err_b, regions, rng, n=N_BOOTSTRAP):
    """
    Bootstrap CI for mean(err_a) - mean(err_b), resampling REGIONS.

    err_a and err_b are per-row absolute errors on the SAME rows, so the
    difference is paired. The resampling unit is the region: a test fold is
    a single season, and wILI within a region is strongly autocorrelated
    week to week, so resampling rows would understate the interval
    severely.
    """
    uniq = np.unique(regions)
    if len(uniq) < 3:
        return np.nan, np.nan
    idx_by_region = {r: np.where(regions == r)[0] for r in uniq}
    deltas = np.empty(n)
    for b in range(n):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_region[r] for r in picked])
        deltas[b] = err_a[rows].mean() - err_b[rows].mean()
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


STRATA = [
    ("all", lambda d: np.ones(len(d), dtype=bool)),
    ("pre-peak", lambda d: (d["peak_side"] == "pre-peak").values),
    ("peak or after", lambda d: (d["peak_side"] == "peak or after").values),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="L8_16", choices=["L8_16", "L4_12"],
                    help="as-of window for the deployable arms "
                         "(L8_16 primary, L4_12 sensitivity)")
    ap.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    args = ap.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    plotstyle.apply()

    print("EpiSurge E3 - genomic feature ablation")
    print("=" * 74)
    print(f"As-of window for deployable arms: {args.window.replace('_', '-')}")

    core = load()
    blocks = feature_blocks(args.window)

    # ── every required column must exist, named, before any fitting ────
    needed = sorted({c for cols in blocks.values() for c in cols})
    missing = [c for c in needed if c not in core.columns]
    if missing:
        print("\n  Missing genomic feature columns:")
        for c in missing:
            print(f"    {c}")
        sys.exit("\nRun build_genomic_features.py then build_master_table.py "
                 "(PHASE3_PLAN.md 9, steps 2).")

    print(f"\nCore-season rows: {len(core)}")
    print("Arms:")
    for arm in ARM_ORDER:
        kind, keys = ARM_SPEC[arm]
        n = len(arm_features(arm, blocks)) - len(B2_FEATURES)
        print(f"  {arm:<4} {kind:<11} +{n:>2} columns   {ARM_LABEL[arm]}")

    # the target must never appear as a feature in any arm. oracle_* columns
    # ARE leaky with respect to availability - that is what they are for -
    # but wili_final is the thing being predicted.
    for arm in ARM_ORDER:
        bad = [c for c in arm_features(arm, blocks)
               if c in ("wili_final", "y_target")]
        if bad:
            sys.exit(f"{arm} contains the target as a feature: {bad}")

    print("\nFeature availability on core rows (fraction non-missing):")
    for arm in ARM_ORDER[1:]:
        cols = [c for c in arm_features(arm, blocks) if c not in B2_FEATURES]
        frac = core[cols].notna().all(axis=1).mean()
        print(f"  {arm:<4} all its genomic columns present: {100 * frac:5.1f}%")
    print("  (HistGradientBoostingRegressor handles NaN natively, so no row is")
    print("   dropped and every arm is scored on an identical row set.)")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows, contrast_rows = [], []

    for h in HORIZONS:
        d = build_targets(core, h)

        for fold, train_seasons, test_season in FOLDS:
            train = usable(d[d["season"].isin(train_seasons)])
            test = usable(d[d["season"] == test_season])
            if train.empty or test.empty:
                print(f"  [skip] {fold} h={h}: no usable rows")
                continue
            assert_no_leakage(train, test, fold)

            y = test["y_target"].values
            regions = test["hhs_region"].values
            errs = {}

            for arm in ARM_ORDER:
                cols = arm_features(arm, blocks)
                p = fit_predict(train, test, cols)
                errs[arm] = np.abs(y - p)

                for sname, sfn in STRATA:
                    m = sfn(test)
                    if m.sum() < 5:
                        continue
                    rows.append({
                        "window": args.window, "fold": fold,
                        "test_season": test_season, "horizon": h,
                        "arm": arm, "kind": ARM_SPEC[arm][0],
                        "stratum": sname, "n": int(m.sum()),
                        "mae": round(float(errs[arm][m].mean()), 4),
                        "rmse": round(float(np.sqrt(
                            (errs[arm][m] ** 2).mean())), 4),
                    })

            for a, b, label in CONTRASTS:
                for sname, sfn in STRATA:
                    m = sfn(test)
                    if m.sum() < 5:
                        continue
                    da = errs[a][m].mean() - errs[b][m].mean()
                    lo, hi = (block_bootstrap_delta(
                        errs[a][m], errs[b][m], regions[m], rng, args.bootstrap)
                        if sname == PRIMARY_STRATUM or (a, b) == PRIMARY
                        else (np.nan, np.nan))
                    base = errs[b][m].mean()
                    contrast_rows.append({
                        "window": args.window, "fold": fold,
                        "test_season": test_season, "horizon": h,
                        "contrast": f"{a}-{b}", "label": label,
                        "stratum": sname, "n": int(m.sum()),
                        "d_mae": round(float(da), 4),
                        "skill_pct": round(float(-100 * da / base), 2)
                        if base > 0 else np.nan,
                        "ci_lo": round(lo, 4) if lo == lo else np.nan,
                        "ci_hi": round(hi, 4) if hi == hi else np.nan,
                    })

        print(f"  h={h}: {len(ARM_ORDER)} arms x {len(FOLDS)} folds fitted")

    res = pd.DataFrame(rows)
    con = pd.DataFrame(contrast_rows)
    if res.empty:
        sys.exit("No results produced")

    res.to_csv(REPORTS / "ablation_results.csv", index=False)
    con.to_csv(REPORTS / "ablation_contrasts.csv", index=False)

    # ── pairing check ─────────────────────────────────────────────────
    paired = res.groupby(["fold", "horizon", "stratum"])["n"].nunique()
    pairing_ok = bool((paired <= 1).all())

    # ── TABLE 1: arms by horizon, per fold, stratum=all ───────────────
    print("\n" + "-" * 74)
    print("TABLE 1  MAE by arm and horizon (stratum=all, mean over folds)")
    print("-" * 74)
    a_all = res[res.stratum == "all"]
    piv = (a_all.pivot_table(index="horizon", columns="arm", values="mae")
                .reindex(columns=ARM_ORDER).round(3))
    print(piv.to_string())
    print("\n  Fold-averaged, shown for orientation only. Per-fold is the")
    print("  reportable view - TABLE 3.")

    # ── TABLE 2: the primary endpoint ─────────────────────────────────
    print("\n" + "-" * 74)
    print(f"TABLE 2  PRIMARY ENDPOINT  {PRIMARY[0]}-{PRIMARY[1]} "
          f"(delay cost), stratum={PRIMARY_STRATUM}")
    print("-" * 74)
    pc = con[(con.contrast == f"{PRIMARY[0]}-{PRIMARY[1]}")
             & (con.stratum == PRIMARY_STRATUM)]
    if pc.empty:
        print("  not evaluable - no pre-peak rows met the minimum")
    else:
        show = pc[["fold", "test_season", "horizon", "n",
                   "d_mae", "skill_pct", "ci_lo", "ci_hi"]]
        print(show.to_string(index=False))
        print("\n  d_mae < 0 means the ORACLE arm has lower error than the")
        print("  deployable arm, i.e. publication delay costs skill.")
        pc.to_csv(REPORTS / "ablation_delay_cost.csv", index=False)

    # ── TABLE 3: sign consistency, the inference that counts ──────────
    print("\n" + "-" * 74)
    print("TABLE 3  Sign consistency across folds (stratum=all)")
    print("-" * 74)
    precovid = [f for f, _, te in FOLDS if te != COVID_FOLD_TEST]
    verdicts = []
    for a, b, label in CONTRASTS:
        sub = con[(con.contrast == f"{a}-{b}") & (con.stratum == "all")]
        if sub.empty:
            continue
        line = {"contrast": f"{a}-{b}", "label": label}
        for h in HORIZONS:
            s = sub[(sub.horizon == h) & (sub.fold.isin(precovid))]["skill_pct"]
            s = s.dropna()
            if s.empty:
                line[f"h{h}"] = "-"
            elif (s > 0).all():
                line[f"h{h}"] = f"+{s.mean():.1f}"
            elif (s < 0).all():
                line[f"h{h}"] = f"{s.mean():.1f}"
            else:
                line[f"h{h}"] = "FLIP"
        verdicts.append(line)
    vt = pd.DataFrame(verdicts)
    print(vt.to_string(index=False))
    print(f"\n  Pre-COVID folds only ({', '.join(precovid)}); fold testing "
          f"{COVID_FOLD_TEST} is reported separately below.")
    print("  FLIP = the sign is not consistent across folds, so no claim is")
    print("  available at that horizon. A signed number is the mean skill %")
    print("  across folds that all agree in sign.")

    # ── TABLE 4: the COVID fold, on its own ───────────────────────────
    covid = [f for f, _, te in FOLDS if te == COVID_FOLD_TEST]
    if covid:
        print("\n" + "-" * 74)
        print(f"TABLE 4  The COVID-discontinuity fold ({COVID_FOLD_TEST}), "
              f"reported separately")
        print("-" * 74)
        cf = con[(con.fold.isin(covid)) & (con.stratum == "all")]
        print(cf.pivot_table(index="contrast", columns="horizon",
                             values="skill_pct").round(1).to_string())

    # ── TABLE 5: phase stratification ─────────────────────────────────
    print("\n" + "-" * 74)
    print("TABLE 5  Key contrasts by epidemic phase (pre-COVID folds, mean)")
    print("-" * 74)
    for a, b, label in CONTRASTS[:4]:
        sub = con[(con.contrast == f"{a}-{b}") & (con.fold.isin(precovid))]
        if sub.empty:
            continue
        print(f"\n  {a}-{b}  {label}")
        print(sub.pivot_table(index="stratum", columns="horizon",
                              values="skill_pct").round(1).to_string())

    # ── TABLE 6: controls. A null is only credible with these ─────────
    print("\n" + "-" * 74)
    print("TABLE 6  Controls (stratum=all, skill % vs A0 per fold)")
    print("-" * 74)
    run_controls(core, blocks, precovid)

    # ── TABLE 7: what effect size could we have detected? ─────────────
    print("\n" + "-" * 74)
    print("TABLE 7  Detectable effect size on the primary endpoint")
    print("-" * 74)
    pw = con[(con.contrast == f"{PRIMARY[0]}-{PRIMARY[1]}")
             & (con.stratum == PRIMARY_STRATUM)].dropna(subset=["ci_lo"])
    a0 = res[(res.arm == "A0") & (res.stratum == PRIMARY_STRATUM)] \
        .groupby("horizon")["mae"].mean()
    if not pw.empty:
        pw = pw.assign(width=pw.ci_hi - pw.ci_lo)
        band = []
        for h in sorted(pw.horizon.unique()):
            half = pw[pw.horizon == h]["width"].median() / 2
            band.append({"horizon": h, "A0_mae": round(float(a0[h]), 3),
                         "ci_half_width": round(float(half), 4),
                         "detectable_pct": round(100 * half / a0[h], 1)})
        print(pd.DataFrame(band).to_string(index=False))
        n_sig = int(((pw.ci_lo > 0) | (pw.ci_hi < 0)).sum())
        print(f"\n  {n_sig} of {len(pw)} fold x horizon CIs exclude zero.")
        print("  detectable_pct is the smallest delay cost this design could")
        print("  have resolved. A null above that band would be a power")
        print("  failure; a null below it is a measurement.")

    # ── figures ───────────────────────────────────────────────────────
    f1 = fig_headline(con, precovid)
    f2 = fig_arms(res)
    f3 = fig_by_fold(con)
    print(f"\n  wrote {f1}\n  wrote {f2}\n  wrote {f3}")

    # ── checks ────────────────────────────────────────────────────────
    print("\n" + "-" * 74)
    print("Checks")
    print("-" * 74)
    print(f"  rows per (fold, horizon, stratum) identical across arms: "
          f"{'yes' if pairing_ok else 'NO'}")
    print(f"  arms scored at every horizon: "
          f"{'yes' if a_all.groupby('horizon')['arm'].nunique().eq(len(ARM_ORDER)).all() else 'NO'}")
    print(f"  bootstrap resamples: {args.bootstrap} over "
          f"{core['hhs_region'].nunique()} regions, seed {BOOTSTRAP_SEED}")

    if not pairing_ok:
        sys.exit("\n[FAIL] unpaired evaluation - comparisons are not valid")

    print("\nE3 complete. Read TABLE 3 first: a FLIP at a horizon means that")
    print("horizon supports no claim, however large the fold-averaged number.")


# ──────────────────────────────────────────────────────────────────────
# controls
# ──────────────────────────────────────────────────────────────────────

CONTROL_SEED = 7


def run_controls(core, blocks, precovid):
    """
    Three controls, run every time. A null genomic result is worth nothing
    without them, and a reviewer will ask for exactly these.

    POSITIVE CONTROL: the target plus Gaussian noise, as one extra column.
    This MUST produce a large improvement. If it does not, the harness
    cannot detect signal that is present and every null here is a bug
    rather than a finding.

    NOISE CONTROL: as many standard-normal columns as the full genomic arm
    adds. This calibrates what a gradient booster loses from uninformative
    extra columns on this sample size, so "genomics does not help" can be
    compared against "random columns do not help".

    MISSINGNESS CONTROL: a single indicator of whether as-of composition is
    present, carrying no composition values at all. As-of coverage is near
    zero before the peak and high after it (E1), so this flag is a partial
    proxy for position in season - and peak timing differs between seasons.
    If the flag alone degrades the forecast, the harm from the genomic arms
    is about the missingness pattern rather than the composition values.
    """
    rng = np.random.default_rng(CONTROL_SEED)
    gen_cols = [c for c in arm_features("A4", blocks) if c not in B2_FEATURES]
    probe = blocks[("asof", "GF1_raw")][1]      # any as-of composition column

    print(f"{'h':>3}  {'control':<28} {'per fold':<40} verdict")
    for h in HORIZONS:
        d = build_targets(core, h)
        noise = [f"_noise{i}" for i in range(len(gen_cols))]
        for c in noise:
            d[c] = rng.standard_normal(len(d))
        d["_positive"] = d["y_target"] + rng.normal(0, 0.5, len(d))
        d["_missflag"] = d[probe].isna().astype(float)

        base = {}
        for fold, tr, te in FOLDS:
            train, test = usable(d[d.season.isin(tr)]), usable(d[d.season == te])
            base[fold] = float(np.abs(
                test["y_target"].values
                - fit_predict(train, test, list(B2_FEATURES))).mean())

        for label, extra in [("A4 genomic", gen_cols),
                             (f"pure noise ({len(noise)} cols)", noise),
                             ("missingness flag alone", ["_missflag"]),
                             ("POSITIVE CONTROL", ["_positive"])]:
            sk = {}
            for fold, tr, te in FOLDS:
                train = usable(d[d.season.isin(tr)])
                test = usable(d[d.season == te])
                cols = list(B2_FEATURES) + extra
                e = float(np.abs(test["y_target"].values
                                 - fit_predict(train, test, cols)).mean())
                sk[fold] = 100 * (base[fold] - e) / base[fold]
            vals = [sk[f] for f in precovid]
            v = (f"{np.mean(vals):+.1f}%"
                 if (all(x > 0 for x in vals) or all(x < 0 for x in vals))
                 else "FLIP")
            cells = "  ".join(f"{f[-1]}:{sk[f]:+6.1f}" for f in sorted(sk))
            print(f"{h:>3}  {label:<28} {cells:<40} {v}")
        print()
    print("  If POSITIVE CONTROL is not strongly and consistently positive,")
    print("  stop: the harness cannot see signal and no null below is valid.")


# ──────────────────────────────────────────────────────────────────────
# figures
# ──────────────────────────────────────────────────────────────────────

def fig_headline(con, precovid):
    """
    The headline: deployable against oracle genomic skill over case history,
    by horizon, pre-peak, pre-COVID folds. The vertical distance between
    the two lines is the delay cost.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 5))

    series = [("A4-A0", "Deployable (published by forecast origin)",
               plotstyle.PALETTE[0]),
              ("A6-A0", "Oracle (collection-dated, as in the literature)",
               plotstyle.PALETTE[1])]

    for key, label, colour in series:
        sub = con[(con.contrast == key) & (con.stratum == "pre-peak")
                  & (con.fold.isin(precovid))]
        if sub.empty:
            continue
        g = sub.groupby("horizon")["skill_pct"].agg(["mean", "min", "max"])
        ax.plot(g.index, g["mean"], marker="o", color=colour, label=label)
        ax.fill_between(g.index, g["min"], g["max"], color=colour,
                        alpha=0.15, linewidth=0)

    plotstyle.zero_line(ax, "case history alone (A0)", x=HORIZONS[0])
    ax.set_xlabel("Forecast horizon (weeks)")
    ax.set_ylabel("MAE improvement over case history (%)")
    ax.set_title("Before the peak, genomic signal exists only under\n"
                 "collection-dating, which no forecaster has")
    ax.set_xticks(HORIZONS)
    ax.legend()
    plotstyle.finish(ax)
    fig.tight_layout()
    return plotstyle.save(fig, "fig_ablation_headline")


def fig_arms(res):
    """Every arm's MAE by horizon, deployable and oracle distinguished."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    a = res[res.stratum == "all"]
    for i, arm in enumerate(ARM_ORDER):
        s = a[a.arm == arm].groupby("horizon")["mae"].mean()
        if s.empty:
            continue
        kind = ARM_SPEC[arm][0]
        style = {"reference": "-", "deployable": "-", "oracle": "--"}[kind]
        colour = ("#000000" if kind == "reference"
                  else plotstyle.PALETTE[i % len(plotstyle.PALETTE)])
        ax.plot(s.index, s.values, marker="o", linestyle=style,
                color=colour, label=ARM_LABEL[arm],
                linewidth=2.4 if kind == "reference" else 1.8)
    ax.set_xlabel("Forecast horizon (weeks)")
    ax.set_ylabel("MAE (wILI percentage points)")
    ax.set_title("Ablation arms by horizon (dashed = oracle, not deployable)")
    ax.set_xticks(HORIZONS)
    ax.legend(fontsize=8, ncol=2)
    plotstyle.finish(ax)
    fig.tight_layout()
    return plotstyle.save(fig, "fig_ablation_arms")


def fig_by_fold(con):
    """
    The primary endpoint fold by fold. A sign flip here is visible rather
    than averaged away - the rule E2 established.
    """
    import matplotlib.pyplot as plt
    keys = ["A4-A0", "A6-A0", "A6-A4"]
    titles = ["Deployable vs case history", "Oracle vs case history",
              "Delay cost (oracle - deployable)"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    folds = sorted(con.fold.unique())
    cmap = plt.get_cmap("viridis")

    for ax, key, title in zip(axes, keys, titles):
        sub = con[(con.contrast == key) & (con.stratum == "all")]
        for i, fold in enumerate(folds):
            s = sub[sub.fold == fold].sort_values("horizon")
            if s.empty:
                continue
            te = s["test_season"].iloc[0]
            ax.plot(s.horizon, s.skill_pct, marker="o", markersize=4,
                    linewidth=1.9, color=cmap(i / max(1, len(folds) - 1)),
                    label=f"{fold} ({te})")
        ax.axhline(0, color=plotstyle.GREY, linestyle="--", linewidth=1.3)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Horizon (weeks)")
        ax.set_xticks(HORIZONS)
        plotstyle.finish(ax)
    axes[0].set_ylabel("MAE improvement (%)")
    axes[0].legend(fontsize=7.5, ncol=1)
    fig.suptitle("Per-fold view: a horizon whose sign flips supports no claim",
                 x=0.01, ha="left")
    fig.tight_layout()
    return plotstyle.save(fig, "fig_ablation_by_fold")


if __name__ == "__main__":
    main()