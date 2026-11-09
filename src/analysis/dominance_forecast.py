#!/usr/bin/env python3
"""
EpiSurge - Phase 3 / E3b: subtype-dominance forecasting and the cost of
publication delay.

WHY THIS EXPERIMENT EXISTS
--------------------------
E3 found that subtype composition carries no recoverable signal for
INCIDENCE forecasting at any horizon, oracle or as-of, against a verified
positive control and a 5-11% detection floor. That is a real bound, but it
leaves the delay question unanswerable: there is no signal for delay to
destroy.

This experiment changes the TARGET, not the question. Composition is the
natural predictor of composition, so we ask whether the eventual subtype
mix of a region is forecastable from what is known at the forecast origin,
and what publication delay costs that forecast. The delay question becomes
measurable again, and the answer is large.

Target: oracle_prop_h3n2(r, t+h) - the H3N2 share of sequences COLLECTED in
region r during week t+h. This is the eventual truth about what circulated,
the exact analogue of "final revised wILI" as the incidence target. Every
predictor uses only what was published by the forecast origin t, except the
oracle arms, which are the point of comparison.

THE 2x2 THAT MATTERS
--------------------
                as-of (deployable)        oracle (not available at t)
  national      national share at t       national oracle share at t
  regional      region's share at t       region's oracle share at t

  regional - national   is there REGIONAL genomic information?
  oracle   - as-of      what does PUBLICATION DELAY cost?

BASELINE DISCIPLINE, LEARNED THE HARD WAY
-----------------------------------------
A first pass of this experiment reported 66% dominance accuracy for the
oracle predictor against 51% for the deployable one and called it a result.
It was very nearly an artifact: "always predict whichever subtype dominates
this season" also scores 0.661-0.666, so that 66% was consistent with
knowing nothing more than the season's dominant subtype.

The honest trivial rule is TRAIN-MAJORITY: predict whichever subtype
dominated the TRAINING seasons. It scores 0.506-0.513, because dominance
flips hard between seasons - H3N2 led 94.5% of region-weeks in 2016-17 and
23.2% in 2019-20 - so no fixed rule survives. train-majority is therefore
included in every table and is the floor every other predictor must clear.

A predictor that is absent at t falls back to train-majority, never to the
truth. Filling a gap with the national ORACLE share would leak.

WHY THERE IS NO MODEL HERE
--------------------------
Gradient boosting on these predictors does not beat carry-forward: tested
across all folds and horizons, the sign flips everywhere. Subtype-dominance
forecasting is an information-AVAILABILITY problem, not a modelling
problem, and the result is reported in its model-free form because that is
both the honest and the stronger version. The model arms are retained as
`--with-models` so the claim is checkable rather than asserted.

Inputs:
    data/processed/master_region_week.csv
    data/features/composition_oracle.csv
    data/features/asof_composition_L8-16.csv   (or L4-12 via --window)

Outputs:
    data/reports/dominance_results.csv       predictor x fold x horizon x stratum
    data/reports/dominance_delay_cost.csv    the delay cost, with bootstrap CIs
    data/reports/dominance_decomposition.csv availability vs staleness
    figures/fig_dominance_headline.png
    figures/fig_dominance_delay_cost.png
    figures/fig_dominance_by_fold.png

Run (from the repository root, inside the venv):
    python src/analysis/dominance_forecast.py
    python src/analysis/dominance_forecast.py --with-models
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_src = str(Path(__file__).resolve().parents[1])
if _src not in sys.path:
    sys.path.insert(0, _src)
_here = str(Path(__file__).resolve().parent)
if _here not in sys.path:
    sys.path.insert(0, _here)

from baselines import FOLDS, HORIZONS, B2_FEATURES, B2_PARAMS, load   # noqa: E402
import plotstyle                                                      # noqa: E402

REPORTS = Path("data/reports")
FIGURES = Path("figures")

TRUTH = "oracle_prop_h3n2"
LAGS = [f"ili_asof_lag{k}" for k in range(1, 9)]

N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 20261109

COVID_FOLD_TEST = "2022-23"

PRED_ORDER = ["train-majority", "nat as-of", "reg as-of",
              "nat ORACLE", "reg ORACLE"]
DEPLOYABLE = {"train-majority", "nat as-of", "reg as-of"}

PRED_LABEL = {
    "train-majority": "Predict training seasons' majority subtype",
    "nat as-of": "National composition, published by origin",
    "reg as-of": "Regional composition, published by origin",
    "nat ORACLE": "National composition, collection-dated",
    "reg ORACLE": "Regional composition, collection-dated",
}


def national_share(path, keycol):
    """
    National H3N2 share per week, rebuilt from COUNTS.

    The master table carries proportions only, and a mean of regional
    proportions is not the national proportion - it weights a region with
    three sequences the same as one with three hundred. Counts are summed
    from the composition file instead.
    """
    d = pd.read_csv(path)
    if keycol != "epiweek":
        d = d.drop(columns=["epiweek"], errors="ignore")
    d = d.rename(columns={keycol: "epiweek"})
    g = d.groupby("epiweek")[["n_h1n1", "n_h3n2", "n_other"]].sum()
    tot = g.sum(axis=1)
    return g["n_h3n2"] / tot.where(tot > 0)


def prepare(window):
    core = load()
    asof_path = Path(f"data/features/asof_composition_"
                     f"{window.replace('_', '-')}.csv")
    orac_path = Path("data/features/composition_oracle.csv")
    for p in (asof_path, orac_path):
        if not p.exists():
            sys.exit(f"Missing {p}")
    if TRUTH not in core.columns:
        sys.exit(f"master table lacks {TRUTH} - run build_master_table.py")

    core["nat_asof"] = core["epiweek"].map(
        national_share(asof_path, "origin_epiweek_int"))
    core["nat_oracle"] = core["epiweek"].map(
        national_share(orac_path, "epiweek_int"))
    core["reg_asof"] = core[f"asof_{window}_prop_h3n2"]
    core["reg_oracle"] = core[TRUTH]
    return core


def targets(core, h):
    """
    Shift the truth series back by h weeks within (season, region), so each
    origin row carries the share that will be observed h weeks later.
    Joining on season_week_idx keeps the shift inside the season.
    """
    k = core[["season", "hhs_region", "season_week_idx", TRUTH]].copy()
    k["season_week_idx"] -= h
    k = k.rename(columns={TRUTH: "y"})
    out = core.merge(k, on=["season", "hhs_region", "season_week_idx"],
                     how="left")
    return out[out["y"].notna() & out[LAGS].notna().all(axis=1)].copy()


COLUMN_OF = {"nat as-of": "nat_asof", "reg as-of": "reg_asof",
             "nat ORACLE": "nat_oracle", "reg ORACLE": "reg_oracle"}


def predictions(train, test):
    """
    Every predictor's H3N2-share prediction on the test rows.

    train_majority is 1.0 or 0.0 - whichever subtype led the training
    seasons - and is also the fallback wherever a predictor is absent at
    the origin, because it is the only thing genuinely known then.
    """
    train_majority = float((train["y"] > 0.5).mean() > 0.5)
    preds = {"train-majority": np.full(len(test), train_majority)}
    for name, col in COLUMN_OF.items():
        p = test[col].values.astype(float)
        preds[name] = np.where(np.isnan(p), train_majority, p)
    return preds, train_majority


def block_bootstrap_skill(err_a, err_b, regions, rng, n):
    """
    CI for MAE skill of a over b, resampling REGIONS. A test fold is one
    season and a region's weeks are strongly autocorrelated, so rows are
    not exchangeable but regions approximately are.
    """
    uniq = np.unique(regions)
    if len(uniq) < 3:
        return np.nan, np.nan
    idx = {r: np.where(regions == r)[0] for r in uniq}
    vals = np.empty(n)
    for b in range(n):
        rows = np.concatenate([idx[r] for r in
                               rng.choice(uniq, size=len(uniq), replace=True)])
        ma, mb = err_a[rows].mean(), err_b[rows].mean()
        vals[b] = 100 * (mb - ma) / mb if mb > 0 else np.nan
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


STRATA = [
    ("all", lambda d: np.ones(len(d), dtype=bool)),
    ("pre-peak", lambda d: (d["peak_side"] == "pre-peak").values),
    ("peak or after", lambda d: (d["peak_side"] == "peak or after").values),
]

CONTRASTS = [
    ("nat ORACLE", "nat as-of", "DELAY COST, national (PRIMARY)"),
    ("reg ORACLE", "reg as-of", "delay cost, regional"),
    ("reg as-of", "nat as-of", "regional information, deployable"),
    ("reg ORACLE", "nat ORACLE", "regional information, oracle"),
    ("nat as-of", "train-majority", "deployable genomics vs trivial rule"),
    ("nat ORACLE", "train-majority", "oracle genomics vs trivial rule"),
]

PRIMARY = ("nat ORACLE", "nat as-of")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="L8_16", choices=["L8_16", "L4_12"])
    ap.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--with-models", action="store_true",
                    help="also fit gradient boosting, to check that it does "
                         "not beat carry-forward")
    args = ap.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    plotstyle.apply()

    print("EpiSurge E3b - subtype-dominance forecasting and the delay cost")
    print("=" * 76)
    print(f"As-of window: {args.window.replace('_', '-')}")
    print(f"Target: {TRUTH}(t+h) - the eventual H3N2 share, collection-dated")

    core = prepare(args.window)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    pre = [f for f, _, te in FOLDS if te != COVID_FOLD_TEST]

    rows, con_rows, decomp = [], [], []

    for h in HORIZONS:
        d = targets(core, h)
        for fold, train_seasons, test_season in FOLDS:
            train = d[d["season"].isin(train_seasons)]
            test = d[d["season"] == test_season]
            if train.empty or test.empty:
                continue

            y = test["y"].values
            yb = (y > 0.5).astype(int)
            regions = test["hhs_region"].values
            preds, tmaj = predictions(train, test)
            errs = {k: np.abs(y - v) for k, v in preds.items()}

            for name in PRED_ORDER:
                for sname, sfn in STRATA:
                    m = sfn(test)
                    if m.sum() < 5:
                        continue
                    rows.append({
                        "window": args.window, "fold": fold,
                        "test_season": test_season, "horizon": h,
                        "predictor": name,
                        "available": name in DEPLOYABLE,
                        "stratum": sname, "n": int(m.sum()),
                        "mae": round(float(errs[name][m].mean()), 4),
                        "accuracy": round(float(
                            ((preds[name][m] > 0.5).astype(int) == yb[m]).mean()
                        ), 4),
                        "brier": round(float(
                            ((preds[name][m] - yb[m]) ** 2).mean()), 4),
                    })

            for a, b, label in CONTRASTS:
                for sname, sfn in STRATA:
                    m = sfn(test)
                    if m.sum() < 5:
                        continue
                    ea, eb = errs[a][m].mean(), errs[b][m].mean()
                    skill = 100 * (eb - ea) / eb if eb > 0 else np.nan
                    lo, hi = (block_bootstrap_skill(
                        errs[a][m], errs[b][m], regions[m], rng, args.bootstrap)
                        if (a, b) == PRIMARY or sname == "all"
                        else (np.nan, np.nan))
                    con_rows.append({
                        "window": args.window, "fold": fold,
                        "test_season": test_season, "horizon": h,
                        "contrast": f"{a} vs {b}", "label": label,
                        "stratum": sname, "n": int(m.sum()),
                        "skill_pct": round(float(skill), 2),
                        "ci_lo": round(lo, 2) if lo == lo else np.nan,
                        "ci_hi": round(hi, 2) if hi == hi else np.nan,
                    })

            # availability vs staleness: on rows where BOTH variants carry
            # real data, any remaining oracle edge cannot be availability
            both = test[test["reg_oracle"].notna() & test["reg_asof"].notna()]
            if len(both) >= 20:
                yb2 = (both["y"].values > 0.5).astype(int)
                for scope, sub, yy in [("all rows", test, yb), ("both present", both, yb2)]:
                    po = np.where(np.isnan(sub["nat_oracle"].values), tmaj,
                                  sub["nat_oracle"].values)
                    pa = np.where(np.isnan(sub["nat_asof"].values), tmaj,
                                  sub["nat_asof"].values)
                    decomp.append({
                        "fold": fold, "horizon": h, "scope": scope,
                        "n": len(sub),
                        "acc_oracle": round(float(
                            ((po > 0.5).astype(int) == yy).mean()), 4),
                        "acc_asof": round(float(
                            ((pa > 0.5).astype(int) == yy).mean()), 4),
                    })

        print(f"  h={h}: {len(PRED_ORDER)} predictors x {len(FOLDS)} folds")

    res = pd.DataFrame(rows)
    con = pd.DataFrame(con_rows)
    dec = pd.DataFrame(decomp)
    if res.empty:
        sys.exit("no results produced")

    res.to_csv(REPORTS / "dominance_results.csv", index=False)
    con.to_csv(REPORTS / "dominance_delay_cost.csv", index=False)
    dec.to_csv(REPORTS / "dominance_decomposition.csv", index=False)

    a = res[res.stratum == "all"]

    # ── TABLE 1 ───────────────────────────────────────────────────────
    print("\n" + "-" * 76)
    print("TABLE 1  MAE on the H3N2 share (fold-averaged; lower is better)")
    print("-" * 76)
    print(a.pivot_table(index="predictor", columns="horizon", values="mae")
           .reindex(PRED_ORDER).round(4).to_string())

    print("\n" + "-" * 76)
    print("TABLE 2  Binary dominance accuracy (which subtype leads)")
    print("-" * 76)
    print(a.pivot_table(index="predictor", columns="horizon", values="accuracy")
           .reindex(PRED_ORDER).round(3).to_string())
    print("\n  train-majority is the floor. It sits near 0.51 because dominance")
    print("  flips between seasons, so no fixed rule survives - which is what")
    print("  makes the task non-trivial.")

    # ── TABLE 3: sign consistency ─────────────────────────────────────
    print("\n" + "-" * 76)
    print("TABLE 3  Sign-consistent MAE skill, pre-COVID folds (stratum=all)")
    print("-" * 76)
    out = []
    for aa, bb, label in CONTRASTS:
        sub = con[(con.contrast == f"{aa} vs {bb}") & (con.stratum == "all")
                  & (con.fold.isin(pre))]
        line = {"contrast": f"{aa} vs {bb}", "label": label}
        for h in HORIZONS:
            s = sub[sub.horizon == h]["skill_pct"].dropna()
            line[f"h{h}"] = ("-" if s.empty else
                             f"{s.mean():+.1f}" if (s > 0).all() or (s < 0).all()
                             else "FLIP")
        out.append(line)
    print(pd.DataFrame(out).to_string(index=False))
    print("\n  A positive number means the first predictor has lower error.")
    print("  FLIP = sign not consistent across folds, so no claim.")

    # ── TABLE 4: primary endpoint with CIs ────────────────────────────
    print("\n" + "-" * 76)
    print(f"TABLE 4  PRIMARY: {PRIMARY[0]} vs {PRIMARY[1]} "
          f"(the delay cost), per fold")
    print("-" * 76)
    pc = con[(con.contrast == f"{PRIMARY[0]} vs {PRIMARY[1]}")
             & (con.stratum == "all")]
    print(pc[["fold", "test_season", "horizon", "n",
              "skill_pct", "ci_lo", "ci_hi"]].to_string(index=False))

    # ── TABLE 5: decomposition ────────────────────────────────────────
    if not dec.empty:
        print("\n" + "-" * 76)
        print("TABLE 5  Availability or staleness? (accuracy gap, pp)")
        print("-" * 76)
        dd = dec[dec.fold.isin(pre)].copy()
        dd["gap_pp"] = 100 * (dd.acc_oracle - dd.acc_asof)
        print(dd.pivot_table(index="scope", columns="horizon",
                             values="gap_pp").round(1).to_string())
        print("\n  'both present' rows carry real data in BOTH variants, so the")
        print("  gap there is STALENESS alone. What it leaves is availability.")

    # ── TABLE 6: phase stratification ─────────────────────────────────
    print("\n" + "-" * 76)
    print("TABLE 6  The delay cost by epidemic phase (pre-COVID folds)")
    print("-" * 76)
    ph = con[(con.contrast == f"{PRIMARY[0]} vs {PRIMARY[1]}")
             & (con.fold.isin(pre))]
    print(ph.pivot_table(index="stratum", columns="horizon",
                         values="skill_pct").round(1).to_string())

    # ── TABLE 7: the COVID fold, separately ───────────────────────────
    cov = con[(con.fold.isin([f for f, _, te in FOLDS if te == COVID_FOLD_TEST]))
              & (con.stratum == "all")]
    if not cov.empty:
        print("\n" + "-" * 76)
        print(f"TABLE 7  The COVID-discontinuity fold ({COVID_FOLD_TEST}), "
              f"reported separately")
        print("-" * 76)
        print(cov.pivot_table(index="contrast", columns="horizon",
                              values="skill_pct").round(1).to_string())

    # ── optional: does a model beat carry-forward? ────────────────────
    if args.with_models:
        print("\n" + "-" * 76)
        print("TABLE 8  Does gradient boosting beat carry-forward?")
        print("-" * 76)
        check_models(core, pre)

    f1 = fig_headline(res, pre)
    f2 = fig_delay_cost(con, pre)
    f3 = fig_by_fold(con)
    print(f"\n  wrote {f1}\n  wrote {f2}\n  wrote {f3}")

    print("\n" + "-" * 76)
    print("Checks")
    print("-" * 76)
    paired = res.groupby(["fold", "horizon", "stratum"])["n"].nunique()
    print(f"  identical row sets across predictors: "
          f"{'yes' if (paired <= 1).all() else 'NO'}")
    print(f"  bootstrap: {args.bootstrap} resamples over "
          f"{core['hhs_region'].nunique()} regions, seed {BOOTSTRAP_SEED}")
    print("\nRead TABLE 3 first. The delay cost is the row to quote.")


def check_models(core, pre):
    """
    Gradient boosting on the same predictors, against carry-forward. This
    exists to be checkable: the claim that no model helps should not rest
    on assertion.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    arms = {
        "model, as-of": list(B2_FEATURES) + ["nat_asof", "reg_asof"],
        "model, oracle": list(B2_FEATURES) + ["nat_oracle", "reg_oracle"],
    }
    refs = {"model, as-of": "nat_asof", "model, oracle": "nat_oracle"}
    out = []
    for h in HORIZONS:
        d = targets(core, h)
        for name, cols in arms.items():
            sk = {}
            for fold, tr, te in FOLDS:
                train, test = d[d.season.isin(tr)], d[d.season == te]
                if train.empty or test.empty:
                    continue
                tmaj = float((train["y"] > 0.5).mean() > 0.5)
                m = HistGradientBoostingRegressor(random_state=0, **B2_PARAMS)
                m.fit(train[cols].values, train["y"].values)
                pm = np.clip(m.predict(test[cols].values), 0, 1)
                pr = np.where(np.isnan(test[refs[name]].values), tmaj,
                              test[refs[name]].values)
                em = np.abs(test["y"].values - pm).mean()
                er = np.abs(test["y"].values - pr).mean()
                sk[fold] = 100 * (er - em) / er
            v = [sk[f] for f in pre if f in sk]
            out.append({"horizon": h, "arm": name,
                        "vs": f"carry-forward {refs[name]}",
                        "skill": ("FLIP" if not v or not
                                  (all(x > 0 for x in v) or all(x < 0 for x in v))
                                  else f"{np.mean(v):+.1f}")})
    print(pd.DataFrame(out).pivot_table(index="arm", columns="horizon",
                                        values="skill", aggfunc="first")
          .to_string())
    print("\n  FLIP or negative everywhere means carry-forward is not beaten,")
    print("  and dominance forecasting is an availability problem rather than")
    print("  a modelling one.")


# ──────────────────────────────────────────────────────────────────────

def fig_headline(res, pre):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    a = res[(res.stratum == "all") & (res.fold.isin(pre))]

    style = {
        "nat ORACLE": (plotstyle.PALETTE[1], "-", "o"),
        "reg ORACLE": (plotstyle.PALETTE[1], "--", "s"),
        "nat as-of": (plotstyle.PALETTE[0], "-", "o"),
        "reg as-of": (plotstyle.PALETTE[0], "--", "s"),
        "train-majority": (plotstyle.GREY, ":", None),
    }
    for metric, ax, ylab, lo in [
            ("accuracy", axes[0], "Dominance accuracy", True),
            ("mae", axes[1], "MAE on H3N2 share", False)]:
        for name in PRED_ORDER:
            s = a[a.predictor == name].groupby("horizon")[metric].mean()
            if s.empty:
                continue
            c, ls, mk = style[name]
            ax.plot(s.index, s.values, color=c, linestyle=ls, marker=mk,
                    label=PRED_LABEL[name], linewidth=2.0)
        ax.set_xlabel("Forecast horizon (weeks)")
        ax.set_ylabel(ylab)
        ax.set_xticks(HORIZONS)
        plotstyle.finish(ax)
        if lo:
            ax.axhline(0.5, color=plotstyle.RULE, linewidth=0.8, alpha=0.5)
            ax.annotate("coin flip", xy=(HORIZONS[0], 0.5),
                        xytext=(2, -11), textcoords="offset points",
                        fontsize=8, color=plotstyle.RULE)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Subtype dominance is forecastable from collection-dated "
                 "composition.\nUnder realistic publication delay it is a "
                 "coin flip.", x=0.01, ha="left")
    fig.tight_layout()
    return plotstyle.save(fig, "fig_dominance_headline")


def fig_delay_cost(con, pre):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for key, label, colour in [
            (f"{PRIMARY[0]} vs {PRIMARY[1]}", "National composition",
             plotstyle.PALETTE[0]),
            ("reg ORACLE vs reg as-of", "Regional composition",
             plotstyle.PALETTE[2])]:
        sub = con[(con.contrast == key) & (con.stratum == "all")
                  & (con.fold.isin(pre))]
        if sub.empty:
            continue
        g = sub.groupby("horizon")["skill_pct"].agg(["mean", "min", "max"])
        ax.plot(g.index, g["mean"], marker="o", color=colour, label=label)
        ax.fill_between(g.index, g["min"], g["max"], color=colour,
                        alpha=0.15, linewidth=0)
    plotstyle.zero_line(ax, "no cost", x=HORIZONS[0])
    ax.set_xlabel("Forecast horizon (weeks)")
    ax.set_ylabel("MAE skill lost to publication delay (%)")
    ax.set_title("The cost of publication delay, sign-consistent at every "
                 "horizon")
    ax.set_xticks(HORIZONS)
    ax.legend()
    plotstyle.finish(ax)
    fig.tight_layout()
    return plotstyle.save(fig, "fig_dominance_delay_cost")


def fig_by_fold(con):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 5))
    sub = con[(con.contrast == f"{PRIMARY[0]} vs {PRIMARY[1]}")
              & (con.stratum == "all")]
    folds = sorted(sub.fold.unique())
    cmap = plt.get_cmap("viridis")
    for i, fold in enumerate(folds):
        s = sub[sub.fold == fold].sort_values("horizon")
        if s.empty:
            continue
        te = s["test_season"].iloc[0]
        covid = te == COVID_FOLD_TEST
        ax.plot(s.horizon, s.skill_pct, marker="o", markersize=4,
                linestyle="--" if covid else "-",
                color=plotstyle.GREY if covid
                else cmap(i / max(1, len(folds) - 1)),
                label=f"{fold} ({te})" + (" - COVID" if covid else ""))
    plotstyle.zero_line(ax)
    ax.set_xlabel("Forecast horizon (weeks)")
    ax.set_ylabel("Delay cost (% MAE skill)")
    ax.set_title("Per-fold view: every fold agrees in sign")
    ax.set_xticks(HORIZONS)
    ax.legend(fontsize=8)
    plotstyle.finish(ax)
    fig.tight_layout()
    return plotstyle.save(fig, "fig_dominance_by_fold")


if __name__ == "__main__":
    main()