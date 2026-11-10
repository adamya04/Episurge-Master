#!/usr/bin/env python3
"""
EpiSurge - Phase 3 / E3c: what would it take to recover the signal?

E3b measured the damage: publication delay costs 43-50% of
subtype-dominance forecasting skill, sign-consistent at every horizon in
every fold. That is a diagnosis. This experiment asks the question a
surveillance programme would actually ask next - what would have to change?

Two levers exist, and the interesting finding is that NEITHER WORKS ALONE.

  1. SUBMISSION SPEED. Shift every submission date D days earlier.
  2. COLLECTION WINDOW. At origin t, current practice reads sequences
     collected in [t-16wk, t-8wk]. A tighter window is closer to the
     present and should carry more signal.

Measured on this dataset, against current practice (L8-16, D=0):

  faster submission alone, L8-16 at D=84      +4.6 to  +6.5 %
  tighter window alone,    L0-4  at D=0      -21.3 to -39.9 %
  BOTH,                    L0-4  at D=56     +33.3 to +34.1 %
  BOTH,                    L0-4  at D=84     +34.7 to +38.2 %

A tighter window ALONE is reliably worse than doing nothing, because at
today's submission speed 89% of forecast origins have no sequence in a
0-4 week window at all and the forecast falls back to a prior. Faster
submission is what makes a tight window usable; a tight window is what
converts the speed into signal. The interaction is the result.

The mechanism is reported alongside the skill, as the fraction of origin
weeks with no usable data - without it the interaction looks like a
curiosity rather than a consequence.

WHAT THIS IS AND IS NOT
-----------------------
This is a COUNTERFACTUAL simulation on real submission dates, not a
forecast of what a faster system would achieve. It assumes delay reduction
is UNIFORM - every sequence D days earlier - which no real intervention
delivers; real speedups change the shape of the delay distribution, not
just its location. Treat the surface as an order-of-magnitude guide to
which lever matters, not as a service-level target.

D is a shift, not a resulting delay. Median collection-to-submission delay
in this dataset is 86 days, so D=56 corresponds to a median of about 30
days and D=84 to near-immediate submission.

Inference is sign consistency across the pre-COVID folds, the rule standing
since E2: a horizon whose sign flips supports no claim. The fold testing
2022-23 is reported separately.

Inputs:
    data/processed/master_region_week.csv
    data/processed/submission_dates.json  (via build_asof_composition)
    data/processed/mapped_*.json          (via build_asof_composition)
    data/features/composition_oracle.csv

Outputs:
    data/reports/policy_surface.csv          skill and coverage per cell
    data/reports/policy_surface_signs.csv    sign consistency per cell
    figures/fig_policy_surface.png
    figures/fig_policy_mechanism.png
    figures/fig_policy_frontier.png

Run (from the repository root, inside the venv):
    python src/analysis/delay_policy_surface.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from epiweeks import Week

_src = str(Path(__file__).resolve().parents[1])
if _src not in sys.path:
    sys.path.insert(0, _src)
for _extra in (str(Path(__file__).resolve().parent), f"{_src}/features"):
    if _extra not in sys.path:
        sys.path.insert(0, _extra)

# the project's OWN record loader - never reimplemented here. A scratch
# version of this experiment silently parsed every date wrongly (dates are
# "14-Dec-2017", not ISO) and produced an all-missing series that looked
# like a flat null result.
from build_asof_composition import load_records, epiweek_start   # noqa: E402
from baselines import FOLDS, HORIZONS, load                      # noqa: E402
from config import SEASON_WINDOWS                                # noqa: E402
import plotstyle                                                 # noqa: E402

REPORTS = Path("data/reports")
FIGURES = Path("figures")

TRUTH = "oracle_prop_h3n2"
LAGS = [f"ili_asof_lag{k}" for k in range(1, 9)]

CURRENT = ("L8-16", 0)          # current practice, the baseline every cell beats or does not
COVID_FOLD_TEST = "2022-23"

WINDOWS = [(0, 2), (0, 4), (2, 6), (4, 12), (8, 16)]
SHIFTS = [0, 14, 28, 56, 84]

N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 20261109

SUBTYPE_CODE = {"H1N1": 0, "H3N2": 1, "other": 2}


def load_arrays():
    """Collection ordinal, submission ordinal and subtype code per record."""
    records = load_records()
    col = np.array([r["collection_date"].toordinal() for r in records])
    sub = np.array([r["submission_date"].toordinal() for r in records])
    st = np.array([SUBTYPE_CODE.get(r["subtype"], 3) for r in records])
    keep = st < 3                     # unsubtyped carries no composition
    print(f"  classified records: {keep.sum()} of {len(records)}")
    print(f"  median collection-to-submission delay: "
          f"{np.median(sub[keep] - col[keep]):.0f} days")
    return col[keep], sub[keep], st[keep]


def origin_weeks():
    out = []
    for start, end in SEASON_WINDOWS:
        w = Week(start // 100, start % 100)
        while w.year * 100 + w.week <= end:
            out.append((w.year, w.week))
            w = w + 1
    return out


def national_series(col, sub, st, od, ew, lo, hi, shift):
    """
    National H3N2 share at each forecast origin, under a counterfactual
    window [t-hi wk, t-lo wk) and submission `shift` days earlier.

    NaN where no sequence qualifies - that is the mechanism the headline
    depends on, so it is never filled in here.
    """
    out = np.full(len(od), np.nan)
    for i, o in enumerate(od):
        m = (col >= o - hi * 7) & (col < o - lo * 7) & (sub - shift <= o)
        if not m.any():
            continue
        counts = np.bincount(st[m], minlength=3)
        total = counts.sum()
        if total:
            out[i] = counts[1] / total
    return dict(zip(ew, out))


def oracle_series():
    d = (pd.read_csv("data/features/composition_oracle.csv")
           .drop(columns=["epiweek"], errors="ignore")
           .rename(columns={"epiweek_int": "epiweek"}))
    g = d.groupby("epiweek")[["n_h1n1", "n_h3n2", "n_other"]].sum()
    tot = g.sum(axis=1)
    return (g["n_h3n2"] / tot.where(tot > 0)).to_dict()


def targets(core, h):
    k = core[["season", "hhs_region", "season_week_idx", TRUTH]].copy()
    k["season_week_idx"] -= h
    d = core.merge(k.rename(columns={TRUTH: "y"}),
                   on=["season", "hhs_region", "season_week_idx"], how="left")
    return d[d["y"].notna() & d[LAGS].notna().all(axis=1)].copy()


def block_bootstrap(err_a, err_b, regions, rng, n):
    uniq = np.unique(regions)
    if len(uniq) < 3:
        return np.nan, np.nan
    idx = {r: np.where(regions == r)[0] for r in uniq}
    vals = np.empty(n)
    for b in range(n):
        rows = np.concatenate([idx[r] for r in
                               rng.choice(uniq, size=len(uniq), replace=True)])
        mb = err_b[rows].mean()
        vals[b] = 100 * (mb - err_a[rows].mean()) / mb if mb > 0 else np.nan
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def cell_name(lo, hi, shift):
    return f"L{lo}-{hi}|D{shift}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    args = ap.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    plotstyle.apply()

    print("EpiSurge E3c - what would it take to recover the signal?")
    print("=" * 76)
    print("Counterfactual: submission D days earlier, collection window [t-hi, t-lo).")
    print("D is a SHIFT, not a resulting delay.\n")

    col, sub, st = load_arrays()
    ows = origin_weeks()
    od = np.array([epiweek_start(y, w).toordinal() for y, w in ows])
    ew = np.array([y * 100 + w for y, w in ows])

    core = load()
    core["nat_oracle"] = core["epiweek"].map(oracle_series())

    print(f"\n  Building {len(WINDOWS) * len(SHIFTS)} counterfactual series "
          f"over {len(ows)} origin weeks")
    for lo, hi in WINDOWS:
        for D in SHIFTS:
            core[cell_name(lo, hi, D)] = core["epiweek"].map(
                national_series(col, sub, st, od, ew, lo, hi, D))

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    pre = [f for f, _, te in FOLDS if te != COVID_FOLD_TEST]
    base_col = cell_name(8, 16, 0)

    rows = []
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
            tmaj = float((train["y"] > 0.5).mean() > 0.5)

            def errors(colname):
                raw = test[colname].values
                p = np.where(np.isnan(raw), tmaj, raw)
                return np.abs(y - p), (p > 0.5).astype(int), np.isnan(raw).mean()

            e_base, _, _ = errors(base_col)
            e_orac, _, _ = errors("nat_oracle")
            mae_base, mae_orac = e_base.mean(), e_orac.mean()

            for lo, hi in WINDOWS:
                for D in SHIFTS:
                    name = cell_name(lo, hi, D)
                    e, pb, miss = errors(name)
                    skill = (100 * (mae_base - e.mean()) / mae_base
                             if mae_base > 0 else np.nan)
                    recovered = (100 * (mae_base - e.mean()) /
                                 (mae_base - mae_orac)
                                 if mae_base > mae_orac else np.nan)
                    lo_ci, hi_ci = (block_bootstrap(e, e_base, regions, rng,
                                                    args.bootstrap)
                                    if (lo, hi) in ((0, 4), (4, 12)) and D in (28, 56)
                                    else (np.nan, np.nan))
                    rows.append({
                        "fold": fold, "test_season": test_season, "horizon": h,
                        "window": f"L{lo}-{hi}", "shift_days": D,
                        "n": len(test),
                        "mae": round(float(e.mean()), 4),
                        "accuracy": round(float((pb == yb).mean()), 4),
                        "no_data_frac": round(float(miss), 4),
                        "skill_vs_current": round(float(skill), 2),
                        "pct_gap_recovered": round(float(recovered), 1),
                        "ci_lo": round(lo_ci, 2) if lo_ci == lo_ci else np.nan,
                        "ci_hi": round(hi_ci, 2) if hi_ci == hi_ci else np.nan,
                    })
        print(f"  h={h}: {len(WINDOWS) * len(SHIFTS)} cells x {len(FOLDS)} folds")

    res = pd.DataFrame(rows)
    res.to_csv(REPORTS / "policy_surface.csv", index=False)
    pr = res[res.fold.isin(pre)]

    print("\n" + "-" * 76)
    print("TABLE 1  Skill vs current practice, % (pre-COVID folds, mean over horizons)")
    print("-" * 76)
    print(pr.pivot_table(index="window", columns="shift_days",
                         values="skill_vs_current")
            .reindex([f"L{a}-{b}" for a, b in WINDOWS]).round(1).to_string())
    print("\n  Current practice is L8-16 at D=0, so that cell is 0 by construction.")

    print("\n" + "-" * 76)
    print("TABLE 2  THE MECHANISM - fraction of forecast origins with no usable data")
    print("-" * 76)
    print(pr.pivot_table(index="window", columns="shift_days",
                         values="no_data_frac")
            .reindex([f"L{a}-{b}" for a, b in WINDOWS]).round(3).to_string())
    print("\n  A tight window at today's submission speed is mostly empty. That is")
    print("  why tightening alone makes things worse, and why the two levers")
    print("  only work together.")

    print("\n" + "-" * 76)
    print("TABLE 3  Dominance accuracy (pre-COVID folds, mean over horizons)")
    print("-" * 76)
    print(pr.pivot_table(index="window", columns="shift_days", values="accuracy")
            .reindex([f"L{a}-{b}" for a, b in WINDOWS]).round(3).to_string())

    # ── sign consistency: the inference that counts ───────────────────
    print("\n" + "-" * 76)
    print("TABLE 4  Sign consistency vs current practice (pre-COVID folds)")
    print("-" * 76)
    signs = []
    for lo, hi in WINDOWS:
        for D in SHIFTS:
            if (f"L{lo}-{hi}", D) == CURRENT:
                continue
            line = {"window": f"L{lo}-{hi}", "shift_days": D}
            for h in HORIZONS:
                s = pr[(pr.window == f"L{lo}-{hi}") & (pr.shift_days == D)
                       & (pr.horizon == h)]["skill_vs_current"].dropna()
                line[f"h{h}"] = ("-" if s.empty else
                                 f"{s.mean():+.1f}"
                                 if (s > 0).all() or (s < 0).all() else "FLIP")
            signs.append(line)
    sg = pd.DataFrame(signs)
    sg.to_csv(REPORTS / "policy_surface_signs.csv", index=False)
    print(sg.to_string(index=False))
    print("\n  FLIP = sign not consistent across folds, so no claim at that horizon.")

    print("\n" + "-" * 76)
    print("TABLE 5  The three headline cells, per fold (h=4)")
    print("-" * 76)
    for win, D, label in [("L8-16", 84, "faster submission ALONE"),
                          ("L0-4", 0, "tighter window ALONE"),
                          ("L0-4", 56, "BOTH")]:
        s = res[(res.window == win) & (res.shift_days == D) & (res.horizon == 4)]
        cells = "  ".join(f"{r.fold[-1]}:{r.skill_vs_current:+6.1f}"
                          for r in s.itertuples())
        print(f"  {label:<26} {win:<7} D={D:<3} {cells}")

    f1 = fig_surface(pr)
    f2 = fig_mechanism(pr)
    f3 = fig_frontier(pr)
    print(f"\n  wrote {f1}\n  wrote {f2}\n  wrote {f3}")

    print("\n" + "-" * 76)
    print("Checks")
    print("-" * 76)
    base = pr[(pr.window == CURRENT[0]) & (pr.shift_days == CURRENT[1])]
    print(f"  current-practice cell is 0.00 by construction: "
          f"{'yes' if abs(base.skill_vs_current).max() < 1e-6 else 'NO'}")
    print(f"  cells evaluated: {pr.groupby(['window', 'shift_days']).ngroups}")
    print(f"  bootstrap: {args.bootstrap} resamples over regions, "
          f"seed {BOOTSTRAP_SEED}")
    print("\nThis is a counterfactual on real submission dates, not a forecast")
    print("of what a faster system would achieve. Uniform delay reduction is an")
    print("idealisation; report it as a guide to which lever matters.")


# ──────────────────────────────────────────────────────────────────────

def fig_surface(pr):
    import matplotlib.pyplot as plt
    g = (pr.pivot_table(index="window", columns="shift_days",
                        values="skill_vs_current")
           .reindex([f"L{a}-{b}" for a, b in WINDOWS]))
    fig, ax = plt.subplots(figsize=(8.2, 5))
    lim = float(np.nanmax(np.abs(g.values)))
    im = ax.imshow(g.values, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(g.columns)))
    ax.set_xticklabels(g.columns)
    ax.set_yticks(range(len(g.index)))
    ax.set_yticklabels(g.index)
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            v = g.values[i, j]
            if v == v:
                ax.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(v) > lim * 0.55 else "#222222")
    ax.set_xlabel("Submission shifted earlier (days)")
    ax.set_ylabel("Collection window used at the forecast origin")
    ax.set_title("Neither lever works alone: skill vs current practice (%)")
    fig.colorbar(im, ax=ax, label="% MAE skill vs L8-16 at D=0")
    fig.tight_layout()
    return plotstyle.save(fig, "fig_policy_surface")


def fig_mechanism(pr):
    import matplotlib.pyplot as plt
    g = (pr.pivot_table(index="window", columns="shift_days",
                        values="no_data_frac")
           .reindex([f"L{a}-{b}" for a, b in WINDOWS]))
    fig, ax = plt.subplots(figsize=(8.2, 5))
    im = ax.imshow(100 * g.values, cmap="magma_r", vmin=0, vmax=100,
                   aspect="auto")
    ax.set_xticks(range(len(g.columns)))
    ax.set_xticklabels(g.columns)
    ax.set_yticks(range(len(g.index)))
    ax.set_yticklabels(g.index)
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            v = 100 * g.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9,
                    color="white" if v > 55 else "#222222")
    ax.set_xlabel("Submission shifted earlier (days)")
    ax.set_ylabel("Collection window used at the forecast origin")
    ax.set_title("Why: forecast origins with no sequence available (%)")
    fig.colorbar(im, ax=ax, label="% of origins with no data")
    fig.tight_layout()
    return plotstyle.save(fig, "fig_policy_mechanism")


def fig_frontier(pr):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for i, (lo, hi) in enumerate(WINDOWS):
        s = (pr[pr.window == f"L{lo}-{hi}"]
             .groupby("shift_days")["skill_vs_current"].mean())
        ax.plot(s.index, s.values, marker="o",
                color=plotstyle.PALETTE[i % len(plotstyle.PALETTE)],
                label=f"window L{lo}-{hi}")
    plotstyle.zero_line(ax, "current practice", x=SHIFTS[0])
    ax.set_xlabel("Submission shifted earlier (days)")
    ax.set_ylabel("MAE skill vs current practice (%)")
    ax.set_title("Faster submission only pays once the window is tight enough")
    ax.set_xticks(SHIFTS)
    ax.legend()
    plotstyle.finish(ax)
    fig.tight_layout()
    return plotstyle.save(fig, "fig_policy_frontier")


if __name__ == "__main__":
    main()