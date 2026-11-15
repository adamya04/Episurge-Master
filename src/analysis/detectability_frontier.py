#!/usr/bin/env python3
"""
EpiSurge - Phase 3 / E8: the detectability frontier.

WHY THIS EXPERIMENT EXISTS
--------------------------
E3a is a null: genomic composition does not improve incidence forecasting at
any horizon, oracle or as-of. The null is backed by a positive control that
fires (+37.6% to +56.0%), a pure-noise control, a missingness control, and an
analytic floor taken from bootstrap CI half-widths (5.5-10.8%).

That package establishes the harness is not broken. It does NOT establish
that the design could have detected a REALISTIC genomic effect, and the
distinction matters because the positive control is enormous:

    _positive = y_target + N(0, 0.5),  and sd(y_target) ~ 2.04
    => correlation with the target 0.971, i.e. 94.3-94.5% of its variance.

No influenza biomarker explains 94% of regional wILI variance. So "the
control fires" answers a question nobody asked. The question a reviewer will
actually ask is:

    HOW STRONG WOULD A GENOMIC PREDICTOR HAVE TO BE BEFORE THIS DESIGN
    COULD SEE IT - AND WOULD BETTER PUBLICATION TIMELINESS HELP?

E3c answers the policy half of that for composition forecasting. Nothing in
the project answers it for the incidence null, which is the frozen research
question itself. This experiment does, by simulation rather than by a
CI-width proxy.

METHOD
------
A synthetic predictor of KNOWN strength alpha is injected:

    synth = alpha * z(y_target) + sqrt(1 - alpha^2) * N(0, 1)

so alpha is its correlation with the standardised target and alpha^2 is the
share of target variance it explains - a scale a reader can price against a
real biomarker. The synthetic column then carries the REAL as-of missingness
pattern, because availability is the project's whole subject: a predictor
that is absent at the forecast origin cannot help however strong it is.

Availability is swept by revealing a fraction phi of the cells that are
missing at the origin, from the true as-of pattern (phi=0) up to complete
coverage (phi=1). That turns E1's coverage gap into a detectability
statement: MDE(phi), the minimum detectable effect as timeliness improves.

Detection uses the project's standing rule - sign-consistent skill across
every pre-COVID fold. Each (alpha, phi, horizon) cell is run over several
independent noise draws, because a single draw is not stable enough to read
an MDE from: in testing, one replicate detected alpha=0.4 where the next
did not.

THE RULE IS CALIBRATED, NOT ASSUMED
-----------------------------------
Sign-consistency across 4 folds would fire 1/16 = 6.25% of the time on
coin flips, and across a 90-cell grid that is ~5 spurious detections. So
alpha=0 is run explicitly as a null arm. Measured: 0 detections in 32 null
cells, with mean skill slightly NEGATIVE (-0.02% to -1.56%) because an
uninformative extra column costs a gradient booster a little. The rule is
therefore conservative rather than 6.25%, and the frontier is not inflated
by false positives.

WHAT IT FOUND
-------------
Mean skill rises monotonically with alpha. Reliable detection (every
replicate) arrives only at alpha=0.7, and mean skill crosses E3a's own 5.5%
floor between alpha=0.6 and alpha=0.7 - so a genomic predictor would need to
explain roughly 40% of regional wILI variance to be resolvable by this
design at all. That is the floor set by the panel: 7 seasons x 10 regions.

Pooled over a whole season, availability then looks irrelevant - phi moves
skill by 1-2 points against 7-11 for the alpha sweep. THAT POOLING IS AN
ARTIFACT AND THE FIRST READING OF IT WAS WRONG. The probe is 89.6% present
after the peak and only 34.8% before it, so a pooled average is dominated by
post-peak rows where coverage is already near-saturated and phi has nothing
left to buy.

Restricted to pre-peak origins - the regime where a forecast has any value -
availability binds hard. At h=4 under the real as-of pattern NOTHING is
detected at any alpha up to 0.7, while complete coverage roughly triples mean
skill (alpha=0.7: 3.33% -> 10.54%; alpha=0.6: 0.15% -> 4.72%) and brings
detection back at alpha=0.6-0.7.

So both constraints bind, in different places. Panel size sets the overall
floor of ~40-49% of variance explained, and publication latency raises that
bar further precisely before the peak. E1's 34.8% pre-peak coverage, which
this experiment reproduces independently, is therefore not merely a
descriptive gap: it is the mechanism that puts a detectable genomic effect
out of reach exactly when one would be useful.

This does not weaken E3b or E3c. There the delay cost is large, sign
consistent and has CIs excluding zero, because the target is composition,
which composition actually predicts. It bounds E3a specifically, and it
converts that null from "genomics carries no signal" into the defensible and
more useful "no effect large enough for this design to resolve, and here is
the threshold".

Inputs:
    data/processed/master_region_week.csv

Outputs:
    data/reports/detectability_frontier.csv
    data/reports/detectability_null.csv      (with --null)
    figures/fig_detectability_frontier.png

Run (from the repository root, inside the venv):
    python src/analysis/detectability_frontier.py
    python src/analysis/detectability_frontier.py --null
    python src/analysis/detectability_frontier.py --stratum pre-peak
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

from baselines import (                                           # noqa: E402
    FOLDS, B2_FEATURES, B2_PARAMS, build_targets, usable, load,
)
import plotstyle                                                  # noqa: E402
from matplotlib import pyplot as plt                              # noqa: E402

REPORTS = Path("data/reports")
FIGURES = Path("figures")

COVID_FOLD_TEST = "2022-23"
SEED = 20261115

# alpha is correlation with the standardised target; alpha^2 is variance
# explained. The low end is coarse because testing showed nothing below
# 0.3 is detectable at any availability.
ALPHAS = [0.3, 0.4, 0.5, 0.6, 0.7]
PHIS = [0.0, 0.5, 1.0]
HS = [1, 4]
REPS = 3

# the as-of column whose missingness pattern the synthetic predictor inherits
PROBE = "asof_L8_16_prop_h3n2"

# E3a's analytic floor, for reading an MDE off the mean-skill curve
E3A_FLOOR_PCT = 5.5


def fit_mae(train, test, cols):
    m = HistGradientBoostingRegressor(random_state=0, **B2_PARAMS)
    m.fit(train[cols].values, train["y_target"].values)
    return float(np.abs(test["y_target"].values
                        - m.predict(test[cols].values)).mean())


def run(core, alphas, phis, hs, reps, stratum, probe):
    pre = [f for f, _, te in FOLDS if te != COVID_FOLD_TEST]
    rng = np.random.default_rng(SEED)
    rows = []

    def ev(df):
        return df if stratum == "all" else df[df["peak_side"] == "pre-peak"]

    for h in hs:
        d = build_targets(core, h)
        obs = d[probe].notna().values
        ystd = (d["y_target"] - d["y_target"].mean()) / d["y_target"].std()

        base = {}
        for fold, tr, te in FOLDS:
            tr_, te_ = usable(d[d.season.isin(tr)]), ev(usable(d[d.season == te]))
            if tr_.empty or len(te_) < 5:
                continue
            base[fold] = fit_mae(tr_, te_, list(B2_FEATURES))

        for alpha in alphas:
            for rep in range(reps):
                z = rng.standard_normal(len(d))
                sig = alpha * ystd.values + np.sqrt(max(0.0, 1 - alpha ** 2)) * z
                for phi in phis:
                    reveal = obs.copy()
                    hid = np.where(~obs)[0]
                    if phi > 0 and len(hid):
                        pick = rng.choice(hid, int(round(phi * len(hid))),
                                          replace=False)
                        reveal[pick] = True
                    d["_synth"] = np.where(reveal, sig, np.nan)

                    # coverage must be measured on the rows actually
                    # EVALUATED, not on the whole panel. Pooled over a
                    # season the probe is 64.4% present, but that is
                    # 89.6% after the peak against 34.8% before it, and
                    # reporting the pooled figure for a pre-peak run
                    # understates by a factor of two exactly where
                    # availability binds hardest.
                    # ...and over rows that are actually EVALUABLE. A row
                    # with no horizon-h target has a NaN target, so the
                    # injected signal is NaN there too; such rows are
                    # dropped by usable() before any fit, and leaving them
                    # in the denominator reported 97% where coverage was
                    # in fact complete.
                    cov = float(ev(usable(d))["_synth"].notna().mean())

                    sk = {}
                    for fold, tr, te in FOLDS:
                        tr_ = usable(d[d.season.isin(tr)])
                        te_ = ev(usable(d[d.season == te]))
                        if tr_.empty or len(te_) < 5 or fold not in base:
                            continue
                        e = fit_mae(tr_, te_, list(B2_FEATURES) + ["_synth"])
                        sk[fold] = 100 * (base[fold] - e) / base[fold]

                    v = [sk[f] for f in pre if f in sk]
                    rows.append({
                        "stratum": stratum, "horizon": h, "alpha": alpha,
                        "rep": rep, "phi": phi,
                        "coverage": round(cov, 4),
                        "mean_skill": round(float(np.mean(v)), 3) if v else np.nan,
                        "folds_pos": sum(1 for x in v if x > 0),
                        "n_folds": len(v),
                        "detected": bool(v) and all(x > 0 for x in v),
                    })
                    print(f"  h={h} alpha={alpha:.2f} r{rep} phi={phi:.2f} "
                          f"cov={cov:.0%} "
                          f"skill={np.mean(v):+6.2f}% "
                          f"{sum(1 for x in v if x > 0)}/{len(v)} "
                          f"{'DETECTED' if rows[-1]['detected'] else '-'}",
                          flush=True)
    return pd.DataFrame(rows)


def mde_from_curve(R, floor):
    """
    MDE read off the MEAN-SKILL curve rather than the binary rule.

    The per-replicate detection flag is too noisy to invert directly - with
    three draws it is non-monotone in alpha. Mean skill is monotone, so the
    threshold alpha is found by linear interpolation to the point where mean
    skill crosses E3a's own analytic floor. Reported as alpha and as
    alpha^2, the share of target variance such a predictor would explain.
    """
    out = []
    g = R.groupby(["horizon", "phi", "alpha"])["mean_skill"].mean().reset_index()
    for (h, phi), sub in g.groupby(["horizon", "phi"]):
        sub = sub.sort_values("alpha")
        a, s = sub["alpha"].values, sub["mean_skill"].values
        hit = np.nan
        for i in range(1, len(a)):
            if s[i - 1] < floor <= s[i]:
                hit = a[i - 1] + (floor - s[i - 1]) * (a[i] - a[i - 1]) / (s[i] - s[i - 1])
                break
        out.append({"horizon": h, "phi": phi,
                    "mde_alpha": round(float(hit), 3) if hit == hit else np.nan,
                    "mde_var_explained": round(float(hit ** 2), 3) if hit == hit else np.nan})
    return pd.DataFrame(out)


def fig_frontier(R, stratum):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for i, phi in enumerate(sorted(R.phi.unique())):
        s = (R[R.phi == phi].groupby("alpha")["mean_skill"].mean())
        cov = R[R.phi == phi]["coverage"].iloc[0]
        ax.plot(s.index, s.values, marker="o", color=plotstyle.PALETTE[i],
                label=f"availability {cov:.0%} of region-weeks")
    ax.axhline(E3A_FLOOR_PCT, ls=":", color="0.35", lw=1.2)
    ax.text(ax.get_xlim()[0], E3A_FLOOR_PCT, " E3a detectable floor (5.5%)",
            va="bottom", ha="left", fontsize=9, color="0.35")
    plotstyle.zero_line(ax, "case history alone", x=min(R.alpha))
    ax.set_xlabel(r"injected predictor strength $\alpha$"
                  "  (correlation with the target)")
    ax.set_ylabel("MAE improvement over case history (%)")
    sub = "pooled over the season" if stratum == "all" else "pre-peak origins only"
    ax.set_title("A genomic predictor must explain ~40% of wILI variance\n"
                 f"before this design could resolve it  ({sub})")
    ax.legend()
    plotstyle.finish(ax)
    fig.tight_layout()
    suffix = "" if stratum == "all" else f"_{stratum.replace('-', '')}"
    return plotstyle.save(fig, f"fig_detectability_frontier{suffix}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stratum", default="all", choices=["all", "pre-peak"])
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--probe", default=PROBE)
    ap.add_argument("--null", action="store_true",
                    help="run the alpha=0 calibration arm instead")
    ap.add_argument("--alphas", default="")
    ap.add_argument("--phis", default="")
    ap.add_argument("--hs", default="")
    a = ap.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    plotstyle.apply()

    core = load()
    if a.probe not in core.columns:
        cand = [c for c in core.columns
                if c.startswith("asof_") and "prop_h3n2" in c]
        sys.exit(f"probe {a.probe} absent; candidates: {cand}")

    alphas = [0.0] if a.null else (
        [float(x) for x in a.alphas.split(",")] if a.alphas else ALPHAS)
    phis = [float(x) for x in a.phis.split(",")] if a.phis else PHIS
    hs = [int(x) for x in a.hs.split(",")] if a.hs else HS
    reps = max(a.reps, 8) if a.null else a.reps

    print("EpiSurge E8 - detectability frontier")
    print("=" * 74)
    print(f"stratum {a.stratum} | probe {a.probe}")
    print(f"real as-of coverage {core[a.probe].notna().mean():.1%} of region-weeks")
    print(f"alphas {alphas} | phis {phis} | horizons {hs} | reps {reps}")
    if a.null:
        print("NULL ARM: alpha=0. Any detection here is a false positive.")

    R = run(core, alphas, phis, hs, reps, a.stratum, a.probe)
    name = "detectability_null" if a.null else "detectability_frontier"
    if a.stratum != "all":
        name += f"_{a.stratum.replace('-', '')}"
    out = REPORTS / f"{name}.csv"
    R.to_csv(out, index=False)
    print(f"\nwrote {out}")

    print("\nDetection rate across replicates:")
    print(R.groupby(["horizon", "phi", "alpha"])["detected"].mean()
          .unstack("alpha").round(2).to_string())
    print("\nMean skill % across replicates:")
    print(R.groupby(["horizon", "phi", "alpha"])["mean_skill"].mean()
          .unstack("alpha").round(2).to_string())

    if a.null:
        n = int(R.detected.sum())
        print(f"\n  FALSE POSITIVES: {n} of {len(R)} null cells "
              f"({n / len(R):.1%}). Sign-consistency across "
              f"{R.n_folds.max()} folds would fire 6.25% on coin flips; an "
              f"uninformative column costs the booster a little, so the rule "
              f"runs conservative.")
        return

    print(f"\nMDE interpolated to E3a's {E3A_FLOOR_PCT}% floor:")
    print(mde_from_curve(R, E3A_FLOOR_PCT).to_string(index=False))
    print("\n  alpha is correlation with the standardised target, so"
          "\n  mde_var_explained is the share of target variance a genomic"
          "\n  predictor would need to carry before this design could see it.")
    print(f"\n  wrote {fig_frontier(R, a.stratum)}")


if __name__ == "__main__":
    main()
