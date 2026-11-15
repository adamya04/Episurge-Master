#!/usr/bin/env python3
"""
EpiSurge - E4 / E5: model-free information flow from composition to incidence.

Why this exists. E3a asked whether genomic composition improves a *forecast*
of wILI and found no recoverable signal. That is a statement about one model
class (gradient-boosted trees on a particular feature set). A reviewer is
entitled to ask whether the signal is there but the model cannot see it.
E4 and E5 answer that without fitting a forecaster:

    E4  panel Granger causality  - does past composition reduce the
        variance of wILI left unexplained by wILI's own past?
    E5  transfer entropy         - the same question without any linearity
        assumption, measured in bits.

Both are corroboration, not a novelty claim. Their value is that they fail
or succeed for different reasons than E3a does, so agreement between them is
evidence and disagreement localises the problem.

What makes this reportable rather than a p-value fishing trip
-------------------------------------------------------------
Three design choices, each of which the project has already paid for
elsewhere by getting it wrong first:

1. SURROGATE NULL, NOT AN ANALYTIC p-VALUE. Panel residuals here are
   autocorrelated within a season and correlated across regions in the same
   week. An OLS F-test p-value under those conditions is anti-conservative
   and would manufacture significance. The null is instead built by
   circularly shifting the source series within each (region, season) by a
   random offset. That preserves the source's own autocorrelation and
   seasonal shape exactly, and destroys only its temporal alignment with the
   target - which is precisely the thing being tested. The reported p is the
   fraction of surrogates whose statistic exceeds the observed one.

2. A SWEPT POSITIVE CONTROL. E3a's null is only interpretable because its
   positive control returns +37.6% to +56.0%. The same rule applies here. A
   synthetic target is built that depends on lagged REAL composition with a
   known coefficient alpha,

       y_synth(t) = rho * y_synth(t-1) + alpha * z(x(t-1)) + noise

   carrying the real missingness pattern of x, and alpha is swept. This
   gives the smallest coupling each test could have detected on this panel
   geometry. Without it, "no information flow" is indistinguishable from
   "no power", and with 60 series of 33 weeks the second is a live
   possibility.

3. STATIONARITY HANDLED BY DIFFERENCING, LAGS NEVER CROSSING A SEASON.
   wILI inside a season is a single rising-then-falling excursion. Granger
   or transfer entropy on levels would report the shared seasonal ramp as
   causality in whichever direction the ramp happened to lead. Both series
   are first-differenced. Lags are built inside (region, season), so no lag
   reaches across the summer gap, where consecutive rows are 20+ weeks
   apart and a lag would be meaningless.

Standing rules this obeys
    Rule 2  as-of, not final: the as-of arm uses only composition published
            by the origin. The oracle arm is reported beside it as the
            literature-equivalent claim, and the gap between them is the
            deployability cost - the same contrast E3a and E3b report.
    Rule 5  phase stratification: as-of coverage is concentrated after the
            peak, so every arm is also reported pre-peak, where the as-of
            arm becomes very thin. The n is printed with every row so a
            thin cell cannot be mistaken for an informative one.

Stated limitations
    - The circular-shift surrogate wraps the end of each series onto its
      start, which injects one artificial discontinuity per series. On the
      pre-peak subset the series are only ~12 weeks long, so that
      discontinuity is a larger fraction of the series and the shift space
      is small (11 non-identity shifts per series). Differencing removes
      most of the trend that would make the wrap severe, but the pre-peak
      p-values are the least robust in this script and are reported as
      such.
    - Twelve arms are tested. Benjamini-Hochberg q-values and the
      Bonferroni threshold are computed and stored alongside the raw p, and
      no arm should be read as a finding on its uncorrected p alone.
    - The positive control measures power on rows where composition is
      PRESENT. That is the fair analogue - a real composition predictor can
      only act where composition exists - but it is not the power of a
      predictor obliged to produce a number on every row.

Run (from the repository root, inside the venv):

    python src/analysis/information_flow.py
    python src/analysis/information_flow.py --surrogates 1000
    python src/analysis/information_flow.py --alphas 0.2,0.4,0.6,0.8
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402

from analysis import plotstyle                                # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MASTER = ROOT / "data" / "processed" / "master_region_week.csv"
REPORTS = ROOT / "data" / "reports"
FIGURES = ROOT / "figures"

# The two source variants. oracle is collection-dated and leaks; as-of is
# publication-dated and deployable. Reporting both is what turns a single
# number into the deployability contrast.
SOURCES = {
    "oracle": "oracle_prop_h3n2",
    "as-of L8-16": "asof_L8_16_prop_h3n2",
}

LAGS = [1, 2]            # 33 weeks per season; p=3 costs 10% of each series
BINS = 3                 # terciles: 3^3 = 27 cells against ~1,900 triples
SURROGATES = 500
ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8]
RHO = 0.5                # AR(1) coefficient of the synthetic target
SEED = 20261115


# ──────────────────────────────────────────────────────────────────────
# panel preparation
# ──────────────────────────────────────────────────────────────────────

def load():
    if not MASTER.exists():
        sys.exit(f"Missing {MASTER} - run build_master_table.py first")
    df = pd.read_csv(MASTER)
    core = df[df["in_core_season"] == True].copy()            # noqa: E712

    core["season_week_idx"] = (
        core.groupby("season")["epiweek"].rank(method="dense").astype(int) - 1
    )

    # Peak-relative position, same definition as E1 and E2, so the
    # stratification is the project's and not a new one invented here.
    peak_idx = {}
    for season, grp in core.groupby("season"):
        nat = grp.groupby("epiweek")["wili_final"].mean()
        pe = int(nat.idxmax())
        peak_idx[season] = int(
            grp.loc[grp["epiweek"] == pe, "season_week_idx"].iloc[0])
    core["weeks_from_peak"] = (core["season_week_idx"]
                               - core["season"].map(peak_idx))
    core["peak_side"] = np.where(core["weeks_from_peak"] < 0,
                                 "pre-peak", "peak or after")

    return core.sort_values(
        ["season", "hhs_region", "season_week_idx"]).reset_index(drop=True)


def differenced(core, srccol):
    """
    First differences of target and source, with lags built strictly inside
    (region, season).

    The grouping is what keeps this honest. A plain .shift() on the sorted
    frame would make the first week of 2017-18 a lag of the last week of
    2016-17, which are 20+ weeks apart in real time.
    """
    d = core[["season", "hhs_region", "season_week_idx", "peak_side",
              "wili_final", srccol]].copy()
    d = d.rename(columns={srccol: "x_raw"})
    g = d.groupby(["season", "hhs_region"], sort=False)
    d["dy"] = g["wili_final"].diff()
    d["dx"] = g["x_raw"].diff()
    return d


def add_lags(d, col, p, prefix):
    g = d.groupby(["season", "hhs_region"], sort=False)[col]
    for k in range(1, p + 1):
        d[f"{prefix}{k}"] = g.shift(k)
    return d


# ──────────────────────────────────────────────────────────────────────
# E4 - panel Granger causality
# ──────────────────────────────────────────────────────────────────────

def granger_F(d, p, ycol="dy", xcol="dx"):
    """
    Pooled panel F statistic for H0: all source-lag coefficients are zero,
    with region fixed effects.

    Restricted    ycol(t) = a_region + sum_k b_k ycol(t-k)
    Unrestricted  ... + sum_k c_k xcol(t-k)

    Returned alongside n so a cell with too few complete rows is visible
    rather than silently reported as a statistic.
    """
    w = d.copy()
    w = add_lags(w, ycol, p, "yl")
    w = add_lags(w, xcol, p, "xl")
    ycols = [f"yl{k}" for k in range(1, p + 1)]
    xcols = [f"xl{k}" for k in range(1, p + 1)]

    w = w.dropna(subset=[ycol] + ycols + xcols)
    n = len(w)
    if n < 10 * (2 * p + 10):
        return np.nan, n

    y = w[ycol].to_numpy(float)
    fe = pd.get_dummies(w["hhs_region"], prefix="r", drop_first=False)
    Zr = np.column_stack([fe.to_numpy(float), w[ycols].to_numpy(float)])
    Zu = np.column_stack([Zr, w[xcols].to_numpy(float)])

    def rss(Z):
        beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
        r = y - Z @ beta
        return float(r @ r)

    rss_r, rss_u = rss(Zr), rss(Zu)
    q, ku = p, Zu.shape[1]
    if rss_u <= 0 or n - ku <= 0:
        return np.nan, n
    F = ((rss_r - rss_u) / q) / (rss_u / (n - ku))
    return float(F), n


# ──────────────────────────────────────────────────────────────────────
# E5 - transfer entropy
# ──────────────────────────────────────────────────────────────────────

def discretise(v, bins):
    """
    Equal-count binning on the pooled non-missing values. Equal-count rather
    than equal-width because the differenced series are heavy-tailed: equal
    width puts almost every observation in one bin and the entropy collapses
    to zero regardless of any coupling.
    """
    out = np.full(len(v), -1, dtype=int)
    ok = np.isfinite(v)
    if ok.sum() < bins * 5:
        return out
    qs = np.quantile(v[ok], np.linspace(0, 1, bins + 1)[1:-1])
    out[ok] = np.searchsorted(qs, v[ok], side="right")
    return out


def transfer_entropy(d, bins, ycol="dy", xcol="dx"):
    """
    TE(X -> Y) with one lag, in bits:

        TE = sum p(y1,y0,x0) * log2[ p(y1|y0,x0) / p(y1|y0) ]

    Finite samples bias TE upward - with 27 cells and ~1,900 triples the
    bias is a sizeable fraction of any real effect - so the reported
    quantity is always TE minus the surrogate mean, never raw TE.
    """
    w = d.copy()
    w = add_lags(w, ycol, 1, "yl")
    w = add_lags(w, xcol, 1, "xl")
    w = w.dropna(subset=[ycol, "yl1", "xl1"])
    n = len(w)
    if n < bins ** 3 * 5:
        return np.nan, n

    y1 = discretise(w[ycol].to_numpy(float), bins)
    y0 = discretise(w["yl1"].to_numpy(float), bins)
    x0 = discretise(w["xl1"].to_numpy(float), bins)
    ok = (y1 >= 0) & (y0 >= 0) & (x0 >= 0)
    y1, y0, x0 = y1[ok], y0[ok], x0[ok]
    n = len(y1)
    if n < bins ** 3 * 5:
        return np.nan, n

    joint = np.zeros((bins, bins, bins))
    np.add.at(joint, (y1, y0, x0), 1.0)
    joint /= joint.sum()

    p_y0x0 = joint.sum(axis=0)                  # (y0, x0)
    p_y1y0 = joint.sum(axis=2)                  # (y1, y0)
    p_y0 = joint.sum(axis=(0, 2))               # (y0,)

    te = 0.0
    for a in range(bins):
        for b in range(bins):
            for c in range(bins):
                pj = joint[a, b, c]
                if pj <= 0 or p_y0x0[b, c] <= 0 or p_y1y0[a, b] <= 0:
                    continue
                num = pj / p_y0x0[b, c]                  # p(y1|y0,x0)
                den = p_y1y0[a, b] / p_y0[b]             # p(y1|y0)
                if num <= 0 or den <= 0:
                    continue
                te += pj * np.log2(num / den)
    return float(te), n


# ──────────────────────────────────────────────────────────────────────
# surrogate null
# ──────────────────────────────────────────────────────────────────────

def circshift_source(d, rng, col="dx"):
    """
    Circularly shift the source inside each (region, season) by a random
    offset. The source keeps its own autocorrelation and seasonal shape;
    only its alignment with the target is destroyed.
    """
    # reset_index is load-bearing: groupby(...).indices returns POSITIONAL
    # offsets, and the pre-peak subset arrives with a non-contiguous index.
    # Without this the shift would scatter values to the wrong rows and the
    # surrogate null would be silently wrong rather than noisy.
    out = d.reset_index(drop=True).copy()
    vals = out[col].to_numpy(float).copy()
    for _, idx in out.groupby(["season", "hhs_region"], sort=False).indices.items():
        m = len(idx)
        if m < 3:
            continue
        k = int(rng.integers(1, m))
        vals[idx] = np.roll(vals[idx], k)
    out[col] = vals
    return out


def surrogate_p(d, stat_fn, observed, n_sur, rng, col="dx"):
    """
    Empirical one-sided p plus the surrogate mean, for bias subtraction.

    `col` must name the series the statistic treats as the SOURCE. Shifting
    the wrong column would leave the tested alignment intact and return a
    null centred on the observed value - a silently inert test rather than
    a failing one.
    """
    if not np.isfinite(observed):
        return np.nan, np.nan, np.nan
    draws = []
    for _ in range(n_sur):
        s, _ = stat_fn(circshift_source(d, rng, col=col))
        if np.isfinite(s):
            draws.append(s)
    if len(draws) < n_sur // 2:
        return np.nan, np.nan, np.nan
    draws = np.asarray(draws)
    p = float((1 + (draws >= observed).sum()) / (1 + len(draws)))
    return p, float(draws.mean()), float(draws.std())


# ──────────────────────────────────────────────────────────────────────
# positive control
# ──────────────────────────────────────────────────────────────────────

def synth_target(d, alpha, rng):
    """
    A target genuinely driven by lagged real composition:

        y(t) = RHO * y(t-1) + alpha * z(dx(t-1)) + sqrt(1-alpha^2) * noise

    Built within (region, season) so the recursion never crosses the summer
    gap, and carrying the real missingness of dx: where composition is
    missing the driving term contributes nothing, exactly as a real
    predictor could contribute nothing there. That is what makes the
    resulting power curve a statement about THIS panel rather than about an
    idealised complete one.
    """
    out = d.reset_index(drop=True).copy()   # see circshift_source
    dx = out["dx"].to_numpy(float)
    z = np.zeros_like(dx)
    ok = np.isfinite(dx)
    if ok.sum() > 2:
        z[ok] = (dx[ok] - dx[ok].mean()) / (dx[ok].std() + 1e-12)
    drive = np.where(ok, z, 0.0)

    ys = np.full(len(out), np.nan)
    noise = rng.standard_normal(len(out))
    scale = np.sqrt(max(1.0 - alpha ** 2, 0.0))
    for _, idx in out.groupby(["season", "hhs_region"], sort=False).indices.items():
        prev = 0.0
        prev_drive = 0.0
        for j in idx:
            # The drive enters LAGGED by one week. This is the whole point:
            # Granger causality asks whether dx(t-1) predicts y(t), so a
            # target built on a contemporaneous dx(t) would be undetectable
            # by construction unless dx happened to be autocorrelated - and
            # differenced composition is close to white noise. An earlier
            # version of this function used drive[j] and produced a positive
            # control that never fired, which would have been read as the
            # test lacking power rather than as the control being wrong.
            prev = RHO * prev + alpha * prev_drive + scale * noise[j]
            ys[j] = prev
            prev_drive = drive[j]
    out["dy_synth"] = ys
    return out


def synth_source(d, alpha, rng):
    """
    The source-side positive control, parameterised to match E8 exactly.

    Why a second control. The target-side control above injects coupling
    into a synthetic AR(1) target, so its alpha measures sensitivity to
    coupling into *that* process - not into real wILI, which carries
    seasonal structure a forecaster already exploits. E8's alpha is the
    correlation of a synthetic PREDICTOR with the standardised real target.
    Comparing the two numbers directly would be comparing different
    quantities and would overstate how much more sensitive E4 is.

    So this builds, against the REAL differenced target,

        x_synth(t) = alpha * z(dy(t+1)) + sqrt(1-alpha^2) * noise

    masked to composition's real availability. Then x_synth(t-1) carries
    alpha * z(dy(t)), so past source genuinely leads the target, and alpha
    is the same quantity E8 sweeps: correlation with the standardised
    target. The two experiments' floors are then on one scale.
    """
    out = d.reset_index(drop=True).copy()
    dy = out["dy"].to_numpy(float)

    # lead the target by one step inside (region, season)
    lead = out.groupby(["season", "hhs_region"], sort=False)["dy"].shift(-1)
    lead = lead.to_numpy(float)
    z = np.full(len(out), np.nan)
    ok = np.isfinite(lead)
    if ok.sum() > 2:
        z[ok] = (lead[ok] - lead[ok].mean()) / (lead[ok].std() + 1e-12)

    noise = rng.standard_normal(len(out))
    xs = alpha * z + np.sqrt(max(1.0 - alpha ** 2, 0.0)) * noise

    # carry composition's REAL missingness: the synthetic column is absent
    # exactly where real composition is absent, so the floor is a statement
    # about this panel's observed coverage and not an idealised one.
    xs = np.where(np.isfinite(out["dx"].to_numpy(float)), xs, np.nan)
    out["x_synth"] = xs
    _ = dy
    return out


# ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surrogates", type=int, default=SURROGATES)
    ap.add_argument("--alphas", default="")
    ap.add_argument("--bins", type=int, default=BINS)
    ap.add_argument("--power-surrogates", type=int, default=200,
                    help="surrogates per power cell; the sweep is the "
                         "expensive part, so it runs fewer than the main arms")
    ap.add_argument("--figure-only", action="store_true",
                    help="redraw the figure from the committed CSVs without "
                         "recomputing. A cosmetic fix to a title should not "
                         "cost a full surrogate run, and recomputing to "
                         "change a label risks the figure and the CSV "
                         "disagreeing if anything else moved meanwhile.")
    a = ap.parse_args()

    if a.figure_only:
        plotstyle.apply()
        R = pd.read_csv(REPORTS / "information_flow.csv")
        P = pd.read_csv(REPORTS / "information_flow_power.csv")
        make_figure(R, P)
        return

    alphas = ([float(x) for x in a.alphas.split(",")] if a.alphas else ALPHAS)
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    plotstyle.apply()

    core = load()
    print("EpiSurge E4/E5 - model-free information flow")
    print("=" * 74)
    print(f"core-season rows {len(core)} | "
          f"{core.hhs_region.nunique()} regions x {core.season.nunique()} seasons")
    print(f"surrogates {a.surrogates} | bins {a.bins} | lags {LAGS}")
    print("statistic is compared against a circular-shift surrogate null;\n"
          "transfer entropy is reported net of the surrogate mean.\n")

    rows = []
    for sname, scol in SOURCES.items():
        if scol not in core.columns:
            print(f"  [skip] {scol} absent")
            continue
        base = differenced(core, scol)
        for stratum in ("all", "pre-peak"):
            d = base if stratum == "all" else base[base.peak_side == "pre-peak"]
            d = d.reset_index(drop=True).copy()

            for p in LAGS:
                rng = np.random.default_rng(SEED)
                obs, n = granger_F(d, p)
                pv, smean, sstd = surrogate_p(
                    d, lambda z, p=p: granger_F(z, p), obs,
                    a.surrogates, rng)
                rows.append(dict(test="E4 Granger", source=sname,
                                 stratum=stratum, param=f"p={p}",
                                 statistic=obs, surrogate_mean=smean,
                                 surrogate_sd=sstd, net=np.nan,
                                 p_emp=pv, n=n))
                print(f"  E4 {sname:<12} {stratum:<13} p={p}  "
                      f"F={obs if np.isfinite(obs) else float('nan'):7.3f}  "
                      f"surr {smean if smean is not None else 0:6.3f}  "
                      f"p={pv:.3f}  n={n}")

            rng = np.random.default_rng(SEED + 1)
            obs, n = transfer_entropy(d, a.bins)
            pv, smean, sstd = surrogate_p(
                d, lambda z: transfer_entropy(z, a.bins), obs,
                a.surrogates, rng)
            net = obs - smean if np.isfinite(obs) and smean == smean else np.nan
            rows.append(dict(test="E5 transfer entropy", source=sname,
                             stratum=stratum, param=f"bins={a.bins}",
                             statistic=obs, surrogate_mean=smean,
                             surrogate_sd=sstd, net=net, p_emp=pv, n=n))
            print(f"  E5 {sname:<12} {stratum:<13} bins={a.bins}  "
                  f"TE={obs:.4f} bits  surr {smean:.4f}  "
                  f"net {net:+.4f}  p={pv:.3f}  n={n}")
        print()

    R = pd.DataFrame(rows)

    # ---- multiple testing ------------------------------------------
    # Twelve arms are tested (2 sources x 2 strata x (2 Granger lags + 1
    # TE)). At a nominal 0.05 the expected number of arms crossing by
    # chance alone is 0.6, so a single p just under 0.05 is exactly what
    # the null predicts. Reporting the uncorrected p alone would invite
    # the reader to treat that arm as a finding, so Benjamini-Hochberg
    # q-values and the Bonferroni threshold are carried in the CSV.
    m = int(R["p_emp"].notna().sum())
    R["bh_q"] = np.nan
    valid = R["p_emp"].notna()
    if m:
        ordr = R.loc[valid, "p_emp"].sort_values()
        q = (ordr.to_numpy() * m / np.arange(1, m + 1))
        q = np.minimum.accumulate(q[::-1])[::-1]          # enforce monotone
        R.loc[ordr.index, "bh_q"] = np.minimum(q, 1.0)
    R["bonferroni_alpha"] = 0.05 / m if m else np.nan

    out = REPORTS / "information_flow.csv"
    R.to_csv(out, index=False)
    print(f"wrote {out}\n")

    print(f"MULTIPLE TESTING over {m} arms")
    print(f"  Bonferroni threshold      {0.05 / m:.4f}" if m else "")
    raw = R[R.p_emp < 0.05]
    print(f"  arms with raw p < 0.05    {len(raw)} "
          f"(expected by chance at nominal 0.05: {0.05 * m:.1f})")
    bh = R[R.bh_q < 0.05]
    print(f"  arms surviving BH q < 0.05 {len(bh)}")
    for r in raw.itertuples():
        print(f"    {r.test} | {r.source} | {r.stratum} | {r.param}: "
              f"p={r.p_emp:.3f}  BH q={r.bh_q:.3f}  "
              f"{'SURVIVES' if r.bh_q < 0.05 else 'does NOT survive correction'}")
    print()

    # ---- positive control sweep -------------------------------------
    print("POSITIVE CONTROL: synthetic target driven by lagged real "
          "composition,\ncarrying its real missingness. alpha=0 is the "
          "false-positive arm.\n")
    prows = []
    base = differenced(core, SOURCES["as-of L8-16"])
    for alpha in alphas:
        rng = np.random.default_rng(SEED + 7)
        d = synth_target(base, alpha, rng)
        fG, nG = granger_F(d, 1, ycol="dy_synth")
        pG, mG, _ = surrogate_p(d, lambda z: granger_F(z, 1, ycol="dy_synth"),
                                fG, a.power_surrogates, rng)
        tE, nE = transfer_entropy(d, a.bins, ycol="dy_synth")
        pE, mE, _ = surrogate_p(d, lambda z: transfer_entropy(z, a.bins,
                                                              ycol="dy_synth"),
                                tE, a.power_surrogates, rng)
        # source-side arm: real target, synthetic predictor, alpha on E8's
        # scale so the two experiments' floors are directly comparable
        rng2 = np.random.default_rng(SEED + 11)
        ds = synth_source(base, alpha, rng2)
        fS, nS = granger_F(ds, 1, ycol="dy", xcol="x_synth")
        pS, _, _ = surrogate_p(
            ds, lambda z: granger_F(z, 1, ycol="dy", xcol="x_synth"),
            fS, a.power_surrogates, rng2, col="x_synth")

        prows.append(dict(alpha=alpha, granger_F=fG, granger_p=pG,
                          te=tE, te_net=tE - mE if tE == tE and mE == mE else np.nan,
                          te_p=pE, n_granger=nG, n_te=nE,
                          src_granger_F=fS, src_granger_p=pS, n_src=nS))
        # .2f, not .1f: the grid includes 0.05/0.10/0.15 and one decimal
        # printed three different couplings as the same "0.1".
        print(f"  alpha={alpha:.2f}  [target-side] Granger F={fG:8.3f} "
              f"p={pG:.3f}  TE net={tE - mE:+.4f} p={pE:.3f}   "
              f"| [source-side, E8 scale] F={fS:8.3f} p={pS:.3f}")

    P = pd.DataFrame(prows)
    pout = REPORTS / "information_flow_power.csv"
    P.to_csv(pout, index=False)
    print(f"\nwrote {pout}")

    det_t = P[(P.alpha > 0) & (P.granger_p < 0.05)]
    det_s = P[(P.alpha > 0) & (P.src_granger_p < 0.05)]
    fp = P[(P.alpha == 0)]
    print()
    for lab, det in (("target-side", det_t), ("source-side, E8 scale", det_s)):
        if len(det):
            amin = det.alpha.min()
            print(f"smallest coupling Granger resolved ({lab}): "
                  f"alpha={amin:.2f}  = {amin ** 2:.2%} of target variance")
        else:
            print(f"smallest coupling Granger resolved ({lab}): "
                  f"none at any alpha tested")
    print("\nThe source-side floor is the one comparable with E8's MDE: both\n"
          "sweep the correlation of a synthetic predictor with the\n"
          "standardised real target, under this panel's real coverage.")
    if len(fp):
        print(f"alpha=0 false-positive arm: Granger p={fp.granger_p.iloc[0]:.3f}, "
              f"TE p={fp.te_p.iloc[0]:.3f}  (both should be non-significant)")

    make_figure(R, P)


def make_figure(R, P):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))

    ax = axes[0]
    obs = R[R.test == "E4 Granger"]
    labels, vals, cols = [], [], []
    for i, r in enumerate(obs.itertuples()):
        labels.append(f"{r.source}\n{r.stratum}, {r.param}")
        vals.append(r.p_emp)
        cols.append(plotstyle.PALETTE[i % len(plotstyle.PALETTE)])
    ax.bar(range(len(vals)), vals, color=cols)
    ax.axhline(0.05, color=plotstyle.RULE, ls="--", lw=1)
    ax.text(len(vals) - 0.5, 0.065, "p = 0.05", ha="right", fontsize=8,
            color=plotstyle.RULE)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6.5, rotation=30, ha="right")
    ax.set_ylabel("surrogate p-value")
    # Computed, never asserted. An earlier hard-coded title read "no arm
    # reaches p = 0.05" while one arm sat at p = 0.040, which is exactly
    # the kind of false claim a figure carries into a talk unchallenged.
    n_raw = int((obs["p_emp"] < 0.05).sum())
    n_bh = int((obs["bh_q"] < 0.05).sum()) if "bh_q" in obs else 0
    if n_raw == 0:
        t = "E4 panel Granger: no arm reaches p = 0.05"
    else:
        # "arms" agrees with the total, the verb with the count:
        # "1 of 8 arms crosses", not "1 of 8 arm crosses".
        verb = "crosses" if n_raw == 1 else "cross"
        surv = "survives" if n_bh == 1 else "survive"
        t = (f"E4 panel Granger: {n_raw} of {len(obs)} arms {verb} p = 0.05 "
             f"raw,\n{n_bh} {surv} multiple-testing correction")
    ax.set_title(t, fontsize=10)
    plotstyle.finish(ax)

    ax = axes[1]
    ax.plot(P.alpha, P.granger_p, "o-", color=plotstyle.PALETTE[0],
            label="Granger p (target-side)")
    ax.plot(P.alpha, P.te_p, "s--", color=plotstyle.PALETTE[1],
            label="transfer entropy p (target-side)")
    # The source-side curve is the one quoted against E8's MDE, so it has
    # to be visible rather than only described in the text.
    if "src_granger_p" in P.columns:
        ax.plot(P.alpha, P.src_granger_p, "^-", color=plotstyle.PALETTE[2],
                label="Granger p (source-side, E8 scale)")
    ax.axhline(0.05, color=plotstyle.RULE, ls="--", lw=1)
    ax.set_xlabel("injected coupling alpha")
    ax.set_ylabel("surrogate p-value")
    ax.set_title("Positive control: the tests do fire when\ncoupling is "
                 "present, which is what licenses the null", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    plotstyle.finish(ax)

    fig.tight_layout()
    plotstyle.save(fig, "fig_information_flow")


if __name__ == "__main__":
    main()
