"""
Every number the app states, derived from the committed artifacts.

The derivations follow src/tools/build_results_document.py and the standing
rules in DECISIONS.md:

  - Per-fold reporting. A contrast is reported as its mean only when every
    pre-COVID fold agrees in sign; otherwise it is FLIP.
  - Pre-COVID folds. Fold 4 crosses the COVID gap and reverses sign in
    several experiments, so headline tables use the four folds whose test
    season is not 2022-23, exactly as the results document does.

No figure is typed in here. If an artifact is regenerated, every page
updates with it.
"""

import numpy as np
import pandas as pd
import streamlit as st

from core import data as D


def sign_verdict(values):
    """(mean, consistent) under the per-fold rule."""
    v = [x for x in values if x == x]
    if not v:
        return np.nan, False
    ok = all(x > 0 for x in v) or all(x < 0 for x in v)
    return float(np.mean(v)), ok


def fmt_verdict(values, digits=1):
    m, ok = sign_verdict(values)
    if m != m:
        return "–"
    return f"{m:+.{digits}f}" if ok else "FLIP"


def verdict_matrix(df, key, horizons=None, value="skill_pct"):
    """Rows of `key`, columns horizons, cells (mean, consistent)."""
    horizons = horizons or D.HORIZONS
    out = {}
    for k in df[key].unique():
        sub = df[df[key] == k]
        out[k] = {h: sign_verdict(sub[sub.horizon == h][value].tolist())
                  for h in horizons}
    return out


def consistent_range(cells):
    """(min, max) of the sign-consistent means in a {h: (mean, ok)} row."""
    v = [m for m, ok in cells.values() if ok]
    return (min(v), max(v)) if v else (np.nan, np.nan)


# ── project record, parsed from the repository's own logs ────────────

@st.cache_data(show_spinner=False)
def run_timeline():
    """(experiment id, title, first run date) from RESULTS.md headings."""
    import re
    from core.paths import ROOT
    text = (ROOT / "RESULTS.md").read_text(encoding="utf-8")
    out, cur = [], None
    for line in text.splitlines():
        m = re.match(r"^## (E[\d]+[a-z]?(?: / E\d+)?) — (.+)$", line)
        if m:
            cur = [m.group(1).replace("E3", "E3a") if m.group(1) == "E3"
                   else m.group(1), m.group(2).strip()]
            continue
        d = re.match(r"^\*\*Run:\*\* (\d{4}-\d{2}-\d{2})", line)
        if d and cur:
            out.append((cur[0], cur[1], d.group(1)))
            cur = None
    return out


@st.cache_data(show_spinner=False)
def decisions(status=None):
    """Rows of DECISIONS.md's decision table: (date, status, decision, why)."""
    import re
    from core.paths import ROOT
    rows = []
    for line in (ROOT / "DECISIONS.md").read_text(encoding="utf-8").splitlines():
        if not re.match(r"^\| \d{4}-\d{2}-\d{2} \|", line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        st_ = re.sub(r"\*", "", cells[1]).split(" ")[0].upper()
        rows.append(dict(date=cells[0], status=st_, decision=cells[2],
                         rationale=" | ".join(cells[3:])))
    if status:
        rows = [r for r in rows if r["status"] == status]
    return rows


# ── E1 ───────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def e1():
    ps = D.report("coverage_by_peak_side.csv")
    rw = ps[ps.season == "ALL (row-weighted)"]
    cov = rw.pivot_table(index="variant", columns="peak_side",
                         values="pct_covered")
    ds = D.report("delay_summary.csv").set_index("season")
    return dict(
        coverage=cov,
        asof_pre=float(cov.loc["asof_L8_16", "pre-peak"]),
        asof_post=float(cov.loc["asof_L8_16", "peak or after"]),
        oracle_pre=float(cov.loc["oracle", "pre-peak"]),
        median_delay_days=float(ds.loc["ALL", "median"]),
        median_delay_weeks=float(ds.loc["ALL", "median_weeks"]),
        pct_over_12wk=float(ds.loc["ALL", "pct_over_12wk"]),
        n_dated=int(ds.loc["ALL", "n_sequences"]),
        delay_summary=ds,
    )


# ── E2 ───────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def e2():
    r = D.report("baseline_results.csv")
    r = r[r.stratum == "all"]
    # mean over seeds within a fold (B0/B1 are deterministic; B2 has 5 seeds)
    f = r.groupby(["fold", "horizon", "model"])["mae"].mean().unstack("model")
    f["B1"] = f["B1"]
    skill = pd.DataFrame({
        "B1": 100 * (f["B0"] - f["B1"]) / f["B0"],
        "B2": 100 * (f["B0"] - f["B2"]) / f["B0"],
    }).reset_index()
    rows = {}
    for m in ("B1", "B2"):
        rows[m] = {}
        for h in D.HORIZONS:
            v = skill[skill.horizon == h][m]
            rows[m][h] = dict(mean=float(v.mean()), pos=int((v > 0).sum()),
                              n=int(v.notna().sum()))
    s = D.report("baseline_summary.csv")
    s = s[s.stratum == "all"].pivot_table(index="horizon", columns="model",
                                          values="mae")
    return dict(per_fold=skill, folds=rows, pooled_mae=s)


# ── E3a ──────────────────────────────────────────────────────────────

E3A_CONTRASTS = ["A6-A4", "A4-A0", "A6-A0", "A5-A0", "A2-A0", "A1-A0",
                 "A4-A1", "A2s-A2"]


@st.cache_data(show_spinner=False)
def e3a():
    con = D.report("ablation_contrasts.csv")
    ca = con[(con.stratum == "all") & (con.fold.isin(D.PRE_FOLDS))]
    labels = dict(ca[["contrast", "label"]].drop_duplicates().values)
    mat = verdict_matrix(ca, "contrast")
    n_consistent_pos = sum(1 for c in mat.values()
                           for m, ok in c.values() if ok and m > 0)
    # detectable floor: median bootstrap half-width of the primary contrast
    # over the A0 MAE, pre-peak, as the document computes it
    ab = D.report("ablation_results.csv")
    pc = con[(con.contrast == "A6-A4") & (con.stratum == "pre-peak")].dropna(
        subset=["ci_lo"])
    a0 = ab[(ab.arm == "A0") & (ab.stratum == "pre-peak")].groupby(
        "horizon")["mae"].mean()
    floor = {}
    for h in sorted(pc.horizon.unique()):
        half = (pc[pc.horizon == h].ci_hi - pc[pc.horizon == h].ci_lo).median() / 2
        floor[int(h)] = float(100 * half / a0[h])
    arms = ab[ab.stratum == "all"].pivot_table(index="horizon", columns="arm",
                                               values="mae")
    return dict(matrix=mat, labels=labels, n_consistent_pos=n_consistent_pos,
                floor=floor, arms=arms, contrasts=con)


# ── E3b ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def e3b():
    c = D.report("dominance_delay_cost.csv")
    ca = c[(c.stratum == "all") & (c.fold.isin(D.PRE_FOLDS))]
    labels = dict(ca[["contrast", "label"]].drop_duplicates().values)
    mat = verdict_matrix(ca, "contrast")
    primary = "nat ORACLE vs nat as-of"
    lo, hi = consistent_range(mat[primary])
    pc = c[(c.contrast == primary) & (c.stratum == "all")]
    n_sig = int(((pc.ci_lo > 0) | (pc.ci_hi < 0)).sum())
    dd = D.report("dominance_decomposition.csv")
    dd = dd[dd.fold.isin(D.PRE_FOLDS)].copy()
    dd["gap"] = 100 * (dd.acc_oracle - dd.acc_asof)
    dec = dd.pivot_table(index="scope", columns="horizon", values="gap")
    res = D.report("dominance_results.csv")
    return dict(matrix=mat, labels=labels, primary=primary, lo=lo, hi=hi,
                n_sig=n_sig, n_cells=len(pc), decomposition=dec,
                results=res, costs=c)


# ── E3c ──────────────────────────────────────────────────────────────

WINDOW_ORDER = ["L0-2", "L0-4", "L2-6", "L4-12", "L8-16"]


@st.cache_data(show_spinner=False)
def e3c():
    p = D.report("policy_surface.csv")
    pp = p[p.fold.isin(D.PRE_FOLDS)]
    sk = pp.groupby(["window", "shift_days", "horizon"])["skill_vs_current"]
    cells = {}
    for (w, s, h), g in sk:
        cells[(w, int(s), int(h))] = sign_verdict(g.tolist())
    miss = pp.groupby(["window", "shift_days"])["no_data_frac"].mean() * 100
    shifts = sorted(p.shift_days.unique())
    return dict(cells=cells, missing=miss, shifts=[int(s) for s in shifts],
                windows=[w for w in WINDOW_ORDER if w in set(p.window)],
                raw=p)


def e3c_row(res, window, shift):
    return {h: res["cells"].get((window, shift, h), (np.nan, False))
            for h in D.HORIZONS}


# ── E4 / E5 ──────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def e45():
    f = D.report("information_flow.csv")
    p = D.report("information_flow_power.csv")
    det = p[(p.alpha > 0) & (p.src_granger_p < 0.05)]
    floor = float(det.alpha.min()) if len(det) else np.nan
    fp = p[p.alpha == 0]
    return dict(arms=f, power=p, n_arms=len(f),
                n_raw=int((f.p_emp < 0.05).sum()),
                n_bh=int((f.bh_q < 0.05).sum()),
                expected=0.05 * len(f),
                bh_min=float(f.bh_q.min()),
                bonferroni=float(f.bonferroni_alpha.iloc[0]),
                floor_alpha=floor,
                floor_var=floor ** 2 * 100 if floor == floor else np.nan,
                fp_p=float(fp.src_granger_p.iloc[0]) if len(fp) else np.nan)


# ── E6 ───────────────────────────────────────────────────────────────

GNN_ARMS = ["none", "corr", "full", "random", "climate", "mass"]
GNN_PRETTY = {"none": "No graph", "corr": "Correlation", "full": "Fully connected",
              "random": "Random (G5 control)", "climate": "Climate similarity",
              "mass": "Population mass", "B0 persistence": "B0 persistence",
              "B2 boosting": "B2 boosting"}


@st.cache_data(show_spinner=False)
def e6():
    g = D.report("graph_gnn_ablation.csv")
    gp = g[g.fold.isin(D.PRE_FOLDS)]
    f = gp.groupby(["fold", "arm", "horizon"])["mae"].mean().reset_index()
    w = f.pivot_table(index=["fold", "horizon"], columns="arm", values="mae")

    # every ordered pair, precomputed: a closure cannot be cached
    names = list(w.columns)
    contrasts = {}
    for a in names:
        for b in names:
            if a == b:
                continue
            per = ((w[b] - w[a]) / w[b] * 100)
            contrasts[(a, b)] = {h: sign_verdict(per.xs(h, level="horizon").tolist())
                                 for h in D.HORIZONS}
    mean_mae = gp.pivot_table(index="arm", columns="horizon", values="mae")
    stab = D.report("graph_stability.csv")
    return dict(raw=g, mean_mae=mean_mae, contrasts=contrasts, stability=stab,
                arms=[a for a in GNN_ARMS if a in set(g.arm)])


def e6_contrast(res, a, b):
    """Skill of arm a against arm b, per horizon, (mean, consistent)."""
    return res["contrasts"].get((a, b), {h: (np.nan, False) for h in D.HORIZONS})


# ── E8 ───────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def e8():
    from detectability_frontier import mde_from_curve, E3A_FLOOR_PCT
    fr = D.report("detectability_frontier.csv")
    pp = D.report("detectability_frontier_prepeak.csv") \
        if D.report_exists("detectability_frontier_prepeak.csv") else None
    nul = D.report("detectability_null.csv")
    mde = mde_from_curve(fr, E3A_FLOOR_PCT)
    mpp = mde_from_curve(pp, E3A_FLOOR_PCT) if pp is not None else None
    ok = mde.mde_alpha.dropna()
    return dict(frontier=fr, prepeak=pp, null=nul, mde=mde, mde_pre=mpp,
                floor_pct=E3A_FLOOR_PCT,
                alpha_lo=float(ok.min()), alpha_hi=float(ok.max()),
                var_lo=float(ok.min() ** 2 * 100),
                var_hi=float(ok.max() ** 2 * 100),
                n_null=len(nul), n_fp=int(nul.detected.sum()),
                pre_never=int(mpp.mde_alpha.isna().sum()) if mpp is not None else 0,
                reps=int(fr.rep.nunique()))
