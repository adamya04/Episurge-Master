import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import data as D
from core import results as R
from core import theme as T
from core.charts import matrix_fig

T.hero(
    "Experiment explorer",
    "Every claim, <span class='es-grad'>fold by fold</span>",
    "The project's standing rule: a result counts only if every pre-COVID fold "
    "agrees on its sign. A horizon whose sign flips between folds is marked "
    "<b>FLIP</b> and supports no claim — however good its average looks.",
)

e2, e3a, e3b, e3c, e6 = R.e2(), R.e3a(), R.e3b(), R.e3c(), R.e6()
H = D.HORIZONS


tabs = st.tabs(["Evidence matrix", "E2 · Baselines", "E3a · Genomic ablation",
                "E3b · Subtype dominance"])

# ── evidence matrix ──────────────────────────────────────────────────
with tabs[0]:
    T.section("All experiments", "One glance at what is supported",
              "Green and red cells are sign-consistent across the four pre-COVID "
              "folds; their value is the mean skill (%). Grey FLIP cells carry no "
              "claim. Skill is always the first model's improvement over the second.")
    b2 = e2["per_fold"][e2["per_fold"].fold.isin(D.PRE_FOLDS)]
    rows = [
        ("E2 · B2 boosting vs B0 persistence",
         {h: R.sign_verdict(b2[b2.horizon == h].B2.tolist()) for h in H}),
        ("E2 · B1 climatology vs B0 persistence",
         {h: R.sign_verdict(b2[b2.horizon == h].B1.tolist()) for h in H}),
    ]
    for c in ["A6-A4", "A4-A0", "A6-A0", "A1-A0"]:
        if c in e3a["matrix"]:
            rows.append((f"E3a · {c} {e3a['labels'][c]}", e3a["matrix"][c]))
    for c in ["nat ORACLE vs nat as-of", "reg ORACLE vs reg as-of",
              "nat as-of vs train-majority"]:
        if c in e3b["matrix"]:
            rows.append((f"E3b · {e3b['labels'][c]}", e3b["matrix"][c]))
    for w, s, lab in [("L4-12", 14, "L4-12 + 14 d faster"), ("L8-16", 84, "faster deposit alone (84 d)"),
                      ("L0-4", 0, "tighter window alone")]:
        rows.append((f"E3c · {lab}", R.e3c_row(e3c, w, s)))
    for a, b, lab in [("none", "B0 persistence", "GNN vs persistence"),
                      ("none", "B2 boosting", "GNN vs boosting"),
                      ("corr", "random", "correlation graph vs random"),
                      ("climate", "random", "climate graph vs random")]:
        rows.append((f"E6 · {lab}", R.e6_contrast(e6, a, b)))
    st.plotly_chart(matrix_fig(rows), width="stretch", config={"displayModeBar": False})
    allc = [c for _, r in rows for c in r.values() if c[0] == c[0]]
    T.cards([
        ("Cells tested", f"{len(allc)}", f"{len(rows)} contrasts × {len(H)} horizons", T.CYAN),
        ("Sign-consistent", f"{sum(ok for _, ok in allc)}", "every pre-COVID fold agrees", T.EMERALD),
        ("FLIP", f"{sum(not ok for _, ok in allc)}", "no claim made", T.SLATE),
    ], cols=3)
    T.provenance("baseline_results.csv", "ablation_contrasts.csv", "dominance_delay_cost.csv",
                 "policy_surface.csv", "graph_gnn_ablation.csv")

# ── E2 ───────────────────────────────────────────────────────────────
with tabs[1]:
    T.section("E2", "How hard is the bar?",
              "Three baselines on as-of inputs: B0 carries the last published value "
              "forward, B1 is the regional climatology of training seasons, B2 is "
              "gradient boosting on eight as-of lags. Any genomic model has to beat B2.")
    r = D.report("baseline_results.csv")
    r = r[r.stratum == "all"].groupby(["fold", "horizon", "model"]).mae.mean().reset_index()
    fig = go.Figure()
    for m, name in (("B0", "B0 persistence"), ("B1", "B1 climatology"), ("B2", "B2 boosting")):
        g = r[r.model == m]
        mean = g.groupby("horizon").mae.mean()
        fig.add_trace(go.Scatter(x=g.horizon + {"B0": -0.12, "B1": 0, "B2": 0.12}[m], y=g.mae,
                                 mode="markers", marker=dict(color=T.MODEL[m], size=7, opacity=0.45),
                                 name=f"{name} · per fold", showlegend=False,
                                 customdata=g.fold.map(D.FOLD_TEST),
                                 hovertemplate=f"{name}<br>test %{{customdata}}<br>MAE %{{y:.3f}}<extra></extra>"))
        fig.add_trace(go.Scatter(x=mean.index, y=mean.values, mode="lines+markers", name=name,
                                 line=dict(color=T.MODEL[m], width=3), marker=dict(size=9)))
    fig.update_layout(height=420, xaxis=dict(title="horizon (weeks ahead)", tickvals=H),
                      yaxis_title="MAE (wILI points)", legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    sk = e2["per_fold"].copy()
    sk["test"] = sk.fold.map(D.FOLD_TEST)
    piv = sk.pivot_table(index="test", columns="horizon", values="B2")
    piv = piv.reindex([D.FOLD_TEST[f] for f, _, _ in D.FOLDS])
    fig = go.Figure(go.Heatmap(z=piv.values, x=[f"h={h}" for h in piv.columns], y=piv.index,
                               zmid=0, colorscale=[[0, T.ROSE], [0.5, "#1A2435"], [1, T.EMERALD]],
                               text=np.round(piv.values, 1), texttemplate="%{text:+.1f}",
                               xgap=3, ygap=3, colorbar=dict(title="skill %")))
    fig.update_layout(height=300, title="B2 skill over persistence, by test season")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    worst = piv.mean(axis=1).idxmin()
    note = (" That is the season after the COVID gap, which the project's rules exclude "
            "from headline tables and report separately as a regime-change finding."
            if worst == D.COVID_TEST else "")
    npos = int((piv.mean(axis=0) > 0).sum())
    T.callout(f"B2's mean skill over persistence is positive at {npos} of {piv.shape[1]} "
              f"horizons; the test "
              f"season where it struggles most is <b>{worst}</b> "
              f"(mean {piv.loc[worst].mean():+.1f}%).{note}", T.BLUE)
    T.provenance("data/reports/baseline_results.csv")

# ── E3a ──────────────────────────────────────────────────────────────
with tabs[2]:
    T.section("E3a", "Does composition improve an incidence forecast?",
              "Eight arms add genomic features to the B2 baseline (A0). The primary "
              "endpoint is the delay cost A6−A4: how much worse deployable genomics "
              "is than oracle genomics.")
    order = [c for c in R.E3A_CONTRASTS if c in e3a["matrix"]]
    st.plotly_chart(matrix_fig([(f"{c} · {e3a['labels'][c]}", e3a["matrix"][c]) for c in order]),
                    width="stretch", config={"displayModeBar": False})
    con = e3a["contrasts"]
    c1, c2 = st.columns([1, 1])
    with c1:
        pick = st.selectbox("Contrast", order, key="e3a_c",
                            format_func=lambda c: f"{c} · {e3a['labels'][c]}")
    with c2:
        strat = st.radio("Stratum", ["all", "pre-peak", "peak or after"], horizontal=True, key="e3a_s")
    s = con[(con.contrast == pick) & (con.stratum == strat)]
    ab = D.report("ablation_results.csv")
    # The intervals are stored in MAE units (d_mae). skill_pct is normalised
    # by the SECOND arm of the contrast ("A6-A4" -> A4), verified against the
    # artifact, so the interval is converted with that same reference.
    ref_arm = pick.split("-")[1]
    fig = go.Figure()
    for i, (f, _, te) in enumerate(D.FOLDS):
        g = s[s.fold == f].sort_values("horizon")
        if g.empty:
            continue
        ref = ab[(ab.arm == ref_arm) & (ab.fold == f) & (ab.stratum == strat)].set_index("horizon").mae
        lo = -100 * g.ci_hi.values / ref.reindex(g.horizon).values
        hi = -100 * g.ci_lo.values / ref.reindex(g.horizon).values
        x = g.horizon + (i - 2) * 0.13
        col = T.SLATE if te == D.COVID_TEST else T.REGION_COLOURS[i]
        fig.add_trace(go.Scatter(
            x=x, y=g.skill_pct, mode="markers", name=f"test {te}" + (" (COVID gap)" if te == D.COVID_TEST else ""),
            marker=dict(size=10, color=col, symbol="diamond" if te == D.COVID_TEST else "circle"),
            error_y=dict(type="data", symmetric=False, array=hi - g.skill_pct.values,
                         arrayminus=g.skill_pct.values - lo, color=col, thickness=1.2, width=0)
            if g.ci_lo.notna().all() else None))
    fig.add_hline(y=0, line=dict(color=T.MUTED))
    fig.update_layout(height=420, xaxis=dict(title="horizon", tickvals=H),
                      yaxis_title="skill % (positive = first arm better)",
                      legend=dict(orientation="h", y=-0.2),
                      title=f"{pick} per fold, 95% block-bootstrap interval")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    fl = e3a["floor"]
    T.callout(f"<b>Detectable floor.</b> The median bootstrap half-width of the primary "
              f"contrast is {min(fl.values()):.1f}% of the baseline MAE at h={min(fl, key=fl.get)} "
              f"rising to {max(fl.values()):.1f}% at h={max(fl, key=fl.get)}. A delay cost larger "
              f"than that would have been resolved. How strong a <i>genomic predictor</i> "
              f"would need to be is answered by E8 on the Detectability page.", T.CYAN)
    T.callout("<b>The null is controlled.</b> The run log records a positive control "
              "(target + noise) that improves every fold at every horizon, and a "
              "missingness-flag-only arm that changes nothing — so the null is neither "
              "a broken harness nor an artefact of the NaN pattern. "
              "<span class='es-foot'>Source: RESULTS.md, E3 section (console output; "
              "not persisted as a CSV, so not re-derived here).</span>", T.SLATE)
    T.provenance("data/reports/ablation_contrasts.csv", "ablation_results.csv")

# ── E3b ──────────────────────────────────────────────────────────────
with tabs[3]:
    T.section("E3b", "Composition does forecast which subtype will dominate",
              "The target changes from incidence to next weeks' H3N2 share. Here "
              "genomics wins — and the cost of publication delay can be measured "
              "directly, as oracle-dated minus as-of-dated skill.")
    res = e3b["results"]
    res = res[(res.stratum == "all") & (res.fold.isin(D.PRE_FOLDS))]
    metric = st.radio("Metric", ["accuracy", "mae", "brier"], horizontal=True, key="e3b_m",
                      format_func=lambda m: {"accuracy": "Dominance accuracy",
                                             "mae": "Share MAE", "brier": "Brier score"}[m])
    pal = {"nat ORACLE": T.VIOLET, "reg ORACLE": T.PINK, "nat as-of": T.AMBER,
           "reg as-of": "#FCD34D", "train-majority": T.SLATE}
    fig = go.Figure()
    for pr in [p for p in pal if p in set(res.predictor)]:
        g = res[res.predictor == pr].groupby("horizon")[metric].mean()
        fig.add_trace(go.Scatter(x=g.index, y=g.values, name=pr, mode="lines+markers",
                                 line=dict(color=pal[pr], width=3,
                                           dash="dot" if "ORACLE" in pr else "solid")))
    fig.update_layout(height=380, xaxis=dict(title="horizon", tickvals=H),
                      yaxis_title=metric, legend=dict(orientation="h", y=-0.22),
                      title="Mean over the four pre-COVID folds")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    costs = e3b["costs"]
    pc = costs[(costs.contrast == e3b["primary"]) & (costs.stratum == "all")]
    fig = go.Figure()
    for i, (f, _, te) in enumerate(D.FOLDS):
        g = pc[pc.fold == f].sort_values("horizon")
        col = T.SLATE if te == D.COVID_TEST else T.REGION_COLOURS[i]
        fig.add_trace(go.Scatter(
            x=g.horizon + (i - 2) * 0.13, y=g.skill_pct, mode="markers",
            name=f"test {te}" + (" (COVID gap)" if te == D.COVID_TEST else ""),
            marker=dict(size=10, color=col),
            error_y=dict(type="data", symmetric=False, array=g.ci_hi - g.skill_pct,
                         arrayminus=g.skill_pct - g.ci_lo, color=col, width=0)))
    fig.add_hline(y=0, line=dict(color=T.MUTED))
    fig.add_hrect(y0=e3b["lo"], y1=e3b["hi"], fillcolor="rgba(52,211,153,0.07)", line_width=0)
    fig.update_layout(height=420, xaxis=dict(title="horizon", tickvals=H),
                      yaxis_title="skill lost to delay (%)", legend=dict(orientation="h", y=-0.2),
                      title=f"Delay cost per fold — {e3b['n_sig']} of {e3b['n_cells']} intervals exclude zero")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    dec = e3b["decomposition"]
    if not dec.empty:
        st.markdown("**Where the accuracy gap comes from** (oracle − as-of, percentage points, pre-COVID mean)")
        st.dataframe(dec.round(1).rename(columns=lambda h: f"h={h}"), width="stretch")
    T.callout(f"Delay costs <b>{e3b['lo']:.1f}% to {e3b['hi']:.1f}%</b> of dominance-forecast "
              f"skill, sign-consistent at every horizon. This is the project's central positive "
              f"result: genomic composition is informative, and the surveillance system's "
              f"publication lag throws most of it away.", T.EMERALD)
    T.provenance("data/reports/dominance_results.csv", "dominance_delay_cost.csv",
                 "dominance_decomposition.csv")
