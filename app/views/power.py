import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import data as D
from core import results as R
from core import theme as T

T.hero(
    "Detectability · E8 + E4/E5",
    "A null is only as strong as <span class='es-grad'>the power behind it</span>",
    "\"We found nothing\" is not a result until you can say what you could have "
    "found. Two experiments answer that by injecting signals of known strength "
    "into the real panel — with its real gaps — and recording when each "
    "instrument starts to see them.",
)

e8, e45 = R.e8(), R.e45()

# ── the ruler ────────────────────────────────────────────────────────
T.section("The ruler", "How much of influenza's variance a genomic predictor would need to carry",
          "Both experiments sweep α, the correlation of a synthetic predictor with "
          "the standardised real target, so α² is the share of target variance it "
          "carries. On that single scale:")
pre_max = float(e8["prepeak"].alpha.max()) if e8["prepeak"] is not None else np.nan
fig = go.Figure()
fig.add_shape(type="rect", x0=np.log10(e8["var_lo"]), x1=np.log10(e8["var_hi"]), y0=0.3, y1=0.7,
              fillcolor="rgba(245,165,36,0.45)", line=dict(color=T.AMBER, width=2))
fig.add_shape(type="line", x0=np.log10(e45["floor_var"]), x1=np.log10(e45["floor_var"]),
              y0=0.2, y1=0.8, line=dict(color=T.CYAN, width=4))
fig.add_annotation(x=np.log10(e45["floor_var"]), y=0.92,
                   text=f"<b>E4 Granger resolves</b><br>α = {e45['floor_alpha']:.2f} → {e45['floor_var']:.1f}%",
                   showarrow=False, font=dict(color=T.CYAN, size=15))
fig.add_annotation(x=np.log10(np.sqrt(e8["var_lo"] * e8["var_hi"])), y=0.12,
                   text=f"<b>E8 forecasting design resolves</b><br>α ≈ {e8['alpha_lo']:.2f}–{e8['alpha_hi']:.2f} "
                        f"→ {e8['var_lo']:.0f}–{e8['var_hi']:.0f}%",
                   showarrow=False, font=dict(color=T.AMBER, size=15))
if pre_max == pre_max and e8["pre_never"]:
    xv = np.log10(100 * pre_max ** 2)
    fig.add_annotation(x=2.0, y=0.5, ax=xv, ay=0.5, xref="x", yref="y", axref="x", ayref="y",
                       showarrow=True, arrowhead=2, arrowwidth=2.5, arrowcolor=T.ROSE)
    fig.add_annotation(x=(xv + 2) / 2, y=0.78,
                       text=f"<b>pre-peak, real coverage:</b><br>not even at α = {pre_max:.1f} ({100 * pre_max ** 2:.0f}%)",
                       showarrow=False, font=dict(color=T.ROSE, size=14))
fig.update_layout(
    height=330, margin=dict(l=10, r=10, t=10, b=40),
    xaxis=dict(range=[-1.1, 2.05], tickvals=[-1, 0, 1, 2], ticktext=["0.1%", "1%", "10%", "100%"],
               tickfont=dict(size=14),
               title="share of target variance carried by the genomic predictor (log scale)"),
    yaxis=dict(visible=False, range=[0, 1.05]))
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
ratio = e8["var_lo"] / e45["floor_var"]
T.callout(f"The model-free test is roughly <b>{ratio:.0f}×</b> finer in variance than the "
          f"forecasting design — and applied to the real data it <b>also finds nothing</b> "
          f"({e45['n_bh']} of {e45['n_arms']} arms survive correction). E3a's null could have "
          f"been a blunt instrument; E8 says exactly how blunt; E4 rules the artefact out.",
          T.CYAN)

# ── E8 frontier ──────────────────────────────────────────────────────
T.section("E8", "The detectability frontier",
          f"Each point is the mean skill of B2 with an injected predictor of strength α, "
          f"over {e8['reps']} replicates. φ reveals a fraction of the cells the real "
          f"surveillance system leaves empty. The dashed line is E3a's detectable floor "
          f"({e8['floor_pct']:.1f}%).")
c1, c2 = st.columns(2)
with c1:
    strat = st.radio("Origins", ["all", "pre-peak"], horizontal=True, key="e8_s",
                     format_func=lambda s: {"all": "whole season", "pre-peak": "pre-peak only"}[s])
with c2:
    h = st.radio("Horizon", sorted(e8["frontier"].horizon.unique()), horizontal=True, key="e8_h",
                 format_func=lambda x: f"h={x}")
F = e8["frontier"] if strat == "all" else e8["prepeak"]
mde = e8["mde"] if strat == "all" else e8["mde_pre"]
g = F[F.horizon == h]
fig = go.Figure()
cols = {0.0: T.ROSE, 0.5: T.AMBER, 1.0: T.EMERALD}
for phi in sorted(g.phi.unique()):
    s = g[g.phi == phi].groupby("alpha").agg(m=("mean_skill", "mean"), lo=("mean_skill", "min"),
                                             hi=("mean_skill", "max"), cov=("coverage", "mean"))
    c = cols.get(phi, T.CYAN)
    fig.add_trace(go.Scatter(x=list(s.index) + list(s.index[::-1]), y=list(s.hi) + list(s.lo[::-1]),
                             fill="toself", fillcolor=c, opacity=0.12, line=dict(width=0),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=s.index, y=s.m, mode="lines+markers",
                             name=f"φ={phi:.1f} · coverage {100 * s['cov'].mean():.0f}%",
                             line=dict(color=c, width=3), marker=dict(size=8)))
    mrow = mde[(mde.horizon == h) & (mde.phi == phi)]
    if len(mrow) and mrow.mde_alpha.iloc[0] == mrow.mde_alpha.iloc[0]:
        fig.add_vline(x=float(mrow.mde_alpha.iloc[0]), line=dict(color=c, width=1, dash="dot"))
fig.add_hline(y=e8["floor_pct"], line=dict(color=T.MUTED, dash="dash"),
              annotation_text="detectable floor", annotation_font=dict(color=T.MUTED))
fig.update_layout(height=430, xaxis_title="injected strength α", yaxis_title="mean skill (%)",
                  legend=dict(orientation="h", y=-0.22), hovermode="x unified")
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
mt = mde[mde.horizon == h].copy()
st.dataframe(pd.DataFrame({
    "availability φ": mt.phi.map(lambda v: f"{v:.1f}"),
    "MDE α": mt.mde_alpha.map(lambda v: "never reaches the floor" if v != v else f"{v:.3f}"),
    "variance a predictor must carry": mt.mde_var_explained.map(lambda v: "—" if v != v else f"{100 * v:.1f}%"),
}), hide_index=True, width="stretch")
T.provenance(f"data/reports/detectability_frontier{'' if strat == 'all' else '_prepeak'}.csv",
             "MDE via detectability_frontier.mde_from_curve (imported, not reimplemented)")

c1, c2 = st.columns([1.4, 1])
with c1:
    dr = g.groupby(["alpha", "phi"]).detected.mean().unstack()
    fig = go.Figure(go.Heatmap(z=dr.values.T, x=[f"{a:.1f}" for a in dr.index],
                               y=[f"φ={p:.1f}" for p in dr.columns], zmin=0, zmax=1,
                               colorscale=[[0, "#0B1424"], [0.55, "#0E5A74"], [1, "#0891B2"]],
                               text=np.round(dr.values.T * 100).astype(int),
                               texttemplate="%{text}%", colorbar=dict(title="detected")))
    fig.update_layout(height=280, title="Share of replicates that pass the per-fold rule",
                      xaxis_title="α")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
with c2:
    T.cards([("Null calibration", f"{e8['n_fp']} / {e8['n_null']}",
              "false positives when pure noise (α = 0) is injected — the rule is "
              "conservative, so every detection above is a floor", T.EMERALD)], cols=1)

# ── E4/E5 ────────────────────────────────────────────────────────────
T.section("E4 / E5", "Model-free information flow",
          "Panel Granger causality and transfer entropy test whether past composition "
          "carries information about incidence without fitting any forecaster. "
          "Significance comes from a circular-shift surrogate null, because panel "
          "residuals are autocorrelated and an analytic p-value would be anti-conservative.")
arms = e45["arms"].copy()
arms["label"] = (arms.test.str.replace("E4 ", "").str.replace("E5 ", "") + " · "
                 + arms.source + " · " + arms.stratum + " · " + arms.param)
arms = arms.sort_values("p_emp")
fig = go.Figure()
fig.add_trace(go.Scatter(x=arms.p_emp, y=arms.label, mode="markers", name="surrogate p",
                         marker=dict(size=11, color=np.where(arms.p_emp < 0.05, T.AMBER, T.SLATE))))
fig.add_trace(go.Scatter(x=arms.bh_q, y=arms.label, mode="markers", name="BH-corrected q",
                         marker=dict(size=11, symbol="diamond-open", color=T.CYAN, line=dict(width=2))))
for v, lab, c in ((0.05, "p = 0.05", T.MUTED), (e45["bonferroni"], "Bonferroni", T.ROSE)):
    # add_vline misplaces its annotation on a log axis; annotations on a
    # log axis take log10 coordinates, so place it explicitly.
    fig.add_vline(x=v, line=dict(color=c, dash="dash"))
    fig.add_annotation(x=np.log10(v), y=1.0, yref="paper", yanchor="bottom", text=lab,
                       showarrow=False, font=dict(color=c, size=11))
fig.update_layout(height=480, xaxis=dict(type="log", title="p (log scale)", range=[-2.6, 0.05]),
                  legend=dict(orientation="h", y=-0.15))
fig.update_yaxes(autorange="reversed")
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
T.callout(f"<b>{e45['n_raw']}</b> of {e45['n_arms']} arms cross p&nbsp;=&nbsp;0.05 uncorrected, "
          f"against <b>{e45['expected']:.1f}</b> expected by chance alone. After "
          f"Benjamini–Hochberg correction the smallest q is {e45['bh_min']:.3f}: "
          f"<b>{e45['n_bh']} arms survive</b>.", T.AMBER)

pw = e45["power"]
fig = go.Figure()
for col, name, c, d in (("src_granger_p", "Granger · source-side (E8 scale)", T.EMERALD, "solid"),
                        ("granger_p", "Granger · target-side", T.CYAN, "dot"),
                        ("te_p", "transfer entropy · target-side", T.VIOLET, "dash")):
    fig.add_trace(go.Scatter(x=pw.alpha, y=pw[col], name=name, mode="lines+markers",
                             line=dict(color=c, width=3 if "source" in name else 2, dash=d)))
fig.add_hline(y=0.05, line=dict(color=T.MUTED, dash="dash"))
fig.add_vline(x=e45["floor_alpha"], line=dict(color=T.EMERALD, width=1, dash="dot"),
              annotation_text=f"floor α={e45['floor_alpha']:.2f}",
              annotation_font=dict(color=T.EMERALD))
fig.update_layout(height=380, xaxis_title="injected coupling α", yaxis_title="surrogate p",
                  legend=dict(orientation="h", y=-0.22),
                  title="Positive control: the tests fire when coupling exists")
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
T.callout("Only the <b>source-side</b> curve is comparable with E8: it injects a predictor "
          "against the real target, exactly as E8 does. The target-side control fires "
          "earlier, but it measures coupling into a synthetic AR(1) process, and quoting it "
          "against E8 would overstate the sensitivity gap. The app, like the write-up, quotes "
          "only the comparable number.", T.SLATE)
T.provenance("data/reports/information_flow.csv", "information_flow_power.csv")
