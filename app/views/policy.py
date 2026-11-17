import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import data as D
from core import results as R
from core import theme as T

T.hero(
    "Policy simulator · E3c",
    "If surveillance changed, <span class='es-grad'>how much skill comes back?</span>",
    "Two levers a public-health system actually controls: how quickly "
    "laboratories deposit sequences, and which lag window of sequences the "
    "forecaster uses. Every cell of this surface is a counterfactual re-run of "
    "the subtype-dominance forecaster on the real folds.",
)

e3c, e3b = R.e3c(), R.e3b()
raw = e3c["raw"]
W, Sh = e3c["windows"], e3c["shifts"]
CUR = ("L8-16", 0)

c1, c2, c3 = st.columns([1.3, 1.6, 1])
with c1:
    win = st.select_slider("Lag window of sequences used", options=W, value="L4-12", key="pol_w",
                           help="Lx-y: sequences collected x to y weeks before the origin")
with c2:
    sh = st.select_slider("Deposit speed-up (days earlier)", options=Sh, value=14, key="pol_s")
with c3:
    h = st.select_slider("Horizon", options=D.HORIZONS, value=4, key="pol_h")

m, ok = e3c["cells"].get((win, sh, h), (np.nan, False))
cell = raw[(raw.window == win) & (raw.shift_days == sh) & (raw.horizon == h)
           & raw.fold.isin(D.PRE_FOLDS)]
miss = float(e3c["missing"].get((win, sh), np.nan))
mae = float(cell.mae.mean())
cur_mae = float(raw[(raw.window == CUR[0]) & (raw.shift_days == CUR[1]) & (raw.horizon == h)
                    & raw.fold.isin(D.PRE_FOLDS)].mae.mean())
gap = float(cell.pct_gap_recovered.mean())
verdict = ("sign-consistent gain" if ok and m > 0 else
           "sign-consistent loss" if ok else "FLIP — folds disagree")
T.cards([
    ("Skill vs today's practice", "—" if m != m else f"{m:+.1f}%", verdict,
     T.EMERALD if ok and m > 0 else T.ROSE if ok else T.SLATE),
    ("H3N2-share error (MAE)", f"{mae:.3f}", f"today: {cur_mae:.3f} (L8-16, no speed-up)", T.CYAN),
    ("Delay gap recovered", "—" if gap != gap else f"{gap:.0f}%",
     f"share of the oracle–as-of gap ({e3b['lo']:.1f}–{e3b['hi']:.1f}% of skill)", T.VIOLET),
    ("Origins with no data", f"{miss:.0f}%", "window empty at the forecast date", T.AMBER),
], cols=4)
T.provenance("data/reports/policy_surface.csv · mean and sign over the four pre-COVID folds")

# ── 3D surface ───────────────────────────────────────────────────────
T.section("The surface", "Skill over both levers at once",
          "Drag to rotate. The amber marker is today's practice; the cyan marker "
          "is your selection. Cliffs on the left are windows that are empty "
          "because nothing that recent has been published yet.")
Z = np.array([[e3c["cells"].get((w, s, h), (np.nan, False))[0] for s in Sh] for w in W])
OK = np.array([[e3c["cells"].get((w, s, h), (np.nan, False))[1] for s in Sh] for w in W])
lim = float(np.nanmax(np.abs(Z)))
fig = go.Figure()
fig.add_trace(go.Surface(
    z=Z, x=Sh, y=list(range(len(W))), cmin=-lim, cmax=lim,
    colorscale=[[0, "#9F1239"], [0.35, "#D9415A"], [0.5, "#1A2435"], [0.65, "#0E9F6E"], [1, "#065F46"]],
    customdata=np.where(OK, "sign-consistent", "FLIP"),
    hovertemplate="window %{y}<br>speed-up %{x} d<br>skill %{z:+.1f}%<br>%{customdata}<extra></extra>",
    contours=dict(z=dict(show=True, usecolormap=True, project_z=True, width=1)),
    colorbar=dict(title="skill %"), opacity=0.95))
for (w_, s_, col, name) in ((CUR[0], CUR[1], T.AMBER, "today"), (win, sh, T.CYAN, "selected")):
    zz = e3c["cells"].get((w_, s_, h), (np.nan, False))[0]
    fig.add_trace(go.Scatter3d(x=[s_], y=[W.index(w_)], z=[zz], mode="markers+text", text=[name],
                               textfont=dict(color=col), marker=dict(size=8, color=col,
                                                                     line=dict(color="#070B13", width=2)),
                               showlegend=False))
fig.update_layout(
    height=560, margin=dict(l=0, r=0, t=10, b=0),
    scene=dict(xaxis=dict(title="deposit speed-up (days)", backgroundcolor="rgba(0,0,0,0)",
                          gridcolor=T.LINE, color=T.MUTED),
               yaxis=dict(title="lag window", tickvals=list(range(len(W))), ticktext=W,
                          backgroundcolor="rgba(0,0,0,0)", gridcolor=T.LINE, color=T.MUTED),
               zaxis=dict(title="skill vs today (%)", backgroundcolor="rgba(0,0,0,0)",
                          gridcolor=T.LINE, color=T.MUTED),
               camera=dict(eye=dict(x=1.45, y=-1.4, z=0.78))))
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

# ── verdict grid ─────────────────────────────────────────────────────
c1, c2 = st.columns([1.25, 1])
with c1:
    txt = np.where(np.isnan(Z), "–", np.where(OK, np.char.mod("%+.1f", np.nan_to_num(Z)), "FLIP"))
    txt = txt.astype(object)
    if CUR[0] in W and CUR[1] in Sh:
        txt[W.index(CUR[0]), Sh.index(CUR[1])] = "today"   # the reference itself: skill 0 by definition
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=np.where(OK, Z, np.nan), x=[f"+{s} d" for s in Sh], y=W,
                             text=txt, texttemplate="%{text}", zmid=0, zmin=-lim, zmax=lim,
                             colorscale=[[0, "#9F1239"], [0.35, "#D9415A"], [0.5, "#1A2435"],
                                         [0.65, "#0E9F6E"], [1, "#065F46"]],
                             xgap=3, ygap=3, colorbar=dict(title="skill %"),
                             textfont=dict(family=T.MONO)))
    fig.update_layout(height=340, title=f"Verdicts at h={h} — grey cells are FLIP",
                      plot_bgcolor="#141C2B")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
with c2:
    fig = go.Figure()
    for i, (f, _, te) in enumerate(D.FOLDS):
        g = raw[(raw.window == win) & (raw.shift_days == sh) & (raw.horizon == h) & (raw.fold == f)]
        if g.empty:
            continue
        col = T.SLATE if te == D.COVID_TEST else T.REGION_COLOURS[i]
        err = None
        if g.ci_lo.notna().all():
            err = dict(type="data", symmetric=False, array=g.ci_hi - g.skill_vs_current,
                       arrayminus=g.skill_vs_current - g.ci_lo, color=T.MUTED)
        fig.add_trace(go.Bar(x=[te + (" *" if te == D.COVID_TEST else "")], y=g.skill_vs_current,
                             marker_color=col, error_y=err, showlegend=False))
    fig.add_hline(y=0, line=dict(color=T.MUTED))
    fig.update_layout(height=340, title=f"{win}, +{sh} d, h={h}: per fold",
                      yaxis_title="skill vs today (%)")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption("Starred: 2022-23 follows the COVID gap and is excluded from the verdict, per the "
               "project's standing rules; it is shown, not hidden.")

# ── what it means ────────────────────────────────────────────────────
T.section("Reading the surface", "Three policy-relevant facts, derived")
fast = R.e3c_row(e3c, "L8-16", max(Sh))
tight = R.e3c_row(e3c, "L0-4", 0)
both = R.e3c_row(e3c, "L4-12", 14)
f_lo, f_hi = R.consistent_range(fast)
t_lo, t_hi = R.consistent_range(tight)
b_lo, b_hi = R.consistent_range(both)
nb = sum(ok for _, ok in both.values())
c1, c2, c3 = st.columns(3)
with c1:
    T.callout(f"<b>A tighter window alone hurts.</b> L0-4 without faster deposit: "
              f"{t_lo:+.1f}% to {t_hi:+.1f}%. Narrowing the window empties it — those "
              f"sequences do not exist yet at the forecast date.", T.ROSE)
with c2:
    T.callout(f"<b>Faster deposit alone helps a little.</b> Current window, "
              f"{max(Sh)} days faster: {f_lo:+.1f}% to {f_hi:+.1f}%.", T.AMBER)
with c3:
    T.callout(f"<b>Together they are complements.</b> L4-12 with a 14-day speed-up gains "
              f"{b_lo:+.1f}% to {b_hi:+.1f}%, sign-consistent at {nb} of {len(both)} horizons — "
              f"a two-week change a surveillance programme could actually make.", T.EMERALD)
