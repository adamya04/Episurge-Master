import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import data as D
from core import models as M
from core import theme as T

T.hero(
    "Forecast studio · live",
    "Train the baselines <span class='es-grad'>right here</span>, on the real folds",
    "This page re-runs experiment E2 inside the app — same functions, same "
    "rolling-origin folds, same evaluation rows — and checks its own numbers "
    "against the committed results file before showing you anything.",
)

c1, c2, c3 = st.columns([1.2, 1, 1.4])
with c1:
    fold = st.selectbox("Test season (fold)", [f for f, _, _ in D.FOLDS], index=2,
                        format_func=lambda f: f"{D.FOLD_TEST[f]}  ·  {f}", key="fs_fold")
with c2:
    h = st.select_slider("Horizon (weeks ahead)", options=D.HORIZONS, value=2, key="fs_h")
with c3:
    region = st.selectbox("Region", D.REGIONS, index=3, format_func=D.region_label, key="fs_r")

frame, n_train, test_season = M.fold_forecasts(fold, h)
live = M.live_mae(frame)
committed = M.committed_mae(fold, h)
match = all(abs(live[m] - committed[m]) < 1e-3 for m in live)
train_seasons = dict((f, tr) for f, tr, _ in D.FOLDS)[fold]

T.cards([
    ("B0 persistence · MAE", f"{live['B0']:.3f}", "last published value", T.MODEL["B0"]),
    ("B1 climatology · MAE", f"{live['B1']:.3f}",
     f"{(live['B0'] - live['B1']) / live['B0'] * 100:+.1f}% vs B0", T.MODEL["B1"]),
    ("B2 boosting · MAE", f"{live['B2']:.3f}",
     f"{(live['B0'] - live['B2']) / live['B0'] * 100:+.1f}% vs B0", T.MODEL["B2"]),
    ("Reproducibility check", "✓ match" if match else "✗ drift",
     f"live vs baseline_results.csv, |Δ| ≤ "
     f"{max(abs(live[m] - committed[m]) for m in live):.4f}", T.EMERALD if match else T.ROSE),
], cols=4)
st.caption(f"Trained on {', '.join(train_seasons)} ({n_train:,} rows) → evaluated on "
           f"{test_season} ({len(frame):,} region-weeks). B2 seed 0.")

# ── trajectory ───────────────────────────────────────────────────────
T.section("Trajectory", f"{D.region_label(region)} · {test_season} · {h}-week-ahead forecasts",
          "Each marker is a forecast made h weeks earlier, drawn at the week it "
          "predicts. The shaded band is the period before the national peak.")
truth = M.season_truth(test_season)
tr = truth[truth.hhs_region == region].sort_values("season_week_idx")
rf = M.region_trajectory_frame(frame, region)
peak = int(tr.loc[tr.weeks_from_peak == 0, "season_week_idx"].iloc[0])
fig = go.Figure()
fig.add_vrect(x0=tr.season_week_idx.min(), x1=peak - 0.5, fillcolor="rgba(251,113,133,0.05)",
              line_width=0)
fig.add_trace(go.Scatter(x=tr.season_week_idx, y=tr.wili_final, name="observed (final)",
                         line=dict(color=T.INK, width=2.5),
                         customdata=tr.epiweek, hovertemplate="%{customdata}: %{y:.2f}<extra></extra>"))
for m, name, sym in (("B0", "B0 persistence", "circle"), ("B1", "B1 climatology", "square"),
                     ("B2", "B2 boosting", "diamond")):
    fig.add_trace(go.Scatter(x=rf.target_idx, y=rf[m], name=name, mode="lines+markers",
                             line=dict(color=T.MODEL[m], width=1.5,
                                       dash="dot" if m != "B2" else "solid"),
                             marker=dict(symbol=sym, size=7, color=T.MODEL[m])))
fig.update_layout(height=430, hovermode="x unified", yaxis_title="wILI (%)",
                  xaxis_title="week of season", legend=dict(orientation="h", y=-0.2))
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
T.provenance("live: baselines.build_targets / b0_persistence / b1_climatology / b2_gbm")

# ── one origin, every horizon ────────────────────────────────────────
T.section("From one origin", "Stand at a week and forecast every horizon",
          "Pick an origin. The app trains one model per horizon on this fold "
          "(cached after the first time) and fans the forecasts out from it.")
origins = sorted(frame[frame.hhs_region == region].season_week_idx.unique())
o = st.select_slider("Origin week", options=origins,
                     value=origins[min(len(origins) - 1, max(0, peak - 4 - origins[0]))],
                     key="fs_origin",
                     format_func=lambda i: f"{int(tr.set_index('season_week_idx').epiweek.get(i, 0))} "
                                           f"({i - peak:+d} wk)")
pts = {m: [] for m in ("B0", "B1", "B2")}
xs = []
for hh in D.HORIZONS:
    fh, _, _ = M.fold_forecasts(fold, hh)
    r = fh[(fh.hhs_region == region) & (fh.season_week_idx == o)]
    if r.empty:
        continue
    xs.append(o + hh)
    for m in pts:
        pts[m].append(float(r[m].iloc[0]))
fig = go.Figure()
fig.add_trace(go.Scatter(x=tr.season_week_idx, y=tr.wili_final, name="observed (final)",
                         line=dict(color=T.INK, width=2.5)))
# The forecaster at origin T has seen weeks up to T-1, as published at T
# (standing rule 2), so the fan starts from that point, not from the
# final revised value.
orow = frame[(frame.hhs_region == region) & (frame.season_week_idx == o)]
seen = float(orow.ili_asof_lag1.iloc[0]) if len(orow) else np.nan
fig.add_trace(go.Scatter(x=[o - 1], y=[seen], mode="markers", name="last value seen (as-of)",
                         marker=dict(color=T.AMBER, size=12, symbol="star")))
for m in pts:
    if not xs:
        break
    fig.add_trace(go.Scatter(x=[o - 1] + xs, y=[seen] + pts[m], name=m, mode="lines+markers",
                             line=dict(color=T.MODEL[m], width=3 if m == "B2" else 2),
                             marker=dict(size=8)))
fig.add_vline(x=o, line=dict(color=T.AMBER, dash="dash"),
              annotation_text="origin T", annotation_font=dict(color=T.AMBER))
fig.update_layout(height=400, yaxis_title="wILI (%)", xaxis_title="week of season",
                  legend=dict(orientation="h", y=-0.2))
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
if len(xs) < len(D.HORIZONS):
    st.caption(f"{len(D.HORIZONS) - len(xs)} horizon(s) unavailable from this origin: the "
               f"target falls past the end of the season.")

# ── where models fail ────────────────────────────────────────────────
T.section("Error anatomy", "Where each model fails",
          "Absolute error by the origin's position relative to the national peak, "
          "all regions, this fold.")
f2 = frame.copy()
f2["bucket"] = pd.cut(f2.weeks_from_peak, [-40, -9, -5, -2, 1, 4, 8, 40],
                      labels=["≤−10", "−9…−5", "−4…−2", "−1…+1", "+2…+4", "+5…+8", "≥+9"])
fig = go.Figure()
for m in ("B0", "B1", "B2"):
    e = f2.assign(err=(f2.y_target - f2[m]).abs()).groupby("bucket", observed=False).err.mean()
    fig.add_trace(go.Bar(x=e.index.astype(str), y=e.values, name=m, marker_color=T.MODEL[m]))
fig.update_layout(height=360, barmode="group", xaxis_title="origin, weeks from peak",
                  yaxis_title="MAE", legend=dict(orientation="h", y=-0.25))
worst = {m: f2.assign(err=(f2.y_target - f2[m]).abs()).groupby("bucket", observed=False).err.mean().idxmax()
         for m in ("B0", "B1", "B2")}
c1, c2 = st.columns([1.3, 1])
with c1:
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption("Worst bucket on this fold — " + ", ".join(f"{m}: {w} wk" for m, w in worst.items())
               + ". Change the fold or horizon to see how it moves.")
with c2:
    hm = pd.DataFrame({m: frame.assign(err=(frame.y_target - frame[m]).abs())
                       .groupby("hhs_region").err.mean() for m in ("B0", "B1", "B2")})
    fig = go.Figure(go.Heatmap(z=hm.values, x=hm.columns, y=[D.region_label(r) for r in hm.index],
                               colorscale=[[0, "#0B1424"], [0.55, "#0E5A74"], [1, "#0891B2"]],
                               text=np.round(hm.values, 2),
                               texttemplate="%{text}", colorbar=dict(title="MAE")))
    fig.update_layout(height=360, title="MAE by region")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
T.provenance("computed live on this fold")

with st.expander("The models, precisely"):
    st.markdown(
        "- **B0 persistence** — the most recent published value, `ili_asof_lag1`. No fitting.\n"
        "- **B1 climatology** — mean wILI for this region and calendar week across *training* "
        "seasons only. It defines the evaluation set: every model is scored on exactly the rows "
        "where B1 exists (paired evaluation, standing rule 3).\n"
        "- **B2 gradient boosting** — `HistGradientBoostingRegressor` (300 trees, depth 3, "
        "learning rate 0.05) on as-of lags 1–8, region and position in season.\n"
        "- **GNN** (E6, not trained live) — GRU over the lag window, a region embedding, one "
        "graph-convolution layer and a multi-horizon head predicting the residual from "
        "persistence. See Graph lab.")
