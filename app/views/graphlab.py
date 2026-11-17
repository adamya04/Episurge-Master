import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import data as D
from core import geo
from core import graphs as GR
from core import results as R
from core import theme as T
from core.charts import matrix_fig

T.hero(
    "Graph lab · E6",
    "Does influenza travel along a graph a model can <span class='es-grad'>learn in advance</span>?",
    "A spatio-temporal GNN can only help if the relation between regions it "
    "is given still holds next season. This lab builds each candidate graph "
    "exactly as the experiment did, shows it on the map, animates how the "
    "data-driven one rewires from season to season, and reports what the "
    "trained GNN made of each.",
)

e6 = R.e6()
cents = geo.region_centroids()
REL_NAME = {"corr": "Correlation (learned from wILI)", "climate": "Climate similarity",
            "mass": "Population mass", "random": "Random (G5 control)",
            "full": "Fully connected"}


def outline_trace():
    xs, ys = [], []
    for f in geo.contiguous_geojson()["features"]:
        g = f["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        for poly in polys:
            ring = poly[0][::2] + [poly[0][0]]       # decorative backdrop: thinned
            xs += [p[0] for p in ring] + [None]
            ys += [p[1] for p in ring] + [None]
    return go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#1C2840", width=0.8),
                      fill="toself", fillcolor="rgba(19,28,45,0.55)", hoverinfo="skip",
                      showlegend=False)


def curve(p0, p1, bend=0.18, n=24):
    (x0, y0), (x1, y1) = p0, p1
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = x1 - x0, y1 - y0
    cx, cy = mx - dy * bend, my + dx * bend
    t = np.linspace(0, 1, n)
    return ((1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t ** 2 * x1,
            (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t ** 2 * y1)


def edges_trace(pairs, colour, width, name):
    xs, ys = [], []
    for i, j in pairs:
        cx, cy = curve(cents[D.REGIONS[i]], cents[D.REGIONS[j]])
        xs += list(cx) + [None]
        ys += list(cy) + [None]
    return go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=colour, width=width),
                      hoverinfo="skip", name=name, opacity=0.85)


def nodes_trace(size=None):
    x = [cents[r][0] for r in D.REGIONS]
    y = [cents[r][1] for r in D.REGIONS]
    return go.Scatter(x=x, y=y, mode="markers+text", text=[f"R{r}" for r in D.REGIONS],
                      textposition="middle center", textfont=dict(color="#070B13", size=10,
                                                                  family=T.MONO),
                      marker=dict(size=size or 26, color=T.REGION_COLOURS,
                                  line=dict(color="#070B13", width=2)),
                      hovertext=[D.region_label(r) for r in D.REGIONS], hoverinfo="text",
                      showlegend=False)


def map_layout(fig, h=470):
    fig.update_layout(height=h, margin=dict(l=0, r=0, t=10, b=0),
                      xaxis=dict(visible=False, range=[-125, -66.5]),
                      yaxis=dict(visible=False, range=[24, 50], scaleanchor="x", scaleratio=1.25),
                      legend=dict(orientation="h", y=0.02, x=0.01, bgcolor="rgba(7,11,19,0.6)"))
    return fig


# ── relation explorer ────────────────────────────────────────────────
T.section("Relation explorer", "Five ways to wire ten regions",
          f"Each region sends messages from its {GR.K} chosen neighbours (all graphs "
          f"share this density, except fully-connected). Built on the fold's training "
          f"seasons, minus the most recent one, which the GNN held out for early stopping.")
c1, c2 = st.columns([1, 1])
with c1:
    kind = st.radio("Relation", GR.REL_KINDS, format_func=REL_NAME.get, horizontal=True,
                    key="gl_kind")
with c2:
    fold = st.selectbox("Fold", [f for f, _, _ in D.FOLDS], index=4, key="gl_fold",
                        format_func=lambda f: f"{f} · trains for test {D.FOLD_TEST[f]}")
train = GR.gnn_train_rows(fold)
A = GR.adjacency(kind, train)
fig = go.Figure([outline_trace(),
                 edges_trace([(i, j) for i, j in GR.edge_set(A)], T.CYAN if kind != "random" else T.SLATE,
                             1.6 if kind != "full" else 0.6, REL_NAME[kind]),
                 nodes_trace()])
c1, c2 = st.columns([2.4, 1])
with c1:
    st.plotly_chart(map_layout(fig), width="stretch", config={"displayModeBar": False})
with c2:
    ins = A.sum(axis=0)
    hub = D.REGIONS[int(np.argmax(ins))]
    T.cards([("Edges", f"{int(A.sum())}", "directed, region → neighbour", T.CYAN),
             ("Most-attended region", f"R{hub}",
              f"{D.REGION_HQ[hub]} · chosen by {int(ins.max())} regions", T.VIOLET)], cols=1)
    expl = {
        "corr": "Neighbours are the regions whose as-of wILI moved most like this one "
                "in training. Intuitive — and the relation the instability result is about.",
        "climate": "Neighbours are the regions with the most similar temperature and "
                   "absolute-humidity climatology. A property of place, so it cannot drift.",
        "mass": "Every region attends to the most populous other regions: a distance-free "
                "gravity hypothesis that spread flows from large population centres.",
        "random": "The density-matched control mandated by Master Plan §4.3: any real "
                  "relation must beat this to mean anything.",
        "full": "Every region hears every other — the upper bound on mixing.",
    }
    T.callout(expl[kind], T.SLATE)
T.provenance("app/core/graphs.py mirrors graph_relation_ablation.layer_c.adjacency")

# ── evolving network ─────────────────────────────────────────────────
T.section("Evolving network", "The correlation graph rewires every season",
          "Each frame fits the top-3 correlation graph on one season alone. Cyan "
          "edges survive from the previous season; amber edges are new. Press play.")
graphs = {s: GR.season_corr_graph(s) for s in D.SEASONS}
frames, ovs = [], []
for i, s in enumerate(D.SEASONS):
    E = GR.edge_set(graphs[s])
    prev = GR.edge_set(graphs[D.SEASONS[i - 1]]) if i else E
    keep, new = [e for e in E if e in prev], [e for e in E if e not in prev]
    if i:
        ovs.append((D.SEASONS[i - 1], s, GR.overlap(graphs[D.SEASONS[i - 1]], graphs[s])))
    # frames update only the two edge traces; outline and nodes are sent once
    frames.append(go.Frame(name=s, traces=[1, 2],
                           data=[edges_trace(keep, T.CYAN, 2.2, "kept from last season"),
                                 edges_trace(new, T.AMBER, 2.2, "new this season")]))
fig = go.Figure(data=[outline_trace()] + list(frames[0].data) + [nodes_trace()], frames=frames)
fig.update_layout(
    updatemenus=[dict(type="buttons", x=0.01, y=0.99, xanchor="left", yanchor="top",
                      bgcolor=T.CYAN, bordercolor=T.CYAN, font=dict(color="#070B13", size=12), showactive=False,
                      buttons=[dict(label="▶ Play seasons", method="animate",
                                    args=[None, dict(frame=dict(duration=1300, redraw=True),
                                                     transition=dict(duration=0), fromcurrent=False)])])],
    sliders=[dict(active=0, x=0.1, len=0.85, y=0.02, currentvalue=dict(prefix="Season ", font=dict(color=T.INK)),
                  font=dict(color=T.MUTED),
                  steps=[dict(label=s, method="animate",
                              args=[[s], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
                         for s in D.SEASONS])])
c1, c2 = st.columns([2.4, 1])
with c1:
    st.plotly_chart(map_layout(fig, 500), width="stretch", config={"displayModeBar": False})
with c2:
    stab = e6["stability"]
    chance = GR.K / (len(D.REGIONS) - 1)
    tt = stab[stab.kind == "train_to_test_overlap"].value
    mp = stab[stab.kind == "matrix_persistence"].value
    T.cards([
        ("Neighbours that survive a season", f"{100 * np.mean([o for *_, o in ovs]):.1f}%",
         f"consecutive seasons, live; chance {100 * chance:.1f}%", T.AMBER),
        ("Train-chosen = test's true top-3", f"{100 * tt.mean():.1f}%",
         f"per fold, from the experiment; chance {100 * chance:.1f}%", T.ROSE),
        ("Correlation-matrix persistence", f"r = {mp.mean():+.2f}",
         f"mean over {len(mp)} season pairs (range {mp.min():+.2f} to {mp.max():+.2f})", T.VIOLET),
    ], cols=1)
T.callout("Regions <i>are</i> correlated within a season — but <b>which</b> regions pair "
          "with which carries almost no information into the next season. A graph fitted "
          "on history is, for the season it must forecast, statistically close to a random "
          "graph. This is the mechanism behind the E6 null.", T.AMBER)
T.provenance("live: per-season correlation graphs on core panel", "data/reports/graph_stability.csv")

# ── stability matrix ─────────────────────────────────────────────────
with st.expander("Season × season stability matrix"):
    m = stab[stab.kind == "matrix_persistence"]
    S = pd.DataFrame(np.nan, index=D.SEASONS, columns=D.SEASONS)
    for _, r in m.iterrows():
        S.loc[r.a, r.b] = S.loc[r.b, r.a] = r.value
    for s in D.SEASONS:
        S.loc[s, s] = 1.0
    fig = go.Figure(go.Heatmap(z=S.values, x=S.columns, y=S.index, zmin=-1, zmax=1,
                               colorscale=[[0, T.ROSE], [0.5, "#1A2435"], [1, T.CYAN]],
                               text=np.round(S.values, 2), texttemplate="%{text}",
                               colorbar=dict(title="r")))
    fig.update_layout(height=380, title="Correlation between seasons' region-correlation matrices")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

# ── the model ────────────────────────────────────────────────────────
T.section("The model", "Spatio-temporal GNN, as trained in E6",
          "A GRU reads each region's eight as-of lags; a region embedding and a "
          "seasonal encoding are mixed in; one graph-convolution layer adds the "
          "mean of the neighbours' states; a head predicts the residual from "
          "persistence for all six horizons at once.")
c1, c2 = st.columns([1.1, 1])
with c1:
    st.graphviz_chart("""
digraph G {
  rankdir=TB; bgcolor="transparent"; nodesep=0.3; ranksep=0.32;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=13,
        color="#1F2B40", fillcolor="#131C2D", fontcolor="#E6EDF7", penwidth=1.4,
        margin="0.18,0.07"];
  edge [color="#56657E", arrowsize=0.7];
  X  [label="as-of wILI lags 1–8  (per region)", fillcolor="#0E3A52"];
  GRU [label="GRU · 48 hidden"];
  E  [label="region embedding · 8"];
  W  [label="week sin/cos → 8"];
  MIX [label="mix · Linear + ReLU"];
  SELF [label="self transform"];
  MSG [label="neighbour message  Â·h", fillcolor="#2E1065", color="#A78BFA"];
  A  [label="adjacency Â\\n(one of 6 relations)", shape=note, fillcolor="#1E1B4B", color="#A78BFA"];
  ADD [label="+", shape=circle, width=0.4];
  HEAD [label="MLP head · 6 horizons"];
  RES [label="+ persistence  (residual target)", fillcolor="#3B2A0B", color="#F5A524"];
  Y  [label="ŷ   h = 1, 2, 3, 4, 6, 8", fillcolor="#064E3B", color="#34D399"];
  {rank=same; GRU; E; W;}
  {rank=same; SELF; MSG; A;}
  X -> GRU -> MIX; E -> MIX; W -> MIX;
  MIX -> SELF -> ADD; MIX -> MSG -> ADD; A -> MSG [style=dashed, color="#A78BFA"];
  ADD -> HEAD -> RES -> Y;
}
""", width="content")
with c2:
    st.markdown(
        "**Temporal encoder.** A GRU reads the eight as-of lags of one region and "
        "summarises them in a 48-dimensional state.\n\n"
        "**Context.** A learned region embedding and a sine/cosine encoding of the "
        "week of season are concatenated and mixed in.\n\n"
        "**Graph convolution.** Each region's state is updated with its own transform "
        "*plus* the degree-normalised mean of its neighbours' states, Â·h. Â is the "
        "only thing that changes between the six arms of E6.\n\n"
        "**Residual head.** One head predicts all six horizons, as a correction to "
        "persistence — so an all-zero output reproduces B0, which is what made the "
        f"network trainable on {len(GR.gnn_train_rows(fold)):,} training rows "
        f"({fold}).\n\n"
        "**Training.** Adam, MAE loss, early stopping on the last *training* season, "
        f"{e6['raw'].seed.nunique()} seeds per arm. No PyTorch Geometric: at ten nodes, "
        "message passing is a single matrix product.")
T.provenance("src/analysis/graph_relation_ablation.py · class Net")

# ── message passing ──────────────────────────────────────────────────
T.section("Message passing, with real numbers", "What the graph layer adds",
          "Pick a week. Each region's own latest as-of wILI (z-scored across "
          "regions) is compared with the message it receives: the mean of its "
          "neighbours' values under the chosen relation. Where the two disagree, "
          "the graph is injecting information the region's own history lacks.")
c1, c2, c3 = st.columns(3)
test_season = D.FOLD_TEST[fold]
core = D.core_panel()
ts = core[core.season == test_season]
with c1:
    widx = st.select_slider("Week of the test season", options=sorted(ts.season_week_idx.unique()),
                            value=int(ts.loc[ts.weeks_from_peak == -3, "season_week_idx"].iloc[0])
                            if (ts.weeks_from_peak == -3).any() else 10, key="mp_week")
with c2:
    mk = st.radio("Relation", GR.REL_KINDS, format_func=REL_NAME.get, key="mp_kind")
with c3:
    st.caption(f"Graph from {fold}'s training data; week from its test season, {test_season}.")
wk = ts[ts.season_week_idx == widx].set_index("hhs_region").reindex(D.REGIONS)
hvec = wk.ili_asof_lag1.values
z = (hvec - np.nanmean(hvec)) / (np.nanstd(hvec) + 1e-9)
An = GR.normalise(GR.adjacency(mk, train))
msg = An @ z
fig = go.Figure()
lbl = [D.region_label(r) for r in D.REGIONS]
fig.add_trace(go.Bar(x=lbl, y=z, name="own latest value (z)", marker_color=T.CYAN))
fig.add_trace(go.Bar(x=lbl, y=msg, name="neighbour message Â·h (z)", marker_color=T.VIOLET))
fig.update_layout(height=360, barmode="group", yaxis_title="z-score across regions",
                  legend=dict(orientation="h", y=-0.25))
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
corr = np.corrcoef(z, msg)[0, 1] if np.std(msg) > 0 else np.nan
st.caption(f"Correlation between own value and incoming message this week: {corr:+.2f}. "
           "Near 1, the graph mostly repeats what the region already knows.")

# ── results ──────────────────────────────────────────────────────────
T.section("What the trained GNN made of it", "Six relations × six horizons × four folds × seeds",
          "Mean MAE over the pre-COVID folds. Persistence and gradient boosting "
          "are the reference lines.")
mm = e6["mean_mae"]
fig = go.Figure()
style = {"B0 persistence": (T.MODEL["B0"], "dot", 2), "B2 boosting": (T.MODEL["B2"], "dash", 3)}
for arm in mm.index:
    col, dash, w = style.get(arm, (None, "solid", 2.5))
    if arm in style:
        fig.add_trace(go.Scatter(x=mm.columns, y=mm.loc[arm], name=arm,
                                 line=dict(color=col, dash=dash, width=w)))
for i, arm in enumerate(e6["arms"]):
    fig.add_trace(go.Scatter(x=mm.columns, y=mm.loc[arm], name=f"GNN · {R.GNN_PRETTY[arm]}",
                             mode="lines+markers",
                             line=dict(color=[T.INK, T.CYAN, T.BLUE, T.SLATE, T.EMERALD, T.PINK][i],
                                       width=2.2)))
fig.update_layout(height=420, xaxis=dict(title="horizon", tickvals=D.HORIZONS),
                  yaxis_title="MAE (wILI points)", legend=dict(font=dict(size=10.5)))
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

rows = [("GNN no-graph vs B0 persistence", R.e6_contrast(e6, "none", "B0 persistence")),
        ("GNN no-graph vs B2 boosting", R.e6_contrast(e6, "none", "B2 boosting"))]
for a in [x for x in e6["arms"] if x != "none"]:
    rows.append((f"{R.GNN_PRETTY[a]} vs no graph", R.e6_contrast(e6, a, "none")))
for a in ["corr", "climate", "mass"]:
    if a in e6["arms"]:
        rows.append((f"{R.GNN_PRETTY[a]} vs RANDOM (G5)", R.e6_contrast(e6, a, "random")))
st.plotly_chart(matrix_fig(rows), width="stretch", config={"displayModeBar": False})
nb0 = R.e6_contrast(e6, "none", "B0 persistence")
ok = [m for h, (m, k) in nb0.items() if k and m > 0]
rel = [(lab, h, m) for lab, cells in rows[2:] for h, (m, k) in cells.items() if k]
n_rel = sum(len(c) for _, c in rows[2:])
pos = [x for x in rel if x[2] > 0]
detail = ("; ".join(f"{lab} at h={h}: {m:+.1f}%" for lab, h, m in rel)) if rel else ""
T.callout(f"<b>The instrument works</b> — the GNN beats persistence sign-consistently at "
          f"{len(ok)} of {len(nb0)} horizons ({min(ok):+.1f}% to {max(ok):+.1f}%). "
          f"<b>The relations do not</b> — {len(rel)} of {n_rel} relation-contrast cells are "
          f"sign-consistent, and {'all' if rel and not pos else len(pos)} of those "
          f"{'are losses' if rel and not pos else 'are gains'}"
          + (f" ({detail})" if detail else "") + ". Removing instability with fixed exogenous "
          "graphs moves point estimates but not the verdict: stability is necessary, not "
          "sufficient, at ten nodes.", T.VIOLET)
T.provenance("data/reports/graph_gnn_ablation.csv (pre-COVID folds)", "graph_stability.csv")
