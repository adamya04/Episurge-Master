import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import data as D
from core import results as R
from core import theme as T
from core.paths import ROOT

import run_pipeline                                               # noqa: E402

T.hero(
    "Architecture",
    "A pipeline built so the past <span class='es-grad'>cannot see the future</span>",
    "Five public sources become one region-week panel in which every model "
    "input is reconstructed exactly as it was published on the forecast date. "
    "The pipeline below is drawn from the stage manifest the project actually "
    "runs (<code>src/tools/run_pipeline.py</code>), and each stage's status is "
    "checked against the files on disk as this page loads.",
)

# ── stage DAG ────────────────────────────────────────────────────────
T.section("Execution graph", f"{len(run_pipeline.STAGES)} stages, one dependency order",
          "Press play to watch data flow group by group. A stage is drawn solid "
          "when every input it declares exists in this checkout; hollow stages "
          "need the raw data pull (gitignored), which is why ingestion is "
          "excluded from a default run.")

GROUP_COL = {"ingestion": T.SLATE, "preprocessing": T.BLUE, "features": T.VIOLET,
             "join": T.AMBER, "analysis": T.CYAN}
stages = run_pipeline.STAGES
G = nx.DiGraph()
for name, script, group, ins, outs in stages:
    ready = all((ROOT / p).exists() for p in ins)
    G.add_node(name, group=group, script=script, ins=ins, outs=outs, ready=ready)
for a in stages:
    for b in stages:
        if a[0] == b[0]:
            continue
        if any(i == o or i.startswith(o.rstrip("/") + "/") or o.startswith(i.rstrip("/") + "/")
               for o in a[4] for i in b[3]):
            G.add_edge(a[0], b[0])

groups = run_pipeline.GROUPS
pos = {}
for gi, grp in enumerate(groups):
    names = [n for n in G.nodes if G.nodes[n]["group"] == grp]
    for k, n in enumerate(names):
        pos[n] = (gi, (len(names) - 1) / 2 - k)

ex, ey = [], []
for a, b in G.edges:
    (x0, y0), (x1, y1) = pos[a], pos[b]
    xm = (x0 + x1) / 2
    ex += [x0, xm, xm, x1, None]
    ey += [y0, y0, y1, y1, None]


def node_trace(active_groups):
    xs, ys, cols, sizes, text, hov, lines = [], [], [], [], [], [], []
    for n in G.nodes:
        a = G.nodes[n]
        on = a["group"] in active_groups
        xs.append(pos[n][0])
        ys.append(pos[n][1])
        c = GROUP_COL[a["group"]]
        cols.append(c if (a["ready"] and on) else "rgba(14,21,34,1)")
        lines.append(c if on else T.LINE)
        sizes.append(20 if on else 14)
        text.append(n)
        hov.append(f"<b>{n}</b><br>{a['script']}<br>status: "
                   f"{'ready' if a['ready'] else 'blocked (needs data/raw)'}"
                   f"<br>in: {', '.join(a['ins']) or '—'}<br>out: "
                   f"{', '.join(a['outs']) or '—'}")
    return go.Scatter(x=xs, y=ys, mode="markers+text", text=text,
                      textposition="middle right",
                      textfont=dict(size=10.5, color=T.INK),
                      marker=dict(size=sizes, color=cols,
                                  line=dict(width=2, color=lines)),
                      hovertext=hov, hoverinfo="text", showlegend=False)


edge = go.Scatter(x=ex, y=ey, mode="lines", hoverinfo="skip", showlegend=False,
                  line=dict(color="rgba(138,155,181,0.35)", width=1.2))
frames = [go.Frame(data=[edge, node_trace(groups[:i + 1])], name=str(i))
          for i in range(len(groups))]
fig = go.Figure(data=[edge, node_trace(groups)], frames=frames)
for gi, grp in enumerate(groups):
    fig.add_annotation(x=gi, y=max(p[1] for p in pos.values()) + 1.1, text=grp.upper(),
                       showarrow=False, font=dict(color=GROUP_COL[grp], size=11,
                                                  family=T.MONO))
fig.update_layout(
    height=620, xaxis=dict(visible=False, range=[-0.3, len(groups) + 0.2]),
    yaxis=dict(visible=False), margin=dict(l=0, r=0, t=30, b=0),
    updatemenus=[dict(type="buttons", x=0, y=-0.02, xanchor="left",
                      bgcolor=T.CYAN, bordercolor=T.CYAN, font=dict(color="#070B13", size=12), showactive=False,
                      buttons=[dict(label="▶  Play data flow", method="animate",
                                    args=[None, dict(frame=dict(duration=900,
                                                                redraw=True),
                                                     fromcurrent=False,
                                                     transition=dict(duration=300))])])])
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
n_ready = sum(1 for n in G.nodes if G.nodes[n]["ready"])
an = [n for n in G.nodes if G.nodes[n]["group"] == "analysis"]
an_ready = sum(1 for n in an if G.nodes[n]["ready"])
blocked_an = [n for n in an if not G.nodes[n]["ready"]]
T.callout(f"<b>{n_ready} of {len(G.nodes)}</b> stages are runnable in this checkout, "
          f"including <b>{an_ready} of {len(an)}</b> analysis stages"
          + (f" (blocked: <code>{', '.join(blocked_an)}</code>, which rebuilds a feature "
             f"table that is itself committed)" if blocked_an else "")
          + ". Every experiment reproduces from a clone with "
          "<code>python src/tools/run_pipeline.py --only analysis</code>.", T.EMERALD)
T.provenance("src/tools/run_pipeline.py · STAGES", "file existence checked live")

# ── data funnel ──────────────────────────────────────────────────────
T.section("From GenBank to features", "The sequence funnel",
          "Every HA record retrieved from NCBI passes the same gates. The counts "
          "are read from the pipeline's own QC and mapping summaries.")

qc = D.processed_json("qc_summary.json")
mp = D.processed_json("mapping_summary.json")
ds = R.e1()["delay_summary"]
sel = st.selectbox("Season", ["All seasons"] + D.SEASONS, key="funnel_season")
ss = D.SEASONS if sel == "All seasons" else [sel]
tot = sum(qc[s]["total"] for s in ss)
kept = sum(qc[s]["kept"] for s in ss)
mapped = sum(mp[s]["mapped"] for s in ss)
dated = int(ds.loc[ss, "n_sequences"].sum())
c1, c2 = st.columns([1.15, 1])
with c1:
    fig = go.Figure(go.Funnel(
        y=["Retrieved from NCBI", "Passed QC", "Mapped to a US state", "Publication date recovered"],
        x=[tot, kept, mapped, dated], texttemplate="%{value:,}<br>%{percentInitial:.1%}",
        marker=dict(color=[T.SLATE, T.BLUE, T.VIOLET, T.AMBER]),
        connector=dict(line=dict(color=T.LINE))))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
with c2:
    rej = {}
    for s in ss:
        for k, v in qc[s]["rejected_breakdown"].items():
            rej[k] = rej.get(k, 0) + v
    rr = pd.Series(rej).sort_values()
    fig = go.Figure(go.Bar(x=rr.values, y=[k.replace("_", " ") for k in rr.index],
                           orientation="h", marker_color=T.ROSE,
                           text=[f"{v:,}" for v in rr.values], textposition="auto"))
    fig.update_layout(height=340, title="Records failing each QC check",
                      xaxis_title="records (a record may fail several checks)")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
T.provenance("data/processed/qc_summary.json", "mapping_summary.json",
             "data/reports/delay_summary.csv")

# ── backfill ─────────────────────────────────────────────────────────
T.section("The as-of principle", "What the forecaster actually saw",
          "Surveillance numbers are revised for weeks after first publication. "
          "At an origin week T the model receives the eight most recent weeks "
          "<i>as published at T</i>; only the target uses the final revised "
          "value. Drag the origin to see the gap between the two.")

full = D.master_full().sort_values(["hhs_region", "epiweek"]).copy()
c1, c2 = st.columns([1, 3])
with c1:
    season = st.selectbox("Season", D.SEASONS, index=len(D.SEASONS) - 1, key="bf_season")
    region = st.selectbox("Region", D.REGIONS, format_func=D.region_label,
                          index=3, key="bf_region")
g = full[(full.season == season) & (full.hhs_region == region)].reset_index(drop=True)
weeks = g.epiweek.tolist()
with c2:
    T_idx = st.select_slider("Forecast origin (epiweek T)", options=list(range(8, len(weeks))),
                             value=min(30, len(weeks) - 1),
                             format_func=lambda i: str(weeks[i]), key="bf_origin")
row = g.iloc[T_idx]
lag_weeks = [g.epiweek.iloc[T_idx - k] for k in range(1, 9)]
asof = [row[f"ili_asof_lag{k}"] for k in range(1, 9)]
final = [g.wili_final.iloc[T_idx - k] for k in range(1, 9)]
xs = list(range(len(g)))
fig = go.Figure()
fig.add_trace(go.Scatter(x=xs, y=g.wili_final, name="final revised wILI",
                         line=dict(color=T.SLATE, width=1.5),
                         customdata=g.epiweek, hovertemplate="%{customdata}: %{y:.2f}<extra></extra>"))
fig.add_trace(go.Scatter(x=[T_idx - k for k in range(1, 9)], y=final, mode="markers",
                         name="final value of each input week",
                         marker=dict(color=T.CYAN, size=9, symbol="circle-open", line=dict(width=2))))
fig.add_trace(go.Scatter(x=[T_idx - k for k in range(1, 9)], y=asof, mode="lines+markers",
                         name="as published at T (model input)",
                         line=dict(color=T.AMBER, width=3), marker=dict(size=8)))
fig.add_vline(x=T_idx, line=dict(color=T.ROSE, dash="dash"),
              annotation_text=f"origin T = {weeks[T_idx]}",
              annotation_font=dict(color=T.ROSE))
fig.update_layout(height=380, hovermode="x unified", yaxis_title="wILI (%)",
                  xaxis=dict(title="week of season", tickmode="array",
                             tickvals=xs[::4], ticktext=[str(w) for w in weeks[::4]]),
                  legend=dict(orientation="h", y=-0.22))
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

# revision magnitude across the panel, computed live
p = full.copy()
p["final_prev"] = p.groupby(["season", "hhs_region"])["wili_final"].shift(1)
p["rev"] = p["ili_asof_lag1"] - p["final_prev"]
p = p[p.in_core_season == True].dropna(subset=["rev"])          # noqa: E712
cp = D.core_panel()[["season", "hhs_region", "epiweek", "weeks_from_peak"]]
p = p.merge(cp, on=["season", "hhs_region", "epiweek"], how="inner")
b = p.groupby("weeks_from_peak")["rev"].agg(["mean", lambda s: s.abs().mean(), "size"])
b.columns = ["signed", "absolute", "n"]
b = b[b.n >= 20]
c1, c2, c3 = st.columns(3)
c1.metric("Mean |revision| of the latest week", f"{p.rev.abs().mean():.3f} pp")
c2.metric("Share first published too low", f"{100 * (p.rev < 0).mean():.1f}%")
c3.metric("Largest single revision", f"{p.rev.abs().max():.2f} pp")
fig = go.Figure()
fig.add_trace(go.Bar(x=b.index, y=b.absolute, name="mean |as-of − final|",
                     marker_color=T.AMBER))
fig.add_trace(go.Scatter(x=b.index, y=b.signed, name="mean signed revision",
                         line=dict(color=T.CYAN, width=2)))
fig.add_vline(x=0, line=dict(color=T.MUTED, dash="dash"))
wk = int(b.absolute.idxmax())
near = b.absolute[b.index.to_series().abs() <= 2].mean()
rest = b.absolute[b.index.to_series().abs() > 2].mean()
fig.update_layout(height=300, xaxis_title="weeks from national peak",
                  yaxis_title="percentage points", legend=dict(orientation="h", y=-0.3),
                  title=f"Absolute backfill peaks {abs(wk)} week{'s' if abs(wk) != 1 else ''} "
                        f"{'before' if wk < 0 else 'after'} the peak — "
                        f"{near / rest:.1f}× the off-peak average within ±2 weeks")
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
st.caption("Absolute revisions partly track the level: wILI is highest near the peak, "
           "so a constant relative revision would also grow there. The point for "
           "forecasting is the absolute one — the inputs are least reliable in the "
           "weeks a forecast matters most.")
T.provenance("data/processed/master_region_week.csv · ili_asof_lag1 vs wili_final of week T−1")

# ── standing rules ───────────────────────────────────────────────────
T.section("Guard rails", "The standing rules every experiment obeys",
          "Read from DECISIONS.md. Each one was adopted after an experiment "
          "showed what goes wrong without it.")
import re                                                         # noqa: E402
txt = (ROOT / "DECISIONS.md").read_text(encoding="utf-8")
blk = txt.split("## Standing rules", 1)[1]
rules = re.findall(r"^\d+\. \*\*(.+?)\*\*(.*?)(?=^\d+\. |\Z)", blk, flags=re.S | re.M)
cols = st.columns(2)
for i, (head, body) in enumerate(rules):
    body = " ".join(body.split())
    with cols[i % 2]:
        T.callout(f"<b>{i + 1}. {T.esc(head)}</b> {T.esc(body)}",
                  [T.CYAN, T.VIOLET, T.AMBER, T.EMERALD, T.ROSE, T.BLUE, T.PINK][i % 7])
T.provenance("DECISIONS.md · Standing rules")
