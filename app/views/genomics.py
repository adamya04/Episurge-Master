import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from core import data as D
from core import geo
from core import results as R
from core import theme as T

T.hero(
    "Genomic surveillance · E1",
    "Every sequence has two dates. <span class='es-grad'>Only one of them is usable.</span>",
    "A sequence is <i>collected</i> from a patient, then weeks or months later "
    "<i>published</i> to GenBank. Research usually indexes genomes by collection "
    "date — the oracle view. A forecaster on that date could only use what had "
    "been published. This page measures the gap between the two.",
)

seqs = D.sequences()
e1 = R.e1()
sub_ok = seqs[seqs.subtype.isin(["H3N2", "H1N1"])]

T.cards([
    ("Mapped HA sequences", f"{len(seqs):,}", f"{seqs.state.nunique()} states and territories", T.VIOLET),
    ("H3N2 : H1N1", f"{(sub_ok.subtype == 'H3N2').mean() * 100:.0f} : {(sub_ok.subtype == 'H1N1').mean() * 100:.0f}",
     f"{(~seqs.subtype.isin(['H3N2', 'H1N1'])).sum()} other / unparsed", T.CYAN),
    ("Median delay", f"{e1['median_delay_days']:.0f} days",
     f"{e1['pct_over_12wk']:.1f}% arrive more than 12 weeks late", T.AMBER),
    ("As-of coverage, pre-peak", f"{e1['asof_pre']:.1f}%",
     f"vs {e1['asof_post']:.1f}% from the peak on", T.ROSE),
], cols=4)

# ── time machine ─────────────────────────────────────────────────────
T.section("Time machine", "What could you have known on a given day?",
          "Pick a date. Solid bars are sequences already public on that date; "
          "ghosted bars were collected by then but published later. The ghosted "
          "mass is the information an oracle-dated study silently uses.")
c1, c2 = st.columns([1, 3])
with c1:
    season = st.selectbox("Season", D.SEASONS, index=D.SEASONS.index("2022-23"), key="tm_season")
ss = sub_ok[sub_ok.season == season].dropna(subset=["collection_date"]).copy()
ss["cweek"] = ss.collection_date.dt.to_period("W-SAT").dt.start_time
weeks = sorted(ss.cweek.unique())
with c2:
    default = weeks[int(len(weeks) * 0.45)]
    today = st.select_slider("Today is", options=weeks, value=default, key="tm_today",
                             format_func=lambda d: pd.Timestamp(d).strftime("%d %b %Y"))
today = pd.Timestamp(today)
ss = ss[ss.cweek <= today + pd.Timedelta(days=7 * 12)]
fig = go.Figure()
for st_, col in (("H3N2", T.VIOLET), ("H1N1", T.CYAN)):
    g = ss[ss.subtype == st_]
    pub = g[g.submission_date <= today].groupby("cweek").size()
    late = g[g.submission_date > today].groupby("cweek").size()
    fig.add_trace(go.Bar(x=pub.index, y=pub.values, name=f"{st_} · public by then",
                         marker_color=col))
    fig.add_trace(go.Bar(x=late.index, y=late.values, name=f"{st_} · published later",
                         marker=dict(color=col, opacity=0.18, line=dict(color=col, width=1))))
fig.add_vline(x=today, line=dict(color=T.AMBER, width=2, dash="dash"))
fig.add_annotation(x=today, y=1, yref="paper", text="today", showarrow=False,
                   font=dict(color=T.AMBER), xanchor="left", yanchor="top")
fig.update_layout(barmode="stack", height=420, xaxis_title="collection week",
                  yaxis_title="sequences", legend=dict(orientation="h", y=-0.2))
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

col = sub_ok[(sub_ok.season == season) & (sub_ok.collection_date <= today)]
known = col[col.submission_date <= today]
last4 = col[col.collection_date > today - pd.Timedelta(days=28)]
last4k = last4[last4.submission_date <= today]
true_share = (last4.subtype == "H3N2").mean() * 100 if len(last4) else np.nan
seen_share = (last4k.subtype == "H3N2").mean() * 100 if len(last4k) else np.nan
c1, c2, c3, c4 = st.columns(4)
c1.metric("Collected by today", f"{len(col):,}")
c2.metric("Public by today", f"{len(known):,}",
          f"{100 * len(known) / max(len(col), 1):.0f}% of collected", delta_color="off")
c3.metric("Last 4 weeks: public / collected", f"{len(last4k)} / {len(last4)}")
c4.metric("H3N2 share · last 4 wk",
          "—" if seen_share != seen_share else f"{seen_share:.0f}%",
          None if true_share != true_share else f"true {true_share:.0f}%", delta_color="off")
T.provenance("data/processed/mapped_*.json (collection, subtype)",
             "data/reports/delay_distribution.csv (GenBank publication date)")

# ── map ──────────────────────────────────────────────────────────────
T.section("Where sequencing happens", "Sequencing effort is as uneven as it is late",
          "Column height is the number of sequences from each state; colour is "
          "its H3N2 share. Composition features are built at HHS-region level, so "
          "a region's signal can rest on a single state's laboratories.")
c1, c2 = st.columns([1, 3])
with c1:
    msel = st.selectbox("Season", ["All seasons"] + D.SEASONS, key="map_season")
    ms = sub_ok if msel == "All seasons" else sub_ok[sub_ok.season == msel]
    agg = ms.groupby("state").agg(n=("accession", "size"),
                                  h3=("subtype", lambda s: (s == "H3N2").mean())).reset_index()
    top = agg.sort_values("n", ascending=False).head(8)
    st.markdown("**Most-sequenced states**")
    st.dataframe(pd.DataFrame({"state": top.state, "sequences": top.n,
                               "H3N2": (100 * top.h3).round(0).astype(int).astype(str) + "%"}),
                 hide_index=True, width="stretch")
    st.caption(f"Top state holds {100 * top.n.iloc[0] / agg.n.sum():.0f}% of the selection.")
cents = geo.state_centroids()
agg = agg[agg.state.isin(cents) & ~agg.state.isin(geo.NON_CONTIGUOUS)].copy()
agg["lon"] = agg.state.map(lambda s: cents[s][0])
agg["lat"] = agg.state.map(lambda s: cents[s][1])
agg["fill"] = agg.h3.map(lambda v: geo.ramp(v, 0, 1, ["#22D3EE", "#6366F1", "#A78BFA", "#F472B6"]) + [235])
agg["pct"] = (100 * agg.h3).round(0)
agg["region"] = agg.state.map(dict(D.state_region_table().values)).map(D.region_label)
base = geo.contiguous_geojson()
scale = 900000 / max(float(agg.n.max()), 1)
with c2:
    st.pydeck_chart(pdk.Deck(
        layers=[
            pdk.Layer("GeoJsonLayer", base, stroked=True, filled=True,
                      get_fill_color=[19, 28, 45, 255], get_line_color=[56, 72, 100, 255],
                      line_width_min_pixels=1),
            pdk.Layer("ColumnLayer", agg, get_position=["lon", "lat"], get_elevation="n",
                      elevation_scale=scale, radius=45000, get_fill_color="fill",
                      pickable=True, auto_highlight=True),
        ],
        initial_view_state=pdk.ViewState(latitude=37.5, longitude=-96, zoom=3.1,
                                          pitch=52, bearing=-10),
        map_style=geo.OFFLINE_BASEMAP, map_provider="carto",
        tooltip={"html": "<b>{state}</b> · {region}<br/>{n} sequences<br/>H3N2 {pct}%",
                 "style": {"backgroundColor": "#131C2D", "color": "#E6EDF7",
                           "fontSize": "12px"}}), height=520)
T.provenance("data/processed/mapped_*.json · state, subtype")

# ── delay ────────────────────────────────────────────────────────────
T.section("Publication delay", "How late, by season and by subtype",
          "The shaded bands are the two lag windows the as-of features are built "
          "from. A sequence published after its window closes is invisible to the "
          "model at that origin.")
d = sub_ok.dropna(subset=["delay_days"]).copy()
d["weeks"] = d.delay_days / 7
split = st.toggle("Split by subtype", value=True, key="delay_split")
fig = go.Figure()
fig.add_hrect(y0=4, y1=12, fillcolor="rgba(52,211,153,0.07)", line_width=0,
              annotation_text="L4-12", annotation_position="top left",
              annotation_font=dict(color=T.EMERALD, size=10))
fig.add_hrect(y0=8, y1=16, fillcolor="rgba(245,165,36,0.07)", line_width=0,
              annotation_text="L8-16", annotation_position="top right",
              annotation_font=dict(color=T.AMBER, size=10))
if split:
    for st_, col, side in (("H3N2", T.VIOLET, "negative"), ("H1N1", T.CYAN, "positive")):
        g = d[d.subtype == st_]
        fig.add_trace(go.Violin(x=g.season, y=g.weeks.clip(upper=52), name=st_, side=side,
                                line_color=col, fillcolor=col, opacity=0.55,
                                meanline_visible=True, points=False, spanmode="hard"))
    fig.update_layout(violingap=0, violinmode="overlay")
else:
    fig.add_trace(go.Violin(x=d.season, y=d.weeks.clip(upper=52), line_color=T.AMBER,
                            fillcolor=T.AMBER, opacity=0.5, meanline_visible=True,
                            points=False, name="all", spanmode="hard"))
fig.update_layout(height=430, yaxis_title="weeks from collection to publication (clipped at 52)",
                  xaxis=dict(categoryorder="array", categoryarray=D.SEASONS))
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
med = d.groupby(["season", "subtype"]).delay_days.median().unstack()
slower = int((med.H3N2 > med.H1N1).sum())
T.callout(f"H3N2 is published more slowly than H1N1 in <b>{slower} of {len(med)}</b> seasons "
          f"(median gap {(med.H3N2 - med.H1N1).mean():+.1f} days). The bias is real and "
          f"directional — and was tested as a correction, then rejected on magnitude "
          f"(see Contributions → what we killed).", T.VIOLET)
T.provenance("data/reports/delay_distribution.csv", "mapped_*.json · subtype")

# ── coverage vs peak (E1) ────────────────────────────────────────────
T.section("E1 result", "Coverage collapses exactly where forecasts matter")
cp = D.report("coverage_by_peak_offset.csv")
cp = cp[cp.n_rows >= 20]
fig = go.Figure()
for colname, name, c, dash in (("oracle_pct", "oracle (collection-dated)", T.VIOLET, "dot"),
                               ("asof_L8_16_pct", "as-of, L8-16 window", T.AMBER, "solid"),
                               ("asof_L4_12_pct", "as-of, L4-12 window", T.EMERALD, "dash")):
    fig.add_trace(go.Scatter(x=cp.weeks_from_peak, y=cp[colname], name=name,
                             line=dict(color=c, width=3, dash=dash)))
fig.add_vline(x=0, line=dict(color=T.MUTED, dash="dash"))
fig.update_layout(height=380, xaxis_title="weeks from national peak",
                  yaxis_title="% of region-weeks with composition", yaxis_range=[0, 102],
                  legend=dict(orientation="h", y=-0.22), hovermode="x unified")
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
cov = e1["coverage"]
T.cards([
    ("Oracle, pre-peak", f"{cov.loc['oracle', 'pre-peak']:.1f}%", "if publication were instant", T.VIOLET),
    ("As-of L8-16, pre-peak", f"{cov.loc['asof_L8_16', 'pre-peak']:.1f}%", "what the model actually gets", T.AMBER),
    ("As-of L8-16, peak on", f"{cov.loc['asof_L8_16', 'peak or after']:.1f}%", "plenty — when it no longer matters", T.SLATE),
    ("As-of L4-12, pre-peak", f"{cov.loc['asof_L4_12', 'pre-peak']:.1f}%", "a tighter window barely helps alone", T.EMERALD),
], cols=4)
T.provenance("data/reports/coverage_by_peak_offset.csv", "coverage_by_peak_side.csv (row-weighted)")

# ── strain explorer ──────────────────────────────────────────────────
T.section("Strain explorer", "Every record, searchable",
          "The raw material. No sequence bodies were retained — only GenBank "
          "metadata — which is why the study is bounded to subtype composition "
          "and why clade-level or phylogenetic analysis is listed as future work "
          "rather than shown here.")
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    fs = st.multiselect("Season", D.SEASONS, default=[D.SEASONS[-1]], key="se_season")
with c2:
    ft = st.multiselect("Subtype", ["H3N2", "H1N1", "other", "unknown"],
                        default=["H3N2", "H1N1"], key="se_sub")
with c3:
    q = st.text_input("Search strain or state", "", key="se_q",
                      placeholder="e.g. Texas or A/California")
t = seqs[seqs.season.isin(fs or D.SEASONS) & seqs.subtype.isin(ft or ["H3N2", "H1N1"])]
if q:
    t = t[t.strain.fillna("").str.contains(q, case=False, regex=False)
          | t.state.fillna("").str.contains(q, case=False, regex=False)]
view = t[["accession", "strain", "subtype", "state", "hhs_region", "collection_date",
          "submission_date", "delay_days"]].sort_values("collection_date")
st.dataframe(view.head(2000), hide_index=True, height=360, width="stretch",
             column_config={
                 "collection_date": st.column_config.DateColumn("collected"),
                 "submission_date": st.column_config.DateColumn("published"),
                 "delay_days": st.column_config.ProgressColumn(
                     "delay (days)", min_value=0, max_value=365, format="%d"),
                 "hhs_region": st.column_config.NumberColumn("region", format="%d")})
st.caption(f"{len(view):,} records match" + (" — showing the first 2,000" if len(view) > 2000 else ""))
T.provenance("data/processed/mapped_*.json", "data/reports/delay_distribution.csv")
