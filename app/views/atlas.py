import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

from core import data as D
from core import geo
from core import theme as T

T.hero(
    "Epidemic atlas",
    "Six seasons of influenza across <span class='es-grad'>ten HHS regions</span>",
    "Height is incidence. Colour is whatever you choose to see — including the "
    "uncomfortable view: the genomic composition a forecaster could actually "
    "have known that week. Dark regions are not low; they are <i>unknown</i>.",
)

core = D.core_panel()
# Plotly's built-in "white-bg" style needs no network; a world-sized dark
# fill drawn beneath the traces gives it the app's background. (An inline
# style object renders, but raises "Map error" on every animation frame.)
DARK_FILL = {"type": "FeatureCollection", "features": [{
    "type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [
        [[-179, -85], [179, -85], [179, 85], [-179, 85], [-179, -85]]]}}]}

# ── controls ─────────────────────────────────────────────────────────
c1, c2, c3 = st.columns([1, 2.2, 1.6])
with c1:
    season = st.selectbox("Season", D.SEASONS, index=D.SEASONS.index("2017-18"),
                          key="atlas_season")
S = core[core.season == season]
idxs = sorted(S.season_week_idx.unique())
lab = S.drop_duplicates("season_week_idx").set_index("season_week_idx")
peak_idx = int(lab.index[lab.weeks_from_peak == 0][0])
with c2:
    w = st.select_slider(
        "Week of season", options=idxs, value=peak_idx - 3, key="atlas_week",
        format_func=lambda i: f"{lab.loc[i, 'epiweek']} ({int(lab.loc[i, 'weeks_from_peak']):+d} wk)")
with c3:
    mode = st.radio("Colour by", ["Incidence", "H3N2 share · true", "H3N2 share · knowable then"],
                    horizontal=False, key="atlas_mode", index=2)

W = S[S.season_week_idx == w].set_index("hhs_region")
vmax = float(S.wili_final.max())


def colour(r):
    if mode == "Incidence":
        return geo.ramp(W.loc[r, "wili_final"], 0, vmax,
                        ["#0B1424", "#1E1B4B", "#6D28D9", "#DB2777", "#F59E0B", "#FEF3C7"])
    col = "oracle_prop_h3n2" if mode.endswith("true") else "asof_L8_16_prop_h3n2"
    v = W.loc[r, col]
    if v != v:
        return [26, 32, 46]
    return geo.ramp(v, 0, 1, ["#22D3EE", "#6366F1", "#A78BFA", "#F472B6"])


gj = geo.contiguous_geojson()
for f in gj["features"]:
    r = f["properties"]["hhs_region"]
    f["properties"].update(
        region=D.region_label(r), wili=round(float(W.loc[r, "wili_final"]), 2),
        h3_true="—" if W.loc[r, "oracle_prop_h3n2"] != W.loc[r, "oracle_prop_h3n2"]
        else f"{100 * W.loc[r, 'oracle_prop_h3n2']:.0f}%",
        h3_asof="no data yet" if W.loc[r, "asof_L8_16_prop_h3n2"] != W.loc[r, "asof_L8_16_prop_h3n2"]
        else f"{100 * W.loc[r, 'asof_L8_16_prop_h3n2']:.0f}%",
        fill=colour(r) + [235], elev=float(W.loc[r, "wili_final"]) * 55000)

deck = pdk.Deck(
    layers=[pdk.Layer("GeoJsonLayer", gj, extruded=True, wireframe=True, pickable=True,
                      get_elevation="properties.elev", get_fill_color="properties.fill",
                      get_line_color=[7, 11, 19, 255], line_width_min_pixels=1,
                      material={"ambient": 0.55, "diffuse": 0.6, "shininess": 40})],
    initial_view_state=pdk.ViewState(latitude=35.5, longitude=-97.5, zoom=2.85,
                                     pitch=45, bearing=-6),
    map_style=geo.OFFLINE_BASEMAP, map_provider="carto",
    tooltip={"html": "<b>{name}</b> · {region}<br/>wILI {wili}%<br/>H3N2 true {h3_true}"
                     "<br/>H3N2 knowable then {h3_asof}",
             "style": {"backgroundColor": "#131C2D", "color": "#E6EDF7",
                       "fontFamily": "Inter, sans-serif", "fontSize": "12px"}},
)
m1, m2 = st.columns([3.2, 1])
with m1:
    st.pydeck_chart(deck, height=560)
with m2:
    nat = float(W.wili_final.mean())
    known = int(W.asof_L8_16_prop_h3n2.notna().sum())
    truth = int(W.oracle_prop_h3n2.notna().sum())
    T.cards([("Week", str(int(lab.loc[w, "epiweek"])),
              f"{int(lab.loc[w, 'weeks_from_peak']):+d} weeks from national peak", T.CYAN),
             ("Mean regional wILI", f"{nat:.2f}%", f"season max {vmax:.2f}%", T.PINK),
             ("Regions with genome, as published", f"{known}/10",
              f"{truth}/10 once every sequence arrives", T.AMBER)], cols=1)
    st.caption("Drag to rotate (right-click or Ctrl-drag), scroll to zoom. "
               "Height = wILI. Hover a state for its region's values.")
T.provenance("data/processed/master_region_week.csv", "app/assets/us_states.geojson (us-atlas 3.0.1)")

# ── animated choropleth ──────────────────────────────────────────────
T.section("Animation", "Watch the season move",
          "Every state is coloured by its HHS region's weighted ILI, week by week. "
          "Press play; the slider scrubs.")
# Frames carry only the changing z values; the state geometry is sent once.
# (plotly.express repeats the full GeoJSON in every frame: ~6 MB for 33 weeks.)
states = [f["properties"]["name"] for f in gj["features"]]
st_reg = [f["properties"]["hhs_region"] for f in gj["features"]]
by_week = S.pivot_table(index="season_week_idx", columns="hhs_region", values="wili_final")
wk_lab = {i: f"{int(lab.loc[i, 'epiweek'])} ({int(lab.loc[i, 'weeks_from_peak']):+d})" for i in idxs}


def z_for(i):
    return [round(float(by_week.loc[i, r]), 3) for r in st_reg]


base = go.Choroplethmap(geojson=gj, locations=states, featureidkey="properties.name",
                        z=z_for(idxs[0]), zmin=0, zmax=vmax, colorscale=T.SEQ_HEAT,
                        marker=dict(opacity=0.95, line=dict(color="#070B13", width=0.6)),
                        customdata=[D.region_label(r) for r in st_reg],
                        hovertemplate="%{location} · %{customdata}<br>wILI %{z:.2f}%<extra></extra>",
                        colorbar=dict(title="wILI %"))
frames = [go.Frame(name=wk_lab[i], data=[go.Choroplethmap(z=z_for(i))], traces=[0]) for i in idxs]
fig = go.Figure(data=[base], frames=frames)
fig.update_layout(
    map=dict(style="white-bg", center=dict(lat=38.5, lon=-96.5), zoom=2.75,
             layers=[dict(sourcetype="geojson", source=DARK_FILL, type="fill",
                          color="#070B13", below="traces")]),
    height=540, margin=dict(l=0, r=0, t=10, b=0),
    updatemenus=[dict(type="buttons", x=0.0, y=-0.02, xanchor="left", yanchor="top",
                      bgcolor=T.CYAN, bordercolor=T.CYAN, showactive=False,
                      font=dict(color="#070B13", size=12),
                      buttons=[dict(label="▶  Play season", method="animate",
                                    args=[None, dict(frame=dict(duration=260, redraw=True),
                                                     transition=dict(duration=0),
                                                     fromcurrent=True)]),
                               dict(label="❚❚  Pause", method="animate",
                                    args=[[None], dict(mode="immediate",
                                                       frame=dict(duration=0, redraw=False))])])],
    sliders=[dict(active=0, x=0.2, len=0.78, y=-0.02, yanchor="top",
                  currentvalue=dict(prefix="week ", font=dict(color=T.INK)),
                  font=dict(color=T.MUTED, size=10),
                  steps=[dict(label=wk_lab[i], method="animate",
                              args=[[wk_lab[i]], dict(mode="immediate",
                                                      frame=dict(duration=0, redraw=True))])
                         for i in idxs])])
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
T.provenance("data/processed/master_region_week.csv · wili_final")

# ── tape: incidence vs knowable genome ───────────────────────────────
T.section("Spatiotemporal tape", "Incidence and the knowable genome, side by side",
          "Top: weighted ILI by region and week. Bottom: whether any as-of "
          "genomic composition existed for that region that week. The hole in "
          "the bottom panel sits under the rising limb of the top one.")
piv = S.pivot_table(index="hhs_region", columns="season_week_idx", values="wili_final")
# The presence flag is computed BEFORE pivoting. pivot_table drops columns
# that are entirely NaN - precisely the early weeks with no genome - which
# would shift every later column under the wrong week label.
gen = (S.assign(present=S.asof_L8_16_prop_h3n2.notna().astype(int))
       .pivot_table(index="hhs_region", columns="season_week_idx", values="present")
       .reindex(index=piv.index, columns=piv.columns))
assert gen.shape == piv.shape
xl = [int(lab.loc[i, "weeks_from_peak"]) for i in piv.columns]
yl = [D.region_label(r) for r in piv.index]
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                    row_heights=[0.62, 0.38])
fig.add_trace(go.Heatmap(z=piv.values, x=xl, y=yl, colorscale=T.SEQ_HEAT,
                         colorbar=dict(title="wILI", len=0.6, y=0.72),
                         hovertemplate="%{y}<br>%{x} wk<br>wILI %{z:.2f}<extra></extra>"), 1, 1)
fig.add_trace(go.Heatmap(z=gen.values, x=xl, y=yl, showscale=False,
                         colorscale=[[0, "#111827"], [1, T.AMBER]], zmin=0, zmax=1,
                         hovertemplate="%{y}<br>%{x} wk<br>%{z}: as-of genome present<extra></extra>"),
              2, 1)
fig.update_layout(height=640, xaxis2=dict(title="weeks from national peak"))
fig.update_yaxes(autorange="reversed")
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
pre = S.weeks_from_peak < 0
T.callout(f"In {season}, as-of composition existed for <b>{100 * S[pre].asof_L8_16_prop_h3n2.notna().mean():.0f}%</b> "
          f"of pre-peak region-weeks and <b>{100 * S[~pre].asof_L8_16_prop_h3n2.notna().mean():.0f}%</b> "
          f"from the peak on.", T.AMBER)
T.provenance("data/processed/master_region_week.csv · wili_final, asof_L8_16_prop_h3n2")

# ── who peaks first ──────────────────────────────────────────────────
T.section("Spread", "Who peaks first — and is it the same every year?",
          "Each cell is a region's peak week relative to the national peak. If "
          "influenza spread along a stable geographic route, the ordering would "
          "repeat across seasons. The agreement score on the right measures how "
          "far it does — this is the epidemiological face of the graph "
          "(in)stability tested in E6.")
pk = []
for (s, r), g in core.groupby(["season", "hhs_region"]):
    i = g.loc[g.wili_final.idxmax()]
    pk.append((s, r, int(i.weeks_from_peak)))
pk = pd.DataFrame(pk, columns=["season", "region", "offset"])
P = pk.pivot_table(index="region", columns="season", values="offset")[D.SEASONS]
c1, c2 = st.columns([1.6, 1])
with c1:
    fig = go.Figure(go.Heatmap(
        z=P.values, x=P.columns, y=[D.region_label(r) for r in P.index],
        colorscale=[[0, T.CYAN], [0.5, "#1A2435"], [1, T.ROSE]], zmid=0,
        text=P.values.astype(int), texttemplate="%{text:+d}",
        colorbar=dict(title="wk vs<br>national"),
        hovertemplate="%{y}<br>%{x}: peak %{z:+d} wk<extra></extra>"))
    fig.update_layout(height=430)
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
with c2:
    rhos = []
    for i, a in enumerate(D.SEASONS):
        for b in D.SEASONS[i + 1:]:
            rho = spearmanr(P[a], P[b]).statistic
            rhos.append((a, b, rho))
    R_ = pd.DataFrame(rhos, columns=["a", "b", "rho"])
    m = R_.rho.mean()
    reading = ("weak — the spread route is largely re-drawn each season" if abs(m) < 0.3
               else "moderate" if abs(m) < 0.6 else "strong — a repeatable route")
    T.cards([("Mean rank agreement", f"{m:+.2f}",
              f"Spearman ρ of regional peak order, averaged over {len(R_)} season pairs "
              f"(range {R_.rho.min():+.2f} to {R_.rho.max():+.2f}): {reading}", T.VIOLET)], cols=1)
    early = (P < 0).sum(axis=1).sort_values(ascending=False)
    st.markdown("**Seasons in which each region peaked before the nation**")
    st.dataframe(pd.DataFrame({"region": [D.region_label(r) for r in early.index],
                               "seasons early": early.values}),
                 hide_index=True, height=260, width="stretch")
T.provenance("computed live from data/processed/master_region_week.csv")

# ── climate ──────────────────────────────────────────────────────────
T.section("Climate context", "The climatology behind the climate graph",
          "Absolute humidity (ERA5) through the season for each region. These "
          "profiles, from training seasons only, define the season-invariant "
          "climate-similarity graph tested in E6.")
clim = core.groupby(["hhs_region", "season_week_idx"])[["abs_humidity", "temp_c"]].mean().reset_index()
var = st.radio("Variable", ["abs_humidity", "temp_c"], horizontal=True, key="clim_var",
               format_func=lambda v: {"abs_humidity": "Absolute humidity (g/m³)",
                                      "temp_c": "Temperature (°C)"}[v])
fig = go.Figure()
for i, r in enumerate(D.REGIONS):
    g = clim[clim.hhs_region == r]
    fig.add_trace(go.Scatter(x=g.season_week_idx, y=g[var], name=D.region_label(r),
                             line=dict(color=T.REGION_COLOURS[i], width=2)))
fig.update_layout(height=380, xaxis_title="week of season (0 = MMWR week 40)",
                  yaxis_title=var.replace("_", " "), legend=dict(font=dict(size=10.5)))
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
T.provenance("data/processed/master_region_week.csv · ERA5-derived temp_c, abs_humidity")
