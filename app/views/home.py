import plotly.graph_objects as go
import streamlit as st

from core import data as D
from core import results as R
from core import theme as T

e1, e2, e3a, e3b = R.e1(), R.e2(), R.e3a(), R.e3b()
e3c, e45, e6, e8 = R.e3c(), R.e45(), R.e6(), R.e8()
core = D.core_panel()
seqs = D.sequences()

T.hero(
    "Final review · Team 18 · 23AID498",
    "Does a virus's genome tell us where influenza goes next "
    "<span class='es-grad'>beyond what case counts already say?</span>",
    "EpiSurge tests one frozen research question on six US influenza seasons: "
    "whether pathogen genomic composition carries forecasting signal for "
    "regional influenza incidence beyond case history — and at which horizons. "
    "Every input is reconstructed <b>as it was knowable at the forecast date</b>, "
    "so the answer is about a system that could have been deployed, not one "
    "that quietly peeks at the future.",
)

n_rw = len(core)
T.cards([
    ("HA sequences mapped", f"{len(seqs):,}",
     f"{seqs.state.nunique()} states → 10 HHS regions", T.VIOLET),
    ("Region-weeks modelled", f"{n_rw:,}",
     f"{core.season.nunique()} seasons × {core.hhs_region.nunique()} regions × "
     f"{core.season_week_idx.nunique()} weeks", T.CYAN),
    ("Median publication delay", f"{e1['median_delay_weeks']:.1f} wk",
     f"collection → GenBank, {e1['n_dated']:,} dated sequences", T.AMBER),
    ("Genomic data available pre-peak", f"{e1['asof_pre']:.1f}%",
     f"as-of, vs {e1['oracle_pre']:.1f}% if delay were zero", T.ROSE),
], cols=4)

# ── the answer ───────────────────────────────────────────────────────
T.section("The answer", "Three findings, each with its evidence attached")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        f"<div class='es-exp'>{T.verdict_pill('BOUNDED NULL')}"
        f"<div class='q'>Subtype composition adds no incidence signal</div>"
        f"<div class='a'>No genomic arm improves a forecast consistently across "
        f"folds at any horizon (E3a). A model-free Granger test that can resolve a "
        f"predictor carrying <b>{e45['floor_var']:.1f}%</b> of variance also finds "
        f"nothing: {e45['n_raw']} of {e45['n_arms']} arms cross p&lt;0.05 against "
        f"{e45['expected']:.1f} expected by chance, {e45['n_bh']} survive correction "
        f"(E4/E5).</div></div>", unsafe_allow_html=True)
with c2:
    st.markdown(
        f"<div class='es-exp'>{T.verdict_pill('POSITIVE')}"
        f"<div class='q'>Publication delay destroys "
        f"{e3b['lo']:.1f}–{e3b['hi']:.1f}% of subtype-dominance forecasting skill</div>"
        f"<div class='a'>Composition <i>does</i> forecast which subtype will "
        f"dominate. Using the data as it was actually published instead of as it "
        f"was collected costs {e3b['lo']:.1f}–{e3b['hi']:.1f}% of that skill, "
        f"sign-consistent at every horizon in every pre-COVID fold, with "
        f"<b>{e3b['n_sig']} of {e3b['n_cells']}</b> fold × horizon intervals "
        f"excluding zero (E3b).</div></div>", unsafe_allow_html=True)
with c3:
    row = R.e3c_row(e3c, "L4-12", 14)
    lo, hi = R.consistent_range(row)
    tight = R.e3c_row(e3c, "L0-4", 0)
    tlo, thi = R.consistent_range(tight)
    st.markdown(
        f"<div class='es-exp'>{T.verdict_pill('POSITIVE')}"
        f"<div class='q'>A two-week faster deposit with a tighter window gains "
        f"{lo:+.1f}% to {hi:+.1f}%</div>"
        f"<div class='a'>The policy surface shows the two levers are "
        f"complements: a tighter lag window <i>alone</i> is harmful "
        f"({tlo:+.1f}% to {thi:+.1f}%) because it empties the window. Paired "
        f"with a 14-day deposit speed-up it gains at <b>every</b> horizon "
        f"(E3c).</div></div>", unsafe_allow_html=True)

# ── the decisive picture ─────────────────────────────────────────────
T.section("The decisive picture", "The timeliness gap",
          "When the epidemic is rising — the weeks a forecast is worth most — "
          "the genomic data that describes it has not been published yet. The "
          "violet line is what exists if you wait for every sequence; the amber "
          "line is what had been deposited by that week.")

cw = D.report("coverage_by_week.csv")
seasons = [s for s in D.SEASONS if s in set(cw.season)]
pick = st.segmented_control("Season", seasons, default=seasons[-1],
                            key="home_season", label_visibility="collapsed")
pick = pick or seasons[-1]
g = cw[cw.season == pick].sort_values("season_week_idx")
x = g.weeks_from_peak
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=x, y=g.mean_wili, name="wILI (mean of regions)", fill="tozeroy",
    line=dict(color=T.CYAN, width=2.5), fillcolor="rgba(34,211,238,0.12)",
    hovertemplate="%{x:+d} wk from peak<br>wILI %{y:.2f}%<extra></extra>"))
fig.add_trace(go.Scatter(
    x=x, y=100 * g.oracle_n_covered / g.n_regions, yaxis="y2",
    name="regions with genomic data — if no delay", line=dict(color=T.VIOLET,
                                                              width=2, dash="dot"),
    hovertemplate="%{y:.0f}% of regions (oracle)<extra></extra>"))
fig.add_trace(go.Scatter(
    x=x, y=100 * g.asof_L8_16_n_covered / g.n_regions, yaxis="y2",
    name="regions with genomic data — as published", line=dict(color=T.AMBER,
                                                               width=3),
    hovertemplate="%{y:.0f}% of regions (as-of)<extra></extra>"))
fig.add_vrect(x0=x.min() - 0.5, x1=-0.5, fillcolor="rgba(251,113,133,0.06)",
              line_width=0, annotation_text="PRE-PEAK · where forecasts matter",
              annotation_position="bottom left",
              annotation_font=dict(color=T.ROSE, size=11))
fig.add_vline(x=0, line=dict(color=T.MUTED, dash="dash", width=1))
fig.update_layout(
    height=430, hovermode="x unified",
    xaxis=dict(title="weeks from the national peak"),
    yaxis=dict(title="weighted ILI (%)"),
    yaxis2=dict(title="% of regions covered", overlaying="y", side="right",
                range=[0, 105], showgrid=False, tickvals=[0, 25, 50, 75, 100]),
    legend=dict(orientation="h", y=-0.2))
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
T.provenance("data/reports/coverage_by_week.csv")

# ── the experiment board ─────────────────────────────────────────────
T.section("Eight experiments", "Each question, its answer, and where to explore it")

e2b = e2["folds"]["B2"]
e2_lo = min(v["mean"] for v in e2b.values())
e2_hi = max(v["mean"] for v in e2b.values())
e2_pos = min(v["pos"] for v in e2b.values())
e2_n = max(v["n"] for v in e2b.values())
nb0 = R.e6_contrast(e6, "none", "B0 persistence")
e6_ok = [h for h, (m, ok) in nb0.items() if ok and m > 0]
pre = e8["mde_pre"]

board = [
    ("E1", "MEASUREMENT", "How late is the genome, and when is it missing?",
     f"Median delay {e1['median_delay_weeks']:.1f} weeks; {e1['pct_over_12wk']:.1f}% "
     f"of sequences arrive over 12 weeks late. As-of coverage is "
     f"{e1['asof_pre']:.1f}% pre-peak vs {e1['asof_post']:.1f}% after.",
     "views/genomics.py"),
    ("E2", "BASELINE", "How hard is the bar to beat?",
     f"Gradient boosting beats persistence by {e2_lo:+.1f}% to {e2_hi:+.1f}% "
     f"(mean over folds), in at least {e2_pos} of {e2_n} folds at every horizon.",
     "views/forecast.py"),
    ("E3a", "NULL", "Does composition improve incidence forecasts?",
     f"No arm is sign-consistently better at any horizon "
     f"({e3a['n_consistent_pos']} positive cells). Detectable floor "
     f"{min(e3a['floor'].values()):.1f}–{max(e3a['floor'].values()):.1f}% of MAE.",
     "views/experiments.py"),
    ("E3b", "POSITIVE", "Does composition forecast subtype dominance?",
     f"Yes — and publication delay costs {e3b['lo']:.1f}–{e3b['hi']:.1f}% of it, "
     f"{e3b['n_sig']}/{e3b['n_cells']} intervals excluding zero.",
     "views/experiments.py"),
    ("E3c", "POSITIVE", "What would recover the lost skill?",
     f"L4-12 window + 14-day speed-up: {lo:+.1f}% to {hi:+.1f}% at every horizon. "
     f"A tighter window alone costs {tlo:+.1f}% to {thi:+.1f}%.",
     "views/policy.py"),
    ("E4/E5", "NULL", "Is there any information flow, model-free?",
     f"{e45['n_bh']} of {e45['n_arms']} Granger / transfer-entropy arms survive "
     f"correction; the test resolves {e45['floor_var']:.1f}% of variance.",
     "views/power.py"),
    ("E6", "NULL", "Which graph relation helps a GNN?",
     f"None beats no-graph or a density-matched random graph. The GNN itself "
     f"learns: beats persistence at {len(e6_ok)} of {len(nb0)} horizons, 4/4 folds.",
     "views/graphlab.py"),
    ("E8", "BOUND", "What could this design have detected?",
     f"Only a predictor carrying {e8['var_lo']:.0f}–{e8['var_hi']:.0f}% of variance. "
     f"Pre-peak at real coverage: none at any strength tested "
     f"({e8['pre_never']} of 2 horizons).",
     "views/power.py"),
]
for i in range(0, len(board), 4):
    cols = st.columns(4)
    for col, (eid, v, q, a, page) in zip(cols, board[i:i + 4]):
        with col:
            st.markdown(
                f"<div class='es-exp'><div class='id'>{eid} &nbsp; "
                f"{T.verdict_pill(v)}</div><div class='q'>{q}</div>"
                f"<div class='a'>{a}</div></div>", unsafe_allow_html=True)
            st.page_link(page, label="Explore", icon=":material/arrow_forward:")
T.provenance("app/core/results.py derives every number above from data/reports/*.csv")

# ── journey ──────────────────────────────────────────────────────────
T.section("The journey", "How the work unfolded, read from the results log")
tl = R.run_timeline()
steps = "".join(
    f"<div class='es-step'><div class='t'>{d}</div><div class='h'>{eid}</div>"
    f"<div class='d'>{T.esc(title)}</div></div>" for eid, title, d in tl)
st.markdown(f"<div class='es-timeline'>{steps}</div>", unsafe_allow_html=True)
T.provenance("RESULTS.md — section headings and **Run:** dates")
