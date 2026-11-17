import re
from datetime import datetime

import pandas as pd
import streamlit as st

from core import data as D
from core import results as R
from core import theme as T
from core.paths import FIGURES, REPORTS, ROOT

T.hero(
    "Synthesis",
    "What EpiSurge contributes — <span class='es-grad'>and what it rules out</span>",
    "A research contribution is as much about the claims it can defend as the "
    "ones it makes. Every number below is derived from the committed artifacts; "
    "every rejected idea is read from the project's own decision log.",
)

e1, e3b, e3c, e45, e6, e8 = R.e1(), R.e3b(), R.e3c(), R.e45(), R.e6(), R.e8()
lo, hi = R.consistent_range(R.e3c_row(e3c, "L4-12", 14))
nb0 = R.e6_contrast(e6, "none", "B0 persistence")
g_ok = [m for m, k in nb0.values() if k and m > 0]

# ── contributions ────────────────────────────────────────────────────
T.section("Contributions", "Five claims the evidence supports")
contrib = [
    ("C1", "Vintage-correct genomic evaluation", T.AMBER,
     f"A pipeline that reconstructs genomic composition and incidence exactly as published at "
     f"each forecast date. It exposes how much oracle-dated studies leak: pre-peak composition "
     f"coverage is {e1['oracle_pre']:.1f}% by collection date but {e1['asof_pre']:.1f}% by "
     f"publication date.", "views/genomics.py"),
    ("C2", "The cost of publication delay, measured", T.EMERALD,
     f"Composition forecasts subtype dominance, and delay costs {e3b['lo']:.1f}–{e3b['hi']:.1f}% "
     f"of that skill — sign-consistent at every horizon in every pre-COVID fold, "
     f"{e3b['n_sig']}/{e3b['n_cells']} intervals excluding zero.", "views/experiments.py"),
    ("C3", "An actionable delay-policy surface", T.CYAN,
     f"Window and deposit speed are complements: L4-12 with a 14-day speed-up recovers "
     f"{lo:+.1f}% to {hi:+.1f}% at every horizon, while a tighter window alone is harmful.",
     "views/policy.py"),
    ("C4", "A power-bounded negative result", T.BLUE,
     f"Subtype composition adds no incidence signal — bounded, not asserted: the forecasting "
     f"design only resolves predictors carrying {e8['var_lo']:.0f}–{e8['var_hi']:.0f}% of "
     f"variance, and a model-free test that resolves {e45['floor_var']:.1f}% also finds "
     f"nothing.", "views/power.py"),
    ("C5", "A mechanism for spatial-GNN failure", T.VIOLET,
     f"The GNN learns (beats persistence {min(g_ok):+.1f}% to {max(g_ok):+.1f}%), yet no relation "
     f"helps — because regional correlation structure barely persists between seasons. "
     f"Stability is necessary but not sufficient.", "views/graphlab.py"),
]
for i in range(0, len(contrib), 3):
    cols = st.columns(3)
    for col, (cid, title, colour, body, page) in zip(cols, contrib[i:i + 3]):
        with col:
            st.markdown(f"<div class='es-exp' style='border-top:3px solid {colour}'>"
                        f"<div class='id'>{cid}</div><div class='q'>{title}</div>"
                        f"<div class='a'>{body}</div></div>", unsafe_allow_html=True)
            st.page_link(page, label="See the evidence", icon=":material/arrow_forward:")

# ── impact ───────────────────────────────────────────────────────────
T.section("Impact", "Why this matters beyond the project")
c1, c2 = st.columns(2)
with c1:
    T.callout("<b>For surveillance programmes.</b> The binding constraint on genomic "
              "forecasting is not modelling sophistication but deposit latency. The policy "
              "surface turns that into a costed recommendation: a two-week reduction in "
              "deposit time, paired with a modestly tighter window, buys a measurable gain.",
              T.EMERALD)
    T.callout("<b>For the literature.</b> Studies that index genomes by collection date "
              "evaluate a forecaster that could never have existed. The oracle-versus-as-of "
              "gap measured here is the size of that optimism.", T.AMBER)
with c2:
    T.callout("<b>For spatial models.</b> A graph neural network is only as good as the "
              "persistence of the relation it is given. Checking that persistence — a cheap, "
              "model-free test — should precede training one.", T.VIOLET)
    T.callout("<b>For negative results.</b> Pairing a null with a measured detection floor "
              "and a sharper model-free test turns \"we found nothing\" into a bounded claim "
              "other researchers can build on.", T.CYAN)

# ── what was killed ──────────────────────────────────────────────────
T.section("Scope discipline", "Ideas tested and killed on evidence",
          "Read from DECISIONS.md (rows marked REJECTED). Each was built, measured, and "
          "dropped because the evidence did not support it — not because it was hard.")


def md_inline(s):
    s = T.esc(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", s)


for r in R.decisions("REJECTED"):
    m = re.match(r"\*\*(.+?)\*\*\s*(.*)", r["decision"])
    title = m.group(1).rstrip(".") if m else r["decision"][:60]
    with st.expander(f"✕  {title}  ·  {r['date']}"):
        st.markdown(f"<div class='es-callout' style='border-left:3px solid {T.ROSE}'>"
                    f"<b>What was found.</b> {md_inline(m.group(2) if m else r['decision'])}"
                    f"<br><br><b>Why it was killed.</b> {md_inline(r['rationale'])}</div>",
                    unsafe_allow_html=True)

fw = (ROOT / "FUTURE_WORK.md").read_text(encoding="utf-8")
rej = []
if "## Explicitly rejected" in fw:
    blk = fw.split("## Explicitly rejected", 1)[1]
    for line in blk.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and len(cells) == 2 and not set(cells[0]) <= set("-") \
                and cells[0].lower() != "rejected":
            rej.append(cells)
        elif rej and not line.startswith("|"):
            break
if rej:
    st.markdown("**Rejected before building** — recorded in FUTURE_WORK.md so they are not re-proposed")
    st.dataframe(pd.DataFrame(rej, columns=["idea", "why it was rejected"]), hide_index=True,
                 width="stretch")

# ── limitations ──────────────────────────────────────────────────────
T.section("Limitations", "Threats to validity, stated plainly")
cov = D.core_panel()
_seq = D.sequences()
_sq = _seq.state.value_counts()
top5 = 100 * _sq.head(5).sum() / _sq.sum()
_g = _seq.groupby(["hhs_region", "state"]).size()
_share = (_g / _g.groupby(level=0).transform("sum")).groupby(level=0).max()
n_dom = int((_share > 0.5).sum())
lim = [
    ("Subtype resolution only", f"Only GenBank metadata was retained, so composition is "
     f"effectively H3N2 vs H1N1. Clade-level and antigenic signal is untested, not ruled out."),
    ("Small panel", f"{cov.hhs_region.nunique()} regions × {cov.season.nunique()} seasons "
     f"({len(cov):,} region-weeks). Fold-level verdicts rest on {len(D.PRE_FOLDS)} pre-COVID folds."),
    ("A COVID gap", f"{' and '.join(k for k, v in D.excluded_seasons().items() if 'COVID' in str(v))} "
     f"are excluded; the fold that crosses the gap "
     f"({D.COVID_TEST}) behaves differently and is reported separately rather than pooled."),
    ("Sequencing is uneven", f"The five most-sequenced states hold {top5:.0f}% of all "
     f"sequences, and in {n_dom} of 10 regions a single state supplies more than half of "
     "the region's sequences — its composition largely reflects one state's laboratories."),
    ("One country, one system", "US ILINet and US GenBank deposit behaviour. Delay costs "
     "will differ where deposit practice differs."),
    ("Ten-node graphs", "At this scale between-fold variance swamps small relational "
     "effects; the E6 null is bounded to this data size, not to graph methods in general."),
]
cols = st.columns(3)
for i, (h, b) in enumerate(lim):
    with cols[i % 3]:
        T.callout(f"<b>{h}.</b> {b}", T.SLATE)

# ── future ───────────────────────────────────────────────────────────
T.section("Future directions", "Where the evidence points next",
          "Each direction follows from a specific result above; the longer list, "
          "with feasibility notes, is in FUTURE_WORK.md.")
future = [
    ("Sequence-level genomics", T.VIOLET, "Follows from C4's scope limit",
     "Retain nucleotides, subsample ~50 sequences per region-season, and build clade "
     "assignments and genetic-distance matrices. This is the only way to test the signal "
     "this study could not: antigenic and clade-level composition."),
    ("Model the deposit process", T.AMBER, "Follows from C2 and C3",
     "Delay is the binding constraint. Modelling it explicitly — which sequences are still "
     "in the pipeline, and when they will arrive — turns staleness from a fixed cost into "
     "something a forecaster can partly correct for."),
    ("A prospective deposit pilot", T.EMERALD, "Follows from C3",
     "The policy surface is counterfactual. A surveillance programme shortening deposit time "
     "by two weeks would test the predicted gain out of sample."),
    ("Finer spatial resolution", T.CYAN, "Follows from C5",
     "State-level incidence gives ~50 nodes instead of 10. The E6 null is bounded to ten "
     "nodes; larger graphs are where relational structure could become resolvable."),
    ("More seasons, more systems", T.BLUE, "Strengthens every fold-level verdict",
     "The two COVID-era seasons as a held-out regime test, and a second surveillance "
     "system (e.g. European data) to see whether delay costs transfer."),
    ("Calibrated uncertainty", T.PINK, "Next step for deployment",
     "Conformal prediction intervals around the forecasts, with coverage checked per fold "
     "under the same as-of discipline."),
]
for i in range(0, len(future), 3):
    cols = st.columns(3)
    for col, (title, colour, why, body) in zip(cols, future[i:i + 3]):
        with col:
            st.markdown(f"<div class='es-exp' style='border-top:3px solid {colour}'>"
                        f"<div class='id'>{why}</div><div class='q'>{title}</div>"
                        f"<div class='a'>{body}</div></div>", unsafe_allow_html=True)

# ── reproducibility ──────────────────────────────────────────────────
T.section("Reproducibility", "Every artifact behind this application")
tab1, tab2, tab3 = st.tabs(["Result artifacts", "Publication figures", "Decision log"])
with tab1:
    rows = []
    for p in sorted(REPORTS.glob("*.csv")):
        with open(p, encoding="utf-8") as fh:
            n = sum(1 for _ in fh) - 1
        rows.append((p.name, n, round(p.stat().st_size / 1024, 1),
                     datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")))
    st.dataframe(pd.DataFrame(rows, columns=["file", "rows", "KB", "modified"]),
                 hide_index=True, width="stretch", height=420)
    st.code("python src/tools/run_pipeline.py --only analysis   # regenerates every file above",
            language="bash")
with tab2:
    figs = sorted(FIGURES.glob("*.png"))
    pick = st.selectbox("Figure", [f.name for f in figs], key="fig_pick")
    st.image(str(FIGURES / pick), width="stretch")
    st.caption(f"{len(figs)} matplotlib figures written by the experiment scripts themselves.")
with tab3:
    dec = pd.DataFrame(R.decisions())
    stt = st.multiselect("Status", sorted(dec.status.unique()), default=["ACTIVE", "REJECTED"],
                         key="dec_status")
    v = dec[dec.status.isin(stt)] if stt else dec
    st.dataframe(v.assign(decision=v.decision.str.replace("**", "", regex=False)),
                 hide_index=True, width="stretch", height=420,
                 column_config={"decision": st.column_config.TextColumn(width="large"),
                                "rationale": st.column_config.TextColumn(width="large")})
    st.caption(f"{len(dec)} decisions recorded in DECISIONS.md.")
