"""
Visual identity: palette, Plotly template, CSS and small HTML components.

One palette is used everywhere so a colour always means the same thing:

    cyan     incidence / case history (what the surveillance system sees)
    violet   genomic composition
    amber    delay, staleness, the as-of view
    emerald  a sign-consistent improvement
    rose     a sign-consistent loss
    slate    FLIP / not resolvable
"""

import html

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

MONO = "JetBrains Mono, Consolas, Menlo, 'DejaVu Sans Mono', monospace"
SANS = "Inter, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
INK = "#E6EDF7"
MUTED = "#8A9BB5"
FAINT = "#56657E"
BG = "#070B13"
PANEL = "#0E1522"
PANEL_2 = "#131C2D"
LINE = "#1F2B40"

CYAN = "#22D3EE"
VIOLET = "#A78BFA"
AMBER = "#F5A524"
EMERALD = "#34D399"
ROSE = "#FB7185"
SLATE = "#64748B"
BLUE = "#60A5FA"
PINK = "#F472B6"

MODEL = {"B0": "#94A3B8", "B1": AMBER, "B2": CYAN, "GNN": VIOLET}
SUBTYPE = {"H3N2": VIOLET, "H1N1": CYAN, "other": SLATE}
REGION_COLOURS = ["#22D3EE", "#A78BFA", "#F5A524", "#34D399", "#FB7185",
                  "#60A5FA", "#F472B6", "#FACC15", "#2DD4BF", "#C084FC"]

SEQ_CYAN = [[0, "#0B1424"], [0.25, "#0E3A52"], [0.5, "#0E7490"],
            [0.75, "#22D3EE"], [1, "#CFFAFE"]]
SEQ_HEAT = [[0, "#0B1424"], [0.2, "#1E1B4B"], [0.45, "#6D28D9"],
            [0.7, "#DB2777"], [0.88, "#F59E0B"], [1, "#FEF3C7"]]
DIVERGING = [[0, ROSE], [0.5, "#1A2435"], [1, EMERALD]]


def _template():
    t = go.layout.Template()
    t.layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=SANS,
                  color="#C9D4E5", size=12.5),
        title=dict(font=dict(family="'Space Grotesk', " + SANS,
                             size=15, color=INK), x=0, xanchor="left"),
        colorway=REGION_COLOURS,
        xaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE,
                   tickcolor=LINE, title_font=dict(color=MUTED)),
        yaxis=dict(gridcolor=LINE, zerolinecolor="#2C3A55", linecolor=LINE,
                   tickcolor=LINE, title_font=dict(color=MUTED)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#C9D4E5")),
        hoverlabel=dict(bgcolor=PANEL_2, bordercolor=LINE,
                        font=dict(color=INK, family=SANS)),
        margin=dict(l=10, r=10, t=48, b=10),
    )
    return t


pio.templates["episurge"] = _template()
pio.templates.default = "episurge"


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
  --ink: #E6EDF7; --muted: #8A9BB5; --faint: #56657E;
  --panel: #0E1522; --panel2: #131C2D; --line: #1F2B40;
  --cyan: #22D3EE; --violet: #A78BFA; --amber: #F5A524;
  --emerald: #34D399; --rose: #FB7185;
}
html, body, [class*="css"], .stMarkdown, p, li, label {
  font-family: Inter, "Segoe UI", system-ui, sans-serif;
}
.stApp {
  background:
    radial-gradient(1200px 600px at 85% -10%, rgba(167,139,250,0.10), transparent 60%),
    radial-gradient(900px 500px at -10% 10%, rgba(34,211,238,0.08), transparent 60%),
    #070B13;
}
.block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1380px; }
h1, h2, h3, h4 { font-family: "Space Grotesk", Inter, sans-serif !important;
                 letter-spacing: -0.01em; color: var(--ink) !important; }
section[data-testid="stSidebar"] { background: #0A101B; border-right: 1px solid var(--line); }
section[data-testid="stSidebar"] .stMarkdown p { color: var(--muted); }
div[data-testid="stMetricValue"] { font-family: "JetBrains Mono", monospace; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] {
  background: transparent; border-radius: 8px 8px 0 0; padding: 8px 16px;
  color: var(--muted); font-weight: 500;
}
.stTabs [aria-selected="true"] { color: var(--ink) !important; background: var(--panel) !important; }
div[data-testid="stExpander"] { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; }
code { font-family: "JetBrains Mono", monospace !important; }

/* hero */
.es-hero { position: relative; padding: 34px 38px 30px; border-radius: 20px;
  border: 1px solid var(--line); overflow: hidden; margin-bottom: 22px;
  background:
    linear-gradient(120deg, rgba(34,211,238,0.10), rgba(167,139,250,0.10) 55%, rgba(245,165,36,0.06)),
    var(--panel); }
.es-hero:after { content: ""; position: absolute; inset: 0;
  background-image: radial-gradient(rgba(255,255,255,0.06) 1px, transparent 1px);
  background-size: 18px 18px; mask-image: linear-gradient(90deg, transparent, black 70%);
  pointer-events: none; }
.es-kicker { font-family: "JetBrains Mono", monospace; font-size: 12px;
  letter-spacing: 0.18em; text-transform: uppercase; color: var(--cyan); margin-bottom: 10px; }
.es-hero h1 { font-size: 40px; line-height: 1.08; margin: 0 0 12px 0; }
.es-hero p { color: #B6C3D8; font-size: 16.5px; max-width: 920px; margin: 0; line-height: 1.55; }
.es-grad { background: linear-gradient(90deg, #22D3EE, #A78BFA 60%, #F5A524);
  -webkit-background-clip: text; background-clip: text; color: transparent; }

/* cards */
.es-grid { display: grid; gap: 14px; margin: 6px 0 18px; }
.es-card { background: var(--panel); border: 1px solid var(--line); border-radius: 16px;
  padding: 18px 20px; position: relative; overflow: hidden; }
.es-card .lab { color: var(--muted); font-size: 12px; letter-spacing: 0.06em;
  text-transform: uppercase; font-weight: 600; }
.es-card .val { font-family: "JetBrains Mono", monospace; font-size: 30px;
  font-weight: 600; color: var(--ink); margin: 6px 0 2px; line-height: 1.1; }
.es-card .sub { color: var(--muted); font-size: 13px; line-height: 1.45; }
.es-card .bar { position: absolute; left: 0; top: 0; bottom: 0; width: 3px; }

.es-section { margin: 30px 0 10px; }
.es-section .k { font-family: "JetBrains Mono", monospace; font-size: 11.5px;
  letter-spacing: 0.16em; text-transform: uppercase; color: var(--faint); }
.es-section h2 { font-size: 24px; margin: 2px 0 4px; }
.es-section p { color: var(--muted); margin: 0; max-width: 980px; line-height: 1.55; }

.es-callout { border-radius: 14px; padding: 14px 18px; margin: 10px 0 14px;
  border: 1px solid var(--line); background: var(--panel); color: #C9D4E5;
  line-height: 1.55; font-size: 14.5px; }
.es-callout b { color: var(--ink); }

.es-pill { display: inline-block; font-family: "JetBrains Mono", monospace; font-size: 11px;
  font-weight: 600; letter-spacing: 0.08em; padding: 3px 9px; border-radius: 999px;
  border: 1px solid currentColor; }

.es-prov { font-family: "JetBrains Mono", monospace; font-size: 11px; color: var(--faint);
  margin: -6px 0 14px; }
.es-prov span { color: #7C8BA3; }

.es-exp { background: var(--panel); border: 1px solid var(--line); border-radius: 16px;
  padding: 16px 18px; height: 100%; }
.es-exp .id { font-family: "JetBrains Mono", monospace; color: var(--faint); font-size: 12px; }
.es-exp .q { color: var(--ink); font-weight: 600; font-size: 15px; margin: 4px 0 8px; line-height: 1.35; }
.es-exp .a { color: var(--muted); font-size: 13.5px; line-height: 1.5; }

.es-timeline { display: flex; gap: 0; margin: 8px 0 20px; }
.es-step { flex: 1; padding: 14px 14px 12px; border-top: 2px solid var(--line); position: relative; }
.es-step:before { content: ""; position: absolute; top: -7px; left: 14px; width: 12px; height: 12px;
  border-radius: 50%; background: var(--panel); border: 2px solid var(--cyan); }
.es-step .t { font-family: "JetBrains Mono", monospace; font-size: 11px; color: var(--cyan); }
.es-step .h { color: var(--ink); font-weight: 600; font-size: 14px; margin: 4px 0; }
.es-step .d { color: var(--muted); font-size: 12.5px; line-height: 1.45; }

.es-foot { color: var(--faint); font-size: 12px; font-family: "JetBrains Mono", monospace; }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def esc(x):
    return html.escape(str(x))


def hero(kicker, title_html, body):
    """title_html is trusted markup written in this repository."""
    st.markdown(
        f"<div class='es-hero'><div class='es-kicker'>{esc(kicker)}</div>"
        f"<h1>{title_html}</h1><p>{body}</p></div>",
        unsafe_allow_html=True)


def section(kicker, title, body=""):
    st.markdown(
        f"<div class='es-section'><div class='k'>{esc(kicker)}</div>"
        f"<h2>{esc(title)}</h2>{'<p>' + body + '</p>' if body else ''}</div>",
        unsafe_allow_html=True)


def cards(items, cols=4):
    """items: list of (label, value, sub, colour)."""
    parts = []
    for lab, val, sub, col in items:
        parts.append(
            f"<div class='es-card'><div class='bar' style='background:{col}'></div>"
            f"<div class='lab'>{esc(lab)}</div><div class='val'>{val}</div>"
            f"<div class='sub'>{sub}</div></div>")
    st.markdown(
        f"<div class='es-grid' style='grid-template-columns:repeat({cols},minmax(0,1fr))'>"
        + "".join(parts) + "</div>", unsafe_allow_html=True)


def callout(body, colour=CYAN):
    st.markdown(f"<div class='es-callout' style='border-left:3px solid {colour}'>"
                f"{body}</div>", unsafe_allow_html=True)


def pill(text, colour):
    return f"<span class='es-pill' style='color:{colour}'>{esc(text)}</span>"


VERDICT_COLOURS = {"POSITIVE": EMERALD, "NULL": SLATE, "BOUND": CYAN,
                   "BOUNDED NULL": BLUE,
                   "REJECTED": ROSE, "BASELINE": BLUE, "MEASUREMENT": AMBER}


def verdict_pill(v):
    return pill(v, VERDICT_COLOURS.get(v, SLATE))


def provenance(*files):
    """Name the artifact every figure is computed from, on the page."""
    st.markdown("<div class='es-prov'>source &middot; "
                + " &middot; ".join(f"<span>{esc(f)}</span>" for f in files)
                + "</div>", unsafe_allow_html=True)


def skill_colour(v):
    if v != v:
        return SLATE
    return EMERALD if v > 0 else ROSE
