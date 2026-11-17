"""
EpiSurge — research demonstration platform.

Run from the repository root, inside the venv:

    streamlit run app/streamlit_app.py

Every number on every page is computed at run time from the committed
artifacts in data/ (see app/core/results.py). Nothing is typed in.
"""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st                                            # noqa: E402

from core.paths import bootstrap                                  # noqa: E402

bootstrap()

st.set_page_config(
    page_title="EpiSurge · Genomic Epidemic Intelligence",
    page_icon=":material/coronavirus:",
    layout="wide",
    initial_sidebar_state="expanded",
)

from core import theme                                            # noqa: E402

theme.inject_css()

V = APP_DIR / "views"
pages = {
    "Story": [
        st.Page(V / "home.py", title="Mission brief", icon=":material/rocket_launch:",
                default=True, url_path="brief"),
        st.Page(V / "pipeline.py", title="Architecture & pipeline",
                icon=":material/account_tree:", url_path="pipeline"),
    ],
    "Explore the data": [
        st.Page(V / "atlas.py", title="Epidemic atlas", icon=":material/public:",
                url_path="atlas"),
        st.Page(V / "genomics.py", title="Genomic surveillance",
                icon=":material/genetics:", url_path="genomics"),
    ],
    "Experiments": [
        st.Page(V / "experiments.py", title="Experiment explorer",
                icon=":material/science:", url_path="experiments"),
        st.Page(V / "forecast.py", title="Forecast studio",
                icon=":material/query_stats:", url_path="forecast"),
        st.Page(V / "graphlab.py", title="Graph lab · GNN",
                icon=":material/hub:", url_path="graph-lab"),
        st.Page(V / "policy.py", title="Policy simulator",
                icon=":material/tune:", url_path="policy"),
        st.Page(V / "power.py", title="Detectability frontier",
                icon=":material/radar:", url_path="detectability"),
    ],
    "Synthesis": [
        st.Page(V / "impact.py", title="Contributions & impact",
                icon=":material/workspace_premium:", url_path="impact"),
    ],
}

with st.sidebar:
    st.markdown(
        "<div style='padding:6px 4px 14px'>"
        "<div style='font-family:Space Grotesk,Inter,sans-serif;font-size:22px;"
        "font-weight:700;letter-spacing:-0.02em;color:#E6EDF7'>Epi"
        "<span class='es-grad'>Surge</span></div>"
        "<div style='font-family:JetBrains Mono,monospace;font-size:10.5px;"
        "letter-spacing:0.16em;color:#56657E;text-transform:uppercase;"
        "margin-top:2px'>Genomic epidemic intelligence</div></div>",
        unsafe_allow_html=True)

nav = st.navigation(pages)

with st.sidebar:
    st.markdown("---")
    st.markdown(
        "<div class='es-foot'>Team 18 · 23AID498<br>Amrita Vishwa Vidyapeetham"
        "<br><br>All figures computed live from<br>committed artifacts in "
        "<code>data/</code></div>", unsafe_allow_html=True)

nav.run()
