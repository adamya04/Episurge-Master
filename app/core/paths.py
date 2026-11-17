"""
Repository paths, resolved absolutely.

The research scripts address their inputs with paths relative to the
repository root (Path("data/processed")). The app is launched as
`streamlit run app/streamlit_app.py` from that root, but nothing should
depend on the launch directory being right, so every path here is absolute
and the research modules are imported with the root on sys.path.
"""

import os
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
ROOT = APP.parent
SRC = ROOT / "src"

REPORTS = ROOT / "data" / "reports"
PROCESSED = ROOT / "data" / "processed"
FEATURES = ROOT / "data" / "features"
FIGURES = ROOT / "figures"
ASSETS = APP / "assets"


def bootstrap():
    """Make the research modules importable and their relative paths valid."""
    for p in (SRC, SRC / "analysis", SRC / "preprocessing", SRC / "features",
              SRC / "tools", APP):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    # The research code reads Path("data/...") relative to the cwd.
    if Path.cwd() != ROOT:
        os.chdir(ROOT)
