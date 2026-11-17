"""Shared chart builders."""

import numpy as np
import plotly.graph_objects as go

from core import data as D
from core.theme import MONO

H = D.HORIZONS


def matrix_fig(rows, height=None, title=""):
    """rows: list of (label, {h: (mean, ok)}). Colour only sign-consistent cells."""
    z, txt = [], []
    for _, cells in rows:
        zr, tr = [], []
        for h in H:
            m, ok = cells.get(h, (np.nan, False))
            zr.append(m if ok else np.nan)
            tr.append("–" if m != m else (f"{m:+.1f}" if ok else "FLIP"))
        z.append(zr)
        txt.append(tr)
    lim = np.nanmax(np.abs(np.array(z, dtype=float))) if np.isfinite(np.array(z, dtype=float)).any() else 1
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=[[0] * len(H)] * len(rows), x=[f"h={h}" for h in H],
                             y=[r[0] for r in rows], showscale=False, hoverinfo="skip",
                             colorscale=[[0, "#141C2B"], [1, "#141C2B"]], xgap=3, ygap=3))
    fig.add_trace(go.Heatmap(z=z, x=[f"h={h}" for h in H], y=[r[0] for r in rows],
                             text=txt, texttemplate="%{text}", zmid=0, zmin=-lim, zmax=lim,
                             colorscale=[[0, "#9F1239"], [0.35, "#D9415A"], [0.5, "#1A2435"],
                                         [0.65, "#0E9F6E"], [1, "#065F46"]],
                             xgap=3, ygap=3, colorbar=dict(title="skill %"),
                             textfont=dict(family=MONO, size=12),
                             hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>"))
    fig.update_layout(height=height or 70 + 34 * len(rows), title=title,
                      margin=dict(l=10, r=10, t=40 if title else 10, b=10))
    fig.update_yaxes(autorange="reversed")
    return fig
