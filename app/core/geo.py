"""
Geography for display only.

State outlines come from us-atlas 3.0.1 (Census cartographic boundaries,
ISC licence), vendored as app/assets/us_states.geojson so every map renders
offline. Region positions are the mean of member-state centroids over the
contiguous states; they are layout coordinates for drawing graphs, never an
input to any statistic.
"""

import copy
from functools import lru_cache

import numpy as np

from core import data as D

NON_CONTIGUOUS = {"Alaska", "Hawaii", "Puerto Rico", "Virgin Islands"}


def _ring_centroid(coords):
    pts = np.asarray(coords, dtype=float)
    return pts[:, 0].mean(), pts[:, 1].mean(), len(pts)


def _feature_centroid(geom):
    rings = []
    if geom["type"] == "Polygon":
        rings = [geom["coordinates"][0]]
    elif geom["type"] == "MultiPolygon":
        rings = [p[0] for p in geom["coordinates"]]
    # weight by vertex count so the mainland dominates island fragments
    xs, ys, ws = zip(*[_ring_centroid(r) for r in rings])
    w = np.asarray(ws, float)
    return float(np.average(xs, weights=w)), float(np.average(ys, weights=w))


@lru_cache(maxsize=1)
def state_centroids():
    out = {}
    for f in D.states_geojson()["features"]:
        out[f["properties"]["name"]] = _feature_centroid(f["geometry"])
    return out


@lru_cache(maxsize=1)
def region_centroids():
    cents = state_centroids()
    out = {}
    for r in D.REGIONS:
        pts = [cents[s] for s, reg in D.state_region_table().values
               if reg == r and s in cents and s not in NON_CONTIGUOUS]
        out[r] = (float(np.mean([p[0] for p in pts])),
                  float(np.mean([p[1] for p in pts])))
    return out


def contiguous_geojson():
    g = copy.deepcopy(D.states_geojson())
    g["features"] = [f for f in g["features"]
                     if f["properties"]["name"] not in NON_CONTIGUOUS]
    return g


def hex_to_rgb(h, a=255):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)] + [a]


def ramp(v, vmin, vmax, stops):
    """Linear colour ramp over hex stops; returns [r, g, b]."""
    if v != v:
        return [40, 50, 70]
    t = 0.0 if vmax <= vmin else min(max((v - vmin) / (vmax - vmin), 0.0), 1.0)
    pos = t * (len(stops) - 1)
    i = min(int(pos), len(stops) - 2)
    f = pos - i
    a, b = hex_to_rgb(stops[i])[:3], hex_to_rgb(stops[i + 1])[:3]
    return [int(a[k] + (b[k] - a[k]) * f) for k in range(3)]


# A style with no sources: deck.gl draws on a plain dark background and
# never requests map tiles, so the maps work in a room with no internet.
# Served by Streamlit from app/static/ (server.enableStaticServing).
OFFLINE_BASEMAP = "/app/static/basemap.json"
