"""
Cached loaders for every artifact the app shows.

Nothing in the app reads a file except through this module, so the set of
inputs is auditable in one place, and each loader is cached for the life of
the server process: the review demo pays the load cost once.
"""

import json
import re
from functools import lru_cache

import numpy as np
import pandas as pd
import streamlit as st

from core.paths import ASSETS, PROCESSED, REPORTS, bootstrap

bootstrap()

import baselines                                                  # noqa: E402
from config import SEASON_LABELS, EXCLUDED_SEASONS                # noqa: E402
from map_region_epiweek import STATE_TO_HHS                       # noqa: E402
from build_oracle_composition import extract_subtype              # noqa: E402

SEASONS = list(SEASON_LABELS)
HORIZONS = list(baselines.HORIZONS)
FOLDS = list(baselines.FOLDS)                 # (name, train_seasons, test)
COVID_TEST = "2022-23"
PRE_FOLDS = [f for f, _, te in FOLDS if te != COVID_TEST]
FOLD_TEST = {f: te for f, _, te in FOLDS}
REGIONS = list(range(1, 11))

# CDC's HHS regional office cities. Reference geography only, used as
# labels; it is never an input to any computation.
REGION_HQ = {1: "Boston", 2: "New York", 3: "Philadelphia", 4: "Atlanta",
             5: "Chicago", 6: "Dallas", 7: "Kansas City", 8: "Denver",
             9: "San Francisco", 10: "Seattle"}


def region_label(r):
    return f"R{int(r)} · {REGION_HQ[int(r)]}"


@st.cache_data(show_spinner=False)
def core_panel():
    """The core-season region-week panel exactly as the experiments load it."""
    baselines.MASTER = PROCESSED / "master_region_week.csv"
    return baselines.load()


@st.cache_data(show_spinner=False)
def master_full():
    return pd.read_csv(PROCESSED / "master_region_week.csv")


@st.cache_data(show_spinner=False)
def report(name):
    return pd.read_csv(REPORTS / name)


def report_exists(name):
    return (REPORTS / name).exists()


@st.cache_data(show_spinner=False)
def processed_json(name):
    with open(PROCESSED / name, encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner="Loading 32k sequence records…")
def sequences():
    """
    Every mapped HA sequence with its state, region, subtype, collection
    date and publication (GenBank create) date.

    The subtype is parsed with the pipeline's own extract_subtype applied to
    the same field, so counts here match the composition features exactly.
    """
    rows = []
    for season in SEASONS:
        p = PROCESSED / f"mapped_{season}.json"
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as fh:
            for r in json.load(fh):
                rows.append((r["accession"], season, r.get("state"),
                             r.get("hhs_region"), r.get("collection_date_parsed"),
                             r.get("description", ""), r.get("organism", ""),
                             r.get("sequence_length")))
    df = pd.DataFrame(rows, columns=["accession", "season", "state", "hhs_region",
                                     "collection_date", "description",
                                     "organism", "length"])
    df["subtype"] = df["description"].map(extract_subtype)
    df["collection_date"] = pd.to_datetime(df["collection_date"], errors="coerce")
    # Older records name the strain in `organism`; newer ones only in
    # `description`. Take the description first, organism as fallback.
    pat = r"\((A/[^()]*?)(?:\(|\))"
    df["strain"] = (df["description"].str.extract(pat)[0]
                    .fillna(df["organism"].str.extract(pat)[0]))
    dl = report("delay_distribution.csv")[["accession", "submission_date",
                                           "delay_days"]]
    df = df.merge(dl, on="accession", how="left")
    df["submission_date"] = pd.to_datetime(df["submission_date"], errors="coerce")
    return df.drop(columns=["description", "organism"])


@lru_cache(maxsize=1)
def states_geojson():
    with open(ASSETS / "us_states.geojson", encoding="utf-8") as fh:
        g = json.load(fh)
    alias = {"United States Virgin Islands": "Virgin Islands"}
    keep = []
    for f in g["features"]:
        name = alias.get(f["properties"]["name"], f["properties"]["name"])
        f["properties"]["name"] = name
        f["properties"]["hhs_region"] = STATE_TO_HHS.get(name)
        if f["properties"]["hhs_region"] is not None:
            keep.append(f)
    g["features"] = keep
    return g


def state_region_table():
    return pd.DataFrame(sorted(STATE_TO_HHS.items()), columns=["state", "hhs_region"])


def excluded_seasons():
    return dict(EXCLUDED_SEASONS)


def season_order_key(s):
    return SEASONS.index(s) if s in SEASONS else 99


def strip_html(s):
    return re.sub(r"<[^>]+>", "", s)


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan
