"""
Live forecasting: the E2 baselines, re-run inside the app.

This calls the research code's own functions (build_targets, usable,
b0_persistence, b1_climatology, b2_gbm) in the same order baselines.main()
does, so a forecast shown in the app is the experiment's forecast and not a
reimplementation. The B1-defines-the-evaluation-set rule is kept, so the
live MAE can be checked against data/reports/baseline_results.csv.
"""

import numpy as np
import pandas as pd
import streamlit as st

from core import data as D

import baselines                                              # noqa: E402


@st.cache_data(show_spinner="Training B2 gradient boosting on the fold…")
def fold_forecasts(fold, h, seed=0):
    core = D.core_panel()
    d = baselines.build_targets(core, h)
    d["target_week"] = d.groupby(["season", "hhs_region"])["week"].shift(-h)

    spec = {f: (tr, te) for f, tr, te in D.FOLDS}
    train_seasons, test_season = spec[fold]
    train_all = d[d["season"].isin(train_seasons)]
    train = baselines.usable(train_all)
    test = baselines.usable(d[d["season"] == test_season])
    baselines.assert_no_leakage(train, test, fold)

    p1, nseas = baselines.b1_climatology(train_all, test)
    ok = ~np.isnan(p1)
    t = test[ok].copy()
    t["B0"] = baselines.b0_persistence(t)
    t["B1"] = p1[ok]
    t["B1_seasons"] = nseas[ok]
    t["B2"] = baselines.b2_gbm(train, t, seed)
    t["target_idx"] = t["season_week_idx"] + h
    keep = ["season", "hhs_region", "epiweek", "season_week_idx", "target_idx",
            "weeks_from_peak", "peak_side", "wili_final", "ili_asof_lag1",
            "y_target", "B0", "B1", "B1_seasons", "B2"]
    return t[keep].reset_index(drop=True), len(train), test_season


def live_mae(frame):
    return {m: float(np.mean(np.abs(frame["y_target"] - frame[m])))
            for m in ("B0", "B1", "B2")}


def committed_mae(fold, h, seed=0):
    r = D.report("baseline_results.csv")
    r = r[(r.fold == fold) & (r.horizon == h) & (r.stratum == "all")]
    out = {}
    for m in ("B0", "B1", "B2"):
        s = r[(r.model == m) & (r.seed == (seed if m == "B2" else 0))]
        out[m] = float(s.mae.iloc[0]) if len(s) else np.nan
    return out


def season_truth(season):
    core = D.core_panel()
    return core[core.season == season][["hhs_region", "season_week_idx",
                                        "epiweek", "wili_final",
                                        "weeks_from_peak"]]


def region_trajectory_frame(frame, region):
    """Per-origin forecasts for one region, indexed by the TARGET week."""
    f = frame[frame.hhs_region == region].sort_values("target_idx")
    return pd.DataFrame({
        "target_idx": f["target_idx"].values,
        "truth": f["y_target"].values,
        "B0": f["B0"].values, "B1": f["B1"].values, "B2": f["B2"].values,
    })
