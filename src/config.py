#!/usr/bin/env python3
"""
EpiSurge - single source of truth for season and window configuration.

Before this file, season definitions were duplicated across fourteen
constants in eleven modules. Extending the study meant editing all of them
consistently, and a missed one would fail silently rather than loudly.
Every module now imports from here.

Two changes are recorded in this file (see DECISIONS.md 2026-10-29):

1. SEASON EXTENSION. The study ran on three seasons, which gave two
   rolling-origin folds. Those two folds produced opposite conclusions
   (climatological skill at h=8: -8.9% in fold 1, +56.7% in fold 2),
   because the three seasons peak at MMWR weeks 06, 47 and 52 and a
   climatology built from one season lands on the wrong week in the next.
   Four pre-COVID seasons are added, giving seven seasons and six folds.

2. FETCH-WINDOW CORRECTION. fetch_ncbi.py searched NCBI by PUBLICATION
   date over Oct-May while qc_filter.py filtered by COLLECTION date over
   the same months. With a median collection-to-publication delay of ~76
   days, anything collected late in the season was published after the
   window closed and was never retrieved. The signature is unmistakable in
   the retrieved data: maximum observed delay tracks days-remaining-to-
   window-close month by month, in every season, and April/May collections
   are near zero. Publication windows now extend FETCH_TAIL_MONTHS past
   the collection window so the delay distribution is uncensored.

   This affected the oracle composition and the delay distribution. It did
   NOT affect the as-of features: those require submission <= origin, and
   evaluated origins stop at MMWR week 20, inside the old window.
"""

from datetime import datetime

# ──────────────────────────────────────────────────────────────────────
# Seasons
# ──────────────────────────────────────────────────────────────────────
# 2020-21 and 2021-22 are excluded: COVID-era non-pharmaceutical
# interventions suppressed influenza circulation to the point where the
# seasons are not comparable. That exclusion is unchanged.
#
# label: (collection_start, collection_end, epiweek_window_start,
#         epiweek_window_end)
#
# Collection windows run 1 Oct - 31 May. Epiweek windows run MMWR week 27
# to week 26 of the following year, giving lead-in weeks before the core
# evaluation window (weeks 40-20).

SEASONS = {
    "2015-16": (datetime(2015, 10, 1), datetime(2016, 5, 31), 201527, 201626),
    "2016-17": (datetime(2016, 10, 1), datetime(2017, 5, 31), 201627, 201726),
    "2017-18": (datetime(2017, 10, 1), datetime(2018, 5, 31), 201727, 201826),
    "2018-19": (datetime(2018, 10, 1), datetime(2019, 5, 31), 201827, 201926),
    "2019-20": (datetime(2019, 10, 1), datetime(2020, 5, 31), 201927, 202026),
    "2022-23": (datetime(2022, 10, 1), datetime(2023, 5, 31), 202227, 202326),
    "2023-24": (datetime(2023, 10, 1), datetime(2024, 5, 31), 202327, 202426),
}

SEASON_LABELS = list(SEASONS.keys())

# The three seasons the study ran on before the extension. Kept so results
# can be reproduced on the original scope for comparison.
ORIGINAL_SEASONS = ["2019-20", "2022-23", "2023-24"]

EXCLUDED_SEASONS = {
    "2020-21": "COVID-era suppression of influenza circulation",
    "2021-22": "COVID-era suppression of influenza circulation",
}

# ──────────────────────────────────────────────────────────────────────
# Derived windows
# ──────────────────────────────────────────────────────────────────────

QC_WINDOWS = {label: (start, end) for label, (start, end, _, _) in SEASONS.items()}
SEASON_WINDOWS = [(a, b) for (_, _, a, b) in SEASONS.values()]
EPIWEEK_RANGE = f"{min(a for a, _ in SEASON_WINDOWS)}-{max(b for _, b in SEASON_WINDOWS)}"

# Core evaluation window: MMWR weeks 40 through 20.
CORE_WEEK_START = 40
CORE_WEEK_END = 20


def is_core_season(epiweek: int) -> bool:
    week = epiweek % 100
    return week >= CORE_WEEK_START or week <= CORE_WEEK_END


def assign_season(epiweek: int):
    for label, (_, _, start, end) in SEASONS.items():
        if start <= epiweek <= end:
            return label
    return None


# ──────────────────────────────────────────────────────────────────────
# NCBI fetch windows
# ──────────────────────────────────────────────────────────────────────
# NCBI indexes publication date ([PDAT]) but not collection date, so the
# search is necessarily by publication date and collection is filtered
# downstream in qc_filter.py. The publication window must therefore extend
# far enough past the collection window to capture slow submissions.
#
# Maximum delay observed under the old (censored) window was 233 days.
# Twenty-four months is roughly three times that, which should leave the
# distribution effectively uncensored. verify_no_censoring() below is the
# test that this actually worked - run it after refetching.

FETCH_TAIL_MONTHS = 24


def fetch_window(label: str):
    """(start, end) publication-date strings for NCBI esearch, YYYY/MM/DD."""
    start, end, _, _ = SEASONS[label]
    tail_year = end.year + FETCH_TAIL_MONTHS // 12
    tail_month = end.month + FETCH_TAIL_MONTHS % 12
    if tail_month > 12:
        tail_year += 1
        tail_month -= 12
    return (start.strftime("%Y/%m/%d"), f"{tail_year}/{tail_month:02d}/{end.day:02d}")


def verify_no_censoring(delay_df, season_col="season",
                        collection_col="collection_date", delay_col="delay_days"):
    """
    Check that the delay distribution is no longer truncated by the fetch
    window.

    Under censoring, the maximum observed delay in each collection month is
    bounded by the days remaining until the publication window closes, so
    max_delay rises monotonically as collection date moves earlier and sits
    just under that bound. Once the window is wide enough, max_delay should
    stop tracking the bound.

    Returns (ok, report_dataframe). ok is False if any collection month's
    maximum delay still sits within 15% of its window bound, which is the
    censoring signature.
    """
    import pandas as pd

    rows = []
    for label, grp in delay_df.groupby(season_col):
        if label not in SEASONS:
            continue
        _, _, _ = None, None, None
        pub_end = datetime.strptime(fetch_window(label)[1], "%Y/%m/%d")
        g = grp.copy()
        g[collection_col] = pd.to_datetime(g[collection_col])
        g["month"] = g[collection_col].dt.to_period("M")
        for month, mg in g.groupby("month"):
            bound = (pub_end - month.to_timestamp()).days
            mx = float(mg[delay_col].max())
            rows.append({
                "season": label, "month": str(month), "n": len(mg),
                "max_delay": mx, "window_bound": bound,
                "pct_of_bound": round(100 * mx / bound, 1) if bound else None,
            })
    rep = pd.DataFrame(rows)
    if rep.empty:
        return True, rep
    ok = bool((rep["pct_of_bound"] < 85).all())
    return ok, rep