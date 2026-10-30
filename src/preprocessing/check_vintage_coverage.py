#!/usr/bin/env python3
"""
EpiSurge - verify vintage coverage supports every forecast origin.

Feature engineering needs issue T to exist for every origin week T.
A missing issue silently drops that origin from the modelling set.
"""

import pandas as pd
from epiweeks import Week

import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from config import SEASON_WINDOWS



v = pd.read_csv("data/raw/fluview/fluview_vintages.csv")
have = set(v["issue"].unique())

want = set()
for start, end in SEASON_WINDOWS:
    w = Week(start // 100, start % 100)
    while w.year * 100 + w.week <= end:
        want.add(w.year * 100 + w.week)
        w = w + 1

print(f"Issues present:  {len(have)}")
print(f"Origins needed:  {len(want)}")
print(f"Missing issues:  {sorted(want - have)}")
print(f"Extra issues:    {sorted(have - want)}")

# every origin must have usable lag>=1 history
by_issue = v[v["lag"] >= 1].groupby("issue")["epiweek"].max()
problems = [i for i in sorted(want) if i not in by_issue.index]
print(f"Origins with no lag>=1 data: {problems}")