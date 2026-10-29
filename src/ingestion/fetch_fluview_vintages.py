#!/usr/bin/env python3
"""
EpiSurge - FluView revision history (vintages) via Delphi Epidata.

CDC revises ILI values after first publication (backfill). Input lag
features must use values AS PUBLISHED at forecast time, not final values.

Fetches every issue in the season windows, all regions, full epiweek range.
Output is long-format: one row per (region, epiweek, issue).

Convention (log in DECISIONS.md):
  TARGET = final revised wILI   (CDC FluSight scoring convention)
  INPUTS = wILI as of the forecast-origin issue

Resumable: issues already in the output file are skipped.
Optional: set DELPHI_API_KEY if anonymous rate limits are hit (HTTP 429).
"""

import csv
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from epiweeks import Week

RAW_DIR = Path("data/raw/fluview")
OUT = RAW_DIR / "fluview_vintages.csv"
FINAL = RAW_DIR / "fluview.csv"

API = "https://api.delphi.cmu.edu/epidata/fluview/"
REGIONS = ",".join(f"hhs{i}" for i in range(1, 11))
EPIWEEKS = "201901-202430"
SEASON_WINDOWS = [(201927, 202026), (202227, 202326), (202327, 202426)]
FIELDS = ["region", "epiweek", "issue", "lag", "wili", "ili",
          "num_ili", "num_patients"]


def issues_to_fetch():
    out = []
    for start, end in SEASON_WINDOWS:
        w = Week(start // 100, start % 100)
        while w.year * 100 + w.week <= end:
            out.append(w.year * 100 + w.week)
            w = w + 1
    return out


def fetch_issue(issue, api_key):
    params = {"regions": REGIONS, "epiweeks": EPIWEEKS, "issues": issue}
    if api_key:
        params["api_key"] = api_key
    r = requests.get(API, params=params, timeout=60)
    if r.status_code == 429:
        sys.exit("HTTP 429 rate limit - register a free Delphi API key and "
                 "set DELPHI_API_KEY, then rerun (progress is kept).")
    r.raise_for_status()
    payload = r.json()
    code = payload.get("result")
    if code == -2:
        return []
    if code == 2:
        sys.exit(f"Issue {issue}: result truncated by API row limit - "
                 f"narrow EPIWEEKS and rerun.")
    if code != 1:
        sys.exit(f"Issue {issue}: API error - {payload.get('message')}")
    return payload["epidata"]


def main():
    api_key = os.getenv("DELPHI_API_KEY")
    issues = issues_to_fetch()

    done = set()
    if OUT.exists():
        done = set(pd.read_csv(OUT, usecols=["issue"])["issue"].unique())
    todo = [i for i in issues if i not in done]
    print(f"Issues in scope: {len(issues)}  already fetched: {len(done)}  "
          f"to fetch: {len(todo)}")

    write_header = not OUT.exists()
    with open(OUT, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(FIELDS)
        for n, issue in enumerate(todo, 1):
            rows = fetch_issue(issue, api_key)
            for r in rows:
                w.writerow([r.get(k, "") for k in FIELDS])
            f.flush()
            print(f"  [{n}/{len(todo)}] issue {issue}: {len(rows)} rows")
            time.sleep(0.25)

    # ── quantify backfill ───────────────────────────────────────
    v = pd.read_csv(OUT)
    final = pd.read_csv(FINAL)[["region", "epiweek", "wili"]].rename(
        columns={"wili": "wili_final"})

    empty = sorted(set(issues) - set(v["issue"].unique()))
    print(f"\nTotal vintage rows: {len(v)}")
    print(f"Issues with no data: {len(empty)} {empty[:10]}")

    first = (v.sort_values("lag")
              .groupby(["region", "epiweek"], as_index=False)
              .first()[["region", "epiweek", "lag", "wili"]]
              .rename(columns={"wili": "wili_first", "lag": "first_lag"}))
    cmp = first.merge(final, on=["region", "epiweek"], how="inner")
    cmp["abs_rev"] = (cmp["wili_final"] - cmp["wili_first"]).abs()

    print(f"\nBackfill: first-published vs final wILI "
          f"({len(cmp)} region-weeks)")
    print(f"  typical first-publication lag: {cmp['first_lag'].median():.0f} week(s)")
    print(f"  mean |revision|:   {cmp['abs_rev'].mean():.4f} pct-points")
    print(f"  median |revision|: {cmp['abs_rev'].median():.4f}")
    print(f"  p90 |revision|:    {cmp['abs_rev'].quantile(0.9):.4f}")
    print(f"  revised by >0.1:   {(cmp['abs_rev'] > 0.1).mean()*100:.1f}%")


if __name__ == "__main__":
    main()