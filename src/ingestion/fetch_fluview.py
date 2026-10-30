#!/usr/bin/env python3
"""
EpiSurge - Phase 1: CDC FluView ILI via Delphi Epidata API.

This is the FORECASTING TARGET: weighted ILI % per HHS region per week.
No API key required.

Docs: https://cmu-delphi.github.io/delphi-epidata/api/fluview.html
"""

import csv
import json
import time
from pathlib import Path

import requests

RAW_DIR = Path("data/raw/fluview")
RAW_DIR.mkdir(parents=True, exist_ok=True)

API = "https://api.delphi.cmu.edu/epidata/fluview/"
REGIONS = [f"hhs{i}" for i in range(1, 11)]

# cover all three seasons plus lead-in weeks for lagged features
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from config import EPIWEEK_RANGE




def fetch():
    params = {
        "regions": ",".join(REGIONS),
        "epiweeks": EPIWEEK_RANGE,
    }
    print(f"Requesting FluView: regions=hhs1-10  epiweeks={EPIWEEK_RANGE}")
    resp = requests.get(API, params=params, timeout=60)
    resp.raise_for_status()
    payload = resp.json()

    if payload.get("result") != 1:
        raise RuntimeError(f"API error: {payload.get('message')}")

    return payload["epidata"]


def main():
    rows = fetch()
    print(f"Received {len(rows)} region-week rows")

    with open(RAW_DIR / "fluview_raw.json", "w") as f:
        json.dump(rows, f, indent=2)

    fields = ["region", "epiweek", "release_date", "issue", "lag",
              "num_ili", "num_patients", "num_providers", "wili", "ili"]

    with open(RAW_DIR / "fluview.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in rows:
            w.writerow([r.get(k, "") for k in fields])

    regions_seen = sorted({r["region"] for r in rows})
    weeks_seen = sorted({r["epiweek"] for r in rows})
    missing_wili = sum(1 for r in rows if r.get("wili") is None)

    print(f"  Regions: {len(regions_seen)}  {regions_seen}")
    print(f"  Epiweeks: {len(weeks_seen)}  ({weeks_seen[0]} to {weeks_seen[-1]})")
    print(f"  Rows missing wili: {missing_wili}")
    print(f"\nSaved -> {RAW_DIR}\\fluview.csv and fluview_raw.json")


if __name__ == "__main__":
    main()