#!/usr/bin/env python3
"""
EpiSurge - Fetch the ERA5 years that failed or weren't reached.

2023 is split into halves because a full 12-month request exceeds
the CDS per-request cost limit.
"""

import sys
from pathlib import Path

import cdsapi

RAW_DIR = Path("data/raw/era5")
RAW_DIR.mkdir(parents=True, exist_ok=True)

DATASET = "derived-era5-single-levels-daily-statistics"
AREA = [50, -125, 24, -66]
VARIABLES = ["2m_temperature", "2m_dewpoint_temperature"]
ALL_DAYS = [f"{d:02d}" for d in range(1, 32)]

# (output_suffix, year, months)
REQUESTS = [
    ("2023a", "2023", ["01", "02", "03", "04", "05", "06"]),
    ("2023b", "2023", ["07", "08", "09", "10", "11", "12"]),
    ("2024",  "2024", ["01", "02", "03", "04", "05", "06"]),
]


def main():
    client = cdsapi.Client()

    for suffix, year, months in REQUESTS:
        out_path = RAW_DIR / f"era5_daily_{suffix}.nc"

        if out_path.exists():
            print(f"{suffix}: already downloaded, skipping")
            continue

        print(f"{suffix}: requesting {year} months {months[0]}-{months[-1]}...")

        try:
            client.retrieve(DATASET, {
                "product_type": "reanalysis",
                "variable": VARIABLES,
                "year": year,
                "month": months,
                "day": ALL_DAYS,
                "daily_statistic": "daily_mean",
                "time_zone": "utc+00:00",
                "frequency": "1_hourly",
                "area": AREA,
                "data_format": "netcdf",
            }, str(out_path))

            print(f"{suffix}: saved ({out_path.stat().st_size/1e6:.1f} MB)")

        except Exception as e:
            print(f"{suffix}: FAILED - {e}")
            if "cost" in str(e).lower():
                print("  -> still too large, split further")
            continue

    print("\nDone.")


if __name__ == "__main__":
    main()