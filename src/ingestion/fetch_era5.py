#!/usr/bin/env python3
"""
EpiSurge - Phase 1: ERA5 climate data for US HHS regions.

Pulls daily-mean 2m temperature and 2m dewpoint over a CONUS bounding box.
Dewpoint + temperature later give absolute humidity, which is the variable
with established influenza-transmission relevance.

Uses derived-era5-single-levels-daily-statistics (pre-aggregated daily means)
rather than raw hourly - roughly 24x smaller requests, and we only need
weekly means downstream.

Requests are split by year to avoid 403 cost-limit rejections.

Setup required first:
  pip install "cdsapi>=0.7.7"
  C:\\Users\\<you>\\.cdsapirc containing:
      url: https://cds.climate.copernicus.eu/api
      key: <PERSONAL-ACCESS-TOKEN>
  Licence accepted on the ERA5 dataset download page.
"""

import sys
from pathlib import Path

try:
    import cdsapi
except ImportError:
    sys.exit("cdsapi not installed. Run: pip install \"cdsapi>=0.7.7\"")

RAW_DIR = Path("data/raw/era5")
RAW_DIR.mkdir(parents=True, exist_ok=True)

DATASET = "derived-era5-single-levels-daily-statistics"

# CONUS bounding box: North, West, South, East
# Covers all 10 HHS regions on the mainland.
# NOTE: excludes Alaska and Hawaii - handled as a documented limitation,
# see DECISIONS.md. Those states are small population shares of regions 9/10.
AREA = [50, -125, 24, -66]

VARIABLES = ["2m_temperature", "2m_dewpoint_temperature"]

# Season coverage: Oct-May of each flu season, plus lead-in months
# for lagged climate features.
YEAR_MONTHS = {
    2019: ["07","08","09","10","11","12"],
    2020: ["01","02","03","04","05","06"],
    2022: ["07","08","09","10","11","12"],
    2023: ["01","02","03","04","05","06","07","08","09","10","11","12"],
    2024: ["01","02","03","04","05","06"],
}

ALL_DAYS = [f"{d:02d}" for d in range(1, 32)]


def fetch_year(client, year, months):
    out_path = RAW_DIR / f"era5_daily_{year}.nc"

    if out_path.exists():
        print(f"  {year}: already downloaded, skipping ({out_path.stat().st_size/1e6:.1f} MB)")
        return out_path

    request = {
        "product_type": "reanalysis",
        "variable": VARIABLES,
        "year": str(year),
        "month": months,
        "day": ALL_DAYS,
        "daily_statistic": "daily_mean",
        "time_zone": "utc+00:00",
        "frequency": "1_hourly",
        "area": AREA,
        "data_format": "netcdf",
    }

    print(f"  {year}: requesting months {months[0]}-{months[-1]}...")
    print(f"         (queues on CDS servers - can take several minutes)")

    client.retrieve(DATASET, request, str(out_path))
    print(f"  {year}: saved -> {out_path.name} ({out_path.stat().st_size/1e6:.1f} MB)")
    return out_path


def main():
    print("ERA5 daily climate fetch")
    print(f"  Dataset:   {DATASET}")
    print(f"  Variables: {VARIABLES}")
    print(f"  Area:      N={AREA[0]} W={AREA[1]} S={AREA[2]} E={AREA[3]} (CONUS)")
    print()

    try:
        client = cdsapi.Client()
    except Exception as e:
        sys.exit(f"Could not create CDS client: {e}\n"
                 f"Check that C:\\Users\\<you>\\.cdsapirc exists and is correctly formatted.")

    downloaded = []
    for year in sorted(YEAR_MONTHS):
        try:
            downloaded.append(fetch_year(client, year, YEAR_MONTHS[year]))
        except Exception as e:
            print(f"  {year}: FAILED - {e}")
            if "licence" in str(e).lower() or "403" in str(e):
                print("\n  -> Licence not accepted. Go to the dataset's Download tab")
                print("     and click 'Accept terms' at the bottom of the page.")
                break
            if "cost" in str(e).lower():
                print("\n  -> Request too large. Reduce months per request in YEAR_MONTHS.")
                break

    print(f"\nDownloaded {len(downloaded)} / {len(YEAR_MONTHS)} year files")
    if downloaded:
        print("Files:")
        for p in downloaded:
            print(f"  {p}")


if __name__ == "__main__":
    main()