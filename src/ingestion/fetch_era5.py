#!/usr/bin/env python3
"""
EpiSurge - Phase 1: ERA5 climate data for US HHS regions.

Pulls daily-mean 2m temperature and 2m dewpoint over a CONUS bounding box.
Dewpoint + temperature later give absolute humidity, which is the variable
with established influenza-transmission relevance.

Uses derived-era5-single-levels-daily-statistics (pre-aggregated daily
means) rather than raw hourly - roughly 24x smaller requests, and only
weekly means are needed downstream.

REQUEST SIZE: every request is a HALF YEAR. A full twelve-month request
exceeds the CDS per-request cost limit; that is why fetch_era5_remaining.py
had to exist. Generating half-years here makes it unnecessary.

COVERAGE: derived from src/config.py, so extending the study automatically
extends the climate pull. Season windows run MMWR week 27 to week 26, i.e.
roughly July to June, so each season needs the second half of its start
year and the first half of its end year.

Setup required first:
  pip install "cdsapi>=0.7.7"
  C:\\Users\\<you>\\.cdsapirc containing:
      url: https://cds.climate.copernicus.eu/api
      key: <PERSONAL-ACCESS-TOKEN>
  Licence accepted on the ERA5 dataset download page.

Run (from the repository root):
    python src/ingestion/fetch_era5.py
    python src/ingestion/fetch_era5.py --list     # show what is needed, fetch nothing
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import SEASONS                               # noqa: E402

RAW_DIR = Path("data/raw/era5")

DATASET = "derived-era5-single-levels-daily-statistics"

# CONUS bounding box: North, West, South, East.
# Excludes Alaska and Hawaii - documented limitation, see DECISIONS.md.
AREA = [50, -125, 24, -66]

VARIABLES = ["2m_temperature", "2m_dewpoint_temperature"]
ALL_DAYS = [f"{d:02d}" for d in range(1, 32)]

H1 = ["01", "02", "03", "04", "05", "06"]
H2 = ["07", "08", "09", "10", "11", "12"]

# Files fetched before this script was generalised, under the old naming.
# Listed so they are not re-downloaded.
LEGACY = {
    (2019, "H2"): "era5_daily_2019.nc",
    (2020, "H1"): "era5_daily_2020.nc",
    (2022, "H2"): "era5_daily_2022.nc",
    (2023, "H1"): "era5_daily_2023a.nc",
    (2023, "H2"): "era5_daily_2023b.nc",
    (2024, "H1"): "era5_daily_2024.nc",
}


def required_halves():
    """
    Half-years needed to cover every season window in config.

    A season labelled YYYY-YY spans epiweek YYYY27 to (YYYY+1)26, so it
    needs H2 of its start year and H1 of its end year.
    """
    halves = set()
    for _, (_, _, start_ew, end_ew) in SEASONS.items():
        halves.add((start_ew // 100, "H2"))
        halves.add((end_ew // 100, "H1"))
    return sorted(halves)


def target_path(year, half):
    legacy = LEGACY.get((year, half))
    if legacy and (RAW_DIR / legacy).exists():
        return RAW_DIR / legacy
    return RAW_DIR / f"era5_{year}{half}.nc"


def fetch(client, year, half):
    out = target_path(year, half)
    if out.exists():
        print(f"  {year}{half}: present, skipping ({out.name}, "
              f"{out.stat().st_size/1e6:.1f} MB)")
        return out, False

    months = H1 if half == "H1" else H2
    print(f"  {year}{half}: requesting months {months[0]}-{months[-1]} "
          f"(queues on CDS, often several minutes)")
    client.retrieve(DATASET, {
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
    }, str(out))
    print(f"  {year}{half}: saved -> {out.name} ({out.stat().st_size/1e6:.1f} MB)")
    return out, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true",
                    help="print what is needed and what is missing, fetch nothing")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    needed = required_halves()

    print("ERA5 daily climate fetch")
    print(f"  dataset:   {DATASET}")
    print(f"  area:      N={AREA[0]} W={AREA[1]} S={AREA[2]} E={AREA[3]} (CONUS)")
    print(f"  seasons:   {', '.join(SEASONS)}")
    print(f"  halves:    {len(needed)} required\n")

    missing = [(y, h) for y, h in needed if not target_path(y, h).exists()]
    for y, h in needed:
        state = "MISSING" if (y, h) in missing else "present"
        print(f"    {y}{h}  {state}")
    print(f"\n  {len(missing)} to download, "
          f"{len(needed) - len(missing)} already present")

    if args.list:
        return
    if not missing:
        print("\nNothing to do.")
        return

    print(f"\n  Estimated time: ~{len(missing) * 40} minutes "
          f"(CDS queues each request). Safe to leave running.\n")

    try:
        import cdsapi
    except ImportError:
        sys.exit("cdsapi not installed. Run: pip install \"cdsapi>=0.7.7\"")

    try:
        client = cdsapi.Client()
    except Exception as e:
        sys.exit(f"Could not create CDS client: {e}\n"
                 f"Check that .cdsapirc exists and holds a Personal Access Token.")

    ok = 0
    for year, half in missing:
        try:
            _, fetched = fetch(client, year, half)
            ok += int(fetched)
        except Exception as e:
            print(f"  {year}{half}: FAILED - {e}")
            msg = str(e).lower()
            if "licence" in msg or "403" in msg:
                print("\n  -> Licence not accepted. Open the dataset's Download tab")
                print("     and accept the terms at the bottom of the page.")
                break
            if "cost" in msg:
                print("\n  -> Still too large. Split into quarters in this script.")
                break

    print(f"\nDownloaded {ok} / {len(missing)}")
    print("Next: python src/preprocessing/unzip_era5.py")


if __name__ == "__main__":
    main()