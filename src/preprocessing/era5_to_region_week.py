#!/usr/bin/env python3
"""
EpiSurge - Phase 1, pipeline step 9:
ERA5 grid -> HHS region-week climate features.

Method:
  1. Assign every ERA5 grid cell to a US state via point-in-polygon
  2. Weight each cell by state_population / n_cells_in_that_state,
     so each state contributes in proportion to its population
  3. Compute absolute humidity PER CELL PER DAY before aggregating
     (nonlinear in T and Td - averaging inputs first would bias it)
  4. Aggregate daily -> MMWR epiweek per HHS region

Absolute humidity is used rather than relative humidity because it is
the established driver of influenza transmission (Shaman & Kohn, PNAS 2009).

Output: data/processed/climate_region_week.csv
"""

import csv
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from epiweeks import Week

try:
    import xarray as xr
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point
except ImportError as e:
    sys.exit(f"Missing dependency: {e}\n"
             f"Run: pip install xarray netCDF4 geopandas pandas shapely")

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

ERA5_DIR = RAW / "era5" / "extracted"
GADM_STATES = RAW / "gadm" / "gadm41_USA_1.json"
CENSUS_STATES = RAW / "census" / "state_populations.json"

OUT_PATH = PROCESSED / "climate_region_week.csv"

STATE_TO_HHS = {
    "Connecticut": 1, "Maine": 1, "Massachusetts": 1, "New Hampshire": 1,
    "Rhode Island": 1, "Vermont": 1,
    "New Jersey": 2, "New York": 2,
    "Delaware": 3, "District of Columbia": 3, "Maryland": 3, "Pennsylvania": 3,
    "Virginia": 3, "West Virginia": 3,
    "Alabama": 4, "Florida": 4, "Georgia": 4, "Kentucky": 4, "Mississippi": 4,
    "North Carolina": 4, "South Carolina": 4, "Tennessee": 4,
    "Illinois": 5, "Indiana": 5, "Michigan": 5, "Minnesota": 5, "Ohio": 5,
    "Wisconsin": 5,
    "Arkansas": 6, "Louisiana": 6, "New Mexico": 6, "Oklahoma": 6, "Texas": 6,
    "Iowa": 7, "Kansas": 7, "Missouri": 7, "Nebraska": 7,
    "Colorado": 8, "Montana": 8, "North Dakota": 8, "South Dakota": 8,
    "Utah": 8, "Wyoming": 8,
    "Arizona": 9, "California": 9, "Hawaii": 9, "Nevada": 9,
    "Alaska": 10, "Idaho": 10, "Oregon": 10, "Washington": 10,
}


def mmwr_epiweek(d):
    """MMWR epiweek as a packed integer, via the epiweeks library."""
    w = Week.fromdate(d)
    return w.year * 100 + w.week

def absolute_humidity(t2m_k, d2m_k):
    """
    Absolute humidity in g/m3 from 2m temperature and 2m dewpoint (both Kelvin).

    Magnus formula for saturation vapour pressure at the dewpoint gives the
    actual vapour pressure, then the ideal gas law for water vapour.
    """
    td_c = d2m_k - 273.15
    e_pa = 611.2 * np.exp(17.67 * td_c / (td_c + 243.5))     # vapour pressure, Pa
    return 1000.0 * e_pa / (461.5 * t2m_k)                    # g/m3


def load_era5_pairs():
    """Pair each period's temperature file with its dewpoint file."""
    files = sorted(ERA5_DIR.glob("*.nc"))
    periods = {}
    for f in files:
        tag = f.stem.split("_")[1]
        kind = "t2m" if "2m_temperature" in f.stem else "d2m"
        periods.setdefault(tag, {})[kind] = f

    incomplete = [t for t, v in periods.items() if set(v) != {"t2m", "d2m"}]
    if incomplete:
        sys.exit(f"Periods missing a variable: {incomplete}")

    print(f"  Found {len(periods)} complete periods: {sorted(periods)}")
    return periods


def build_cell_weights(sample_ds):
    """
    Assign each ERA5 grid cell to a state, then compute its weight:
        weight = state_population / n_cells_in_state

    So the weights for a state sum to that state's population, and no state
    is over-counted just because it covers more grid cells.
    """
    lat_name = "latitude" if "latitude" in sample_ds.coords else "lat"
    lon_name = "longitude" if "longitude" in sample_ds.coords else "lon"

    lats = sample_ds[lat_name].values
    lons = sample_ds[lon_name].values

    # ERA5 may use 0-360; normalise to -180..180
    lons_norm = np.where(lons > 180, lons - 360, lons)

    print(f"  Grid: {len(lats)} lat x {len(lons)} lon = {len(lats)*len(lons)} cells")

    lon_grid, lat_grid = np.meshgrid(lons_norm, lats)
    flat_lat = lat_grid.ravel()
    flat_lon = lon_grid.ravel()

    cells = gpd.GeoDataFrame(
        {"cell_idx": np.arange(flat_lat.size)},
        geometry=[Point(x, y) for x, y in zip(flat_lon, flat_lat)],
        crs="EPSG:4326",
    )

    print(f"  Loading state polygons...")
    states = gpd.read_file(GADM_STATES)
    name_col = "NAME_1" if "NAME_1" in states.columns else states.columns[0]

    # GADM strips spaces from state names
    lookup = {s.replace(" ", ""): s for s in STATE_TO_HHS}
    states["state"] = states[name_col].str.replace(" ", "", regex=False).map(lookup)
    states = states.dropna(subset=["state"])[["state", "geometry"]]

    print(f"  Spatial join ({len(cells)} cells against {len(states)} states)...")
    joined = gpd.sjoin(cells, states, how="inner", predicate="within")
    print(f"  {len(joined)} cells fall inside a state "
          f"({100*len(joined)/len(cells):.1f}% - rest are ocean/outside US)")

    with open(CENSUS_STATES) as f:
        state_pops = {k: int(v) for k, v in __import__("json").load(f).items()}

    cells_per_state = joined.groupby("state").size()

    rows = []
    for _, r in joined.iterrows():
        st = r["state"]
        pop = state_pops.get(st)
        if pop is None:
            continue
        rows.append({
            "cell_idx": int(r["cell_idx"]),
            "state": st,
            "hhs_region": STATE_TO_HHS[st],
            "weight": pop / cells_per_state[st],
        })

    weights = pd.DataFrame(rows)

    print(f"\n  Cells assigned per region:")
    for reg, grp in weights.groupby("hhs_region"):
        print(f"    Region {reg:>2}: {len(grp):>5} cells, "
              f"weight sum {grp['weight'].sum():>12,.0f}")

    return weights, len(lats), len(lons)


def main():
    print("ERA5 -> HHS region-week climate features\n")

    periods = load_era5_pairs()

    first_tag = sorted(periods)[0]
    with xr.open_dataset(periods[first_tag]["t2m"]) as ds0:
        weights, n_lat, n_lon = build_cell_weights(ds0)

    # pre-group cell indices and weights per region for fast weighted sums
    region_cells = {
        reg: (grp["cell_idx"].values, grp["weight"].values)
        for reg, grp in weights.groupby("hhs_region")
    }

    daily_records = []

    for tag in sorted(periods):
        print(f"\n  Processing period {tag}...")
        with xr.open_dataset(periods[tag]["t2m"]) as dst, \
             xr.open_dataset(periods[tag]["d2m"]) as dsd:

            tdim = "valid_time" if "valid_time" in dst.dims else "time"
            times = pd.to_datetime(dst[tdim].values)

            t2m = dst["t2m"].values     # (time, lat, lon)
            d2m = dsd["d2m"].values

            if t2m.shape != d2m.shape:
                sys.exit(f"Shape mismatch in period {tag}: {t2m.shape} vs {d2m.shape}")

            ah = absolute_humidity(t2m, d2m)
            temp_c = t2m - 273.15

            n_days = t2m.shape[0]
            ah_flat = ah.reshape(n_days, -1)
            temp_flat = temp_c.reshape(n_days, -1)

            for i in range(n_days):
                d = times[i].date()
                ew = mmwr_epiweek(d)
                for reg, (idx, w) in region_cells.items():
                    wsum = w.sum()
                    daily_records.append({
                        "hhs_region": reg,
                        "date": d.isoformat(),
                        "epiweek": ew,
                        "temp_c": float(np.dot(temp_flat[i, idx], w) / wsum),
                        "abs_humidity": float(np.dot(ah_flat[i, idx], w) / wsum),
                    })

            print(f"    {n_days} days x {len(region_cells)} regions")

    df = pd.DataFrame(daily_records)
    print(f"\n  Daily records: {len(df)}")

    weekly = (df.groupby(["hhs_region", "epiweek"])
                .agg(temp_c=("temp_c", "mean"),
                     abs_humidity=("abs_humidity", "mean"),
                     n_days=("date", "count"))
                .reset_index()
                .sort_values(["hhs_region", "epiweek"]))

    # flag partial weeks at period boundaries
    partial = weekly[weekly["n_days"] < 7]
    print(f"  Region-weeks: {len(weekly)}  ({len(partial)} with <7 days)")

    weekly.to_csv(OUT_PATH, index=False)

    print(f"\n  Temp range:     {weekly['temp_c'].min():.1f} to "
          f"{weekly['temp_c'].max():.1f} C")
    print(f"  Abs humidity:   {weekly['abs_humidity'].min():.2f} to "
          f"{weekly['abs_humidity'].max():.2f} g/m3")
    print(f"  Epiweek range:  {weekly['epiweek'].min()} to {weekly['epiweek'].max()}")
    print(f"\n  Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()