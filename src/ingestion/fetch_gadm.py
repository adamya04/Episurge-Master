#!/usr/bin/env python3
"""
EpiSurge - Phase 1: GADM US state boundaries -> HHS region polygons.

Needed to aggregate ERA5's 0.25-degree climate grid into per-HHS-region
weekly values (point-in-polygon / spatial join).

GADM level 1 = states/provinces. We download the US file, dissolve the
50 states + DC into 10 HHS region polygons, and save as GeoJSON.

No account or API key required - direct download.
"""

import sys
import zipfile
from pathlib import Path

import requests

try:
    import geopandas as gpd
except ImportError:
    sys.exit("geopandas not installed. Run: pip install geopandas")

RAW_DIR = Path("data/raw/gadm")
RAW_DIR.mkdir(parents=True, exist_ok=True)

GADM_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_USA_1.json.zip"
ZIP_PATH = RAW_DIR / "gadm41_USA_1.json.zip"
JSON_PATH = RAW_DIR / "gadm41_USA_1.json"
OUT_PATH = RAW_DIR / "hhs_regions.geojson"

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


def download():
    if JSON_PATH.exists():
        print(f"Already downloaded: {JSON_PATH.name} "
              f"({JSON_PATH.stat().st_size/1e6:.1f} MB)")
        return

    print(f"Downloading GADM US level-1 boundaries...")
    print(f"  {GADM_URL}")

    resp = requests.get(GADM_URL, timeout=300, stream=True)
    resp.raise_for_status()

    with open(ZIP_PATH, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)

    print(f"  Downloaded {ZIP_PATH.stat().st_size/1e6:.1f} MB, extracting...")

    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(RAW_DIR)

    ZIP_PATH.unlink()
    print(f"  Extracted -> {JSON_PATH.name}")


def build_hhs_polygons():
    print(f"\nLoading state geometries...")
    gdf = gpd.read_file(JSON_PATH)
    print(f"  Loaded {len(gdf)} level-1 units")
    print(f"  CRS: {gdf.crs}")

    name_col = "NAME_1" if "NAME_1" in gdf.columns else gdf.columns[0]

    # GADM strips spaces from state names ("New York" -> "NewYork"),
    # so normalise both sides of the lookup.
    normalised_lookup = {
        state.replace(" ", ""): hhs for state, hhs in STATE_TO_HHS.items()
    }
    gdf["hhs_region"] = gdf[name_col].str.replace(" ", "", regex=False).map(normalised_lookup)

    unmapped = sorted(gdf.loc[gdf["hhs_region"].isna(), name_col].unique())
    if unmapped:
        print(f"  Unmapped (excluded): {unmapped}")

    mapped = gdf.dropna(subset=["hhs_region"]).copy()
    mapped["hhs_region"] = mapped["hhs_region"].astype(int)
    print(f"  Mapped {len(mapped)} states to HHS regions")

    print(f"\nDissolving states into HHS region polygons...")
    regions = mapped.dissolve(by="hhs_region", as_index=False)[["hhs_region", "geometry"]]
    regions = regions.sort_values("hhs_region").reset_index(drop=True)

    regions.to_file(OUT_PATH, driver="GeoJSON")

    print(f"\nHHS region polygons:")
    for _, row in regions.iterrows():
        b = row.geometry.bounds  # minx, miny, maxx, maxy
        print(f"  Region {int(row.hhs_region):>2}: "
              f"bounds lon[{b[0]:>8.2f}, {b[2]:>8.2f}] "
              f"lat[{b[1]:>6.2f}, {b[3]:>6.2f}]")

    print(f"\nSaved -> {OUT_PATH}")
    print(f"  Regions: {len(regions)}")

    if len(regions) != 10:
        print(f"  [WARN] Expected 10 regions, got {len(regions)}")


def main():
    download()
    build_hhs_polygons()


if __name__ == "__main__":
    main()