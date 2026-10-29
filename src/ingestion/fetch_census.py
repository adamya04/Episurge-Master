#!/usr/bin/env python3

"""
EpiSurge - Phase 1: US Census state populations -> HHS region populations.

Uses the U.S. Census Bureau Population Estimates Program (PEP)
2023 state-level population estimates.

Pipeline:
    Census state populations
        ↓
    HHS region mapping
        ↓
    HHS region populations
        ↓
    Population-weighted region centroids

These outputs are later used by the gravity graph:

    w_ij ~ (P_i * P_j) / distance^alpha
"""

import csv
import json
import os
from pathlib import Path

import requests


# Paths

RAW_DIR = Path("data/raw/census")
RAW_DIR.mkdir(parents=True, exist_ok=True)


# Census API

API = "https://api.census.gov/data/2023/pep/charv"


# State to HHS region mapping

STATE_TO_HHS = {
    "Connecticut": 1,
    "Maine": 1,
    "Massachusetts": 1,
    "New Hampshire": 1,
    "Rhode Island": 1,
    "Vermont": 1,

    "New Jersey": 2,
    "New York": 2,

    "Delaware": 3,
    "District of Columbia": 3,
    "Maryland": 3,
    "Pennsylvania": 3,
    "Virginia": 3,
    "West Virginia": 3,

    "Alabama": 4,
    "Florida": 4,
    "Georgia": 4,
    "Kentucky": 4,
    "Mississippi": 4,
    "North Carolina": 4,
    "South Carolina": 4,
    "Tennessee": 4,

    "Illinois": 5,
    "Indiana": 5,
    "Michigan": 5,
    "Minnesota": 5,
    "Ohio": 5,
    "Wisconsin": 5,

    "Arkansas": 6,
    "Louisiana": 6,
    "New Mexico": 6,
    "Oklahoma": 6,
    "Texas": 6,

    "Iowa": 7,
    "Kansas": 7,
    "Missouri": 7,
    "Nebraska": 7,

    "Colorado": 8,
    "Montana": 8,
    "North Dakota": 8,
    "South Dakota": 8,
    "Utah": 8,
    "Wyoming": 8,

    "Arizona": 9,
    "California": 9,
    "Hawaii": 9,
    "Nevada": 9,

    "Alaska": 10,
    "Idaho": 10,
    "Oregon": 10,
    "Washington": 10,
}


# State centroids used for population-weighted HHS region centroids

STATE_CENTROIDS = {
    "Alabama": (32.806671, -86.791130),
    "Alaska": (61.370716, -152.404419),
    "Arizona": (33.729759, -111.431221),
    "Arkansas": (34.969704, -92.373123),
    "California": (36.116203, -119.681564),
    "Colorado": (39.059811, -105.311104),
    "Connecticut": (41.597782, -72.755371),
    "Delaware": (39.318523, -75.507141),
    "District of Columbia": (38.897438, -77.026817),
    "Florida": (27.766279, -81.686783),
    "Georgia": (33.040619, -83.643074),
    "Hawaii": (21.094318, -157.498337),
    "Idaho": (44.240459, -114.478828),
    "Illinois": (40.349457, -88.986137),
    "Indiana": (39.849426, -86.258278),
    "Iowa": (42.011539, -93.210526),
    "Kansas": (38.526600, -96.726486),
    "Kentucky": (37.668140, -84.670067),
    "Louisiana": (31.169546, -91.867805),
    "Maine": (44.693947, -69.381927),
    "Maryland": (39.063946, -76.802101),
    "Massachusetts": (42.230171, -71.530106),
    "Michigan": (43.326618, -84.536095),
    "Minnesota": (45.694454, -93.900192),
    "Mississippi": (32.741646, -89.678696),
    "Missouri": (38.456085, -92.288368),
    "Montana": (46.921925, -110.454353),
    "Nebraska": (41.125370, -98.268082),
    "Nevada": (38.313515, -117.055374),
    "New Hampshire": (43.452492, -71.563896),
    "New Jersey": (40.298904, -74.521011),
    "New Mexico": (34.840515, -106.248482),
    "New York": (42.165726, -74.948051),
    "North Carolina": (35.630066, -79.806419),
    "North Dakota": (47.528912, -99.784012),
    "Ohio": (40.388783, -82.764915),
    "Oklahoma": (35.565342, -96.928917),
    "Oregon": (44.572021, -122.070938),
    "Pennsylvania": (40.590752, -77.209755),
    "Puerto Rico": (18.220833, -66.590149),
    "Rhode Island": (41.680893, -71.511780),
    "South Carolina": (33.856892, -80.945007),
    "South Dakota": (44.299782, -99.438828),
    "Tennessee": (35.747845, -86.692345),
    "Texas": (31.054487, -97.563461),
    "Utah": (40.150032, -111.862434),
    "Vermont": (44.045876, -72.710686),
    "Virginia": (37.769337, -78.169968),
    "Washington": (47.400902, -121.490494),
    "West Virginia": (38.491226, -80.954453),
    "Wisconsin": (44.268543, -89.616508),
    "Wyoming": (42.755966, -107.302490),
}


# Fetch state populations from the Census API

def fetch_state_populations():
    api_key = os.getenv("CENSUS_API_KEY")

    if not api_key:
        raise RuntimeError(
            "CENSUS_API_KEY environment variable is not set."
        )

    params = {
        "get": "NAME,POP",
        "for": "state:*",
        "MONTH": "7",
        "YEAR": "2023",
        "UNIVERSE": "R",
        "AGE": "0000",
        "SEX": "0",
        "key": api_key,
    }

    print("Requesting Census state populations (2023 PEP)...")

    response = requests.get(
        API,
        params=params,
        timeout=60,
    )

    print(f"  HTTP status: {response.status_code}")

    if not response.ok:
        raise RuntimeError(
            f"Census API returned HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )

    try:
        rows = response.json()
    except ValueError:
        raise RuntimeError(
            "Census API did not return valid JSON.\n"
            f"Response: {response.text[:500]}"
        )

    if len(rows) < 2:
        raise RuntimeError(
            f"Unexpected Census API response: {rows}"
        )

    header = rows[0]
    data = rows[1:]

    name_i = header.index("NAME")
    pop_i = header.index("POP")

    return {
        row[name_i]: int(row[pop_i])
        for row in data
    }


# Aggregate state populations into HHS regions

def build_hhs_regions(state_pops):
    region_pop = {}
    region_lat_num = {}
    region_lon_num = {}
    unmatched = []

    for state, population in state_pops.items():
        hhs = STATE_TO_HHS.get(state)

        if hhs is None:
            unmatched.append(state)
            continue

        centroid = STATE_CENTROIDS.get(state)

        if centroid is None:
            unmatched.append(f"{state} (no centroid)")
            continue

        lat, lon = centroid

        region_pop[hhs] = (
            region_pop.get(hhs, 0) + population
        )

        region_lat_num[hhs] = (
            region_lat_num.get(hhs, 0) + lat * population
        )

        region_lon_num[hhs] = (
            region_lon_num.get(hhs, 0) + lon * population
        )

    return (
        region_pop,
        region_lat_num,
        region_lon_num,
        unmatched,
    )


# Save HHS region populations and centroids

def save_hhs_regions(
    region_pop,
    region_lat_num,
    region_lon_num,
):
    output_path = RAW_DIR / "hhs_region_population.csv"

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "hhs_region",
                "population",
                "centroid_lat",
                "centroid_lon",
            ]
        )

        for hhs in sorted(region_pop):
            population = region_pop[hhs]

            centroid_lat = (
                region_lat_num[hhs] / population
            )

            centroid_lon = (
                region_lon_num[hhs] / population
            )

            writer.writerow(
                [
                    hhs,
                    population,
                    round(centroid_lat, 6),
                    round(centroid_lon, 6),
                ]
            )

    return output_path


# Main pipeline

def main():
    try:
        state_pops = fetch_state_populations()

        print(
            f"  Retrieved {len(state_pops)} "
            "states/territories"
        )

    except Exception as e:
        print(f"[ERROR] Census API failed: {e}")
        return

    # Save the raw Census data

    state_output = RAW_DIR / "state_populations.json"

    with open(
        state_output,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(state_pops, f, indent=2)

    print(f"Saved raw populations -> {state_output}")

    # Build HHS region populations and centroids

    (
        region_pop,
        region_lat_num,
        region_lon_num,
        unmatched,
    ) = build_hhs_regions(state_pops)

    # Save the aggregated HHS data

    output_path = save_hhs_regions(
        region_pop,
        region_lat_num,
        region_lon_num,
    )

    # Display the results

    print("\nHHS region populations:")

    for hhs in sorted(region_pop):
        population = region_pop[hhs]

        lat = (
            region_lat_num[hhs] / population
        )

        lon = (
            region_lon_num[hhs] / population
        )

        print(
            f"  Region {hhs:>2}: "
            f"{population:>12,} "
            f"({lat:.2f}, {lon:.2f})"
        )

    if unmatched:
        print(
            f"\nUnmatched (excluded): {unmatched}"
        )

    print(f"\nSaved -> {output_path}")


if __name__ == "__main__":
    main()