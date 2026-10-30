#!/usr/bin/env python3
"""
EpiSurge - Phase 1: Map QC'd records to (hhs_region, epiweek).

geo_loc_name looks like "USA: California" or "USA: Colorado, Boulder" -
we extract the state, map to HHS region, and convert collection_date
to MMWR epiweek (the standard CDC surveillance week format).
"""

import json
import re
from pathlib import Path
from datetime import datetime
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from config import SEASON_LABELS

from epiweeks import Week

PROCESSED_DIR = Path("data/processed")

# CDC HHS region groupings
STATE_TO_HHS = {
    "Connecticut": 1, "Maine": 1, "Massachusetts": 1, "New Hampshire": 1, "Rhode Island": 1, "Vermont": 1,
    "New Jersey": 2, "New York": 2, "Puerto Rico": 2, "Virgin Islands": 2,
    "Delaware": 3, "District of Columbia": 3, "Maryland": 3, "Pennsylvania": 3, "Virginia": 3, "West Virginia": 3,
    "Alabama": 4, "Florida": 4, "Georgia": 4, "Kentucky": 4, "Mississippi": 4, "North Carolina": 4, "South Carolina": 4, "Tennessee": 4,
    "Illinois": 5, "Indiana": 5, "Michigan": 5, "Minnesota": 5, "Ohio": 5, "Wisconsin": 5,
    "Arkansas": 6, "Louisiana": 6, "New Mexico": 6, "Oklahoma": 6, "Texas": 6,
    "Iowa": 7, "Kansas": 7, "Missouri": 7, "Nebraska": 7,
    "Colorado": 8, "Montana": 8, "North Dakota": 8, "South Dakota": 8, "Utah": 8, "Wyoming": 8,
    "Arizona": 9, "California": 9, "Hawaii": 9, "Nevada": 9,
    "Alaska": 10, "Idaho": 10, "Oregon": 10, "Washington": 10,
}

STATE_ABBREV_TO_NAME = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia",
    "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}
STATE_NAME_LOOKUP_LOWER = {name.lower(): name for name in STATE_TO_HHS}

def extract_state(geo_loc_name):
    """
    Handles: 'USA: California', 'USA: Colorado, Boulder' (city after state),
    'USA: Maricopa County, AZ' (abbreviation after county - checked last segment),
    'USA: WA' (bare abbreviation), 'USA: texas' (lowercase).
    """
    match = re.match(r'USA:\s*(.+)', geo_loc_name)
    if not match:
        return None

    parts = [p.strip() for p in match.group(1).split(",")]

    # try every comma-separated segment - state could be first OR last
    for part in parts:
        if part.upper() in STATE_ABBREV_TO_NAME:
            return STATE_ABBREV_TO_NAME[part.upper()]
        if part.lower() in STATE_NAME_LOOKUP_LOWER:
            return STATE_NAME_LOOKUP_LOWER[part.lower()]

    # Pass 2: forms the exact-match pass misses. A diagnostic over all seven
    # seasons found 187 recoverable records, 172 of them
    # 'USA: New York City, New York County' - dropping those removed New
    # York City from Region 2 entirely, a geographic bias rather than a
    # volume loss. The remaining failures are bare 'USA' with no state
    # recorded at source and are genuinely unrecoverable.
    for part in parts:
        p = part.strip()

        fixed = SPELLING_FIXES.get(p.lower())
        if fixed:
            return fixed

        # 'New York County' / 'New York City' -> 'New York'
        stripped = re.sub(r'\s+(County|Parish|Borough|City)$', '', p, flags=re.I).strip()
        if stripped.lower() in STATE_NAME_LOOKUP_LOWER:
            return STATE_NAME_LOOKUP_LOWER[stripped.lower()]

        # state name embedded in a regional description:
        # 'Southcentral Alaska' -> 'Alaska'. Longest first so 'West
        # Virginia' is not matched as 'Virginia'.
        for name in STATES_BY_LENGTH:
            if re.search(rf'\b{re.escape(name)}\b', p, flags=re.I):
                return name

    return None


# Misspellings observed in geo_loc_name across the seven seasons.
SPELLING_FIXES = {
    "misouri": "Missouri",
    "southcarolina": "South Carolina",
    "northcarolina": "North Carolina",
    "westvirginia": "West Virginia",
    "newyork": "New York",
    "newjersey": "New Jersey",
    "newmexico": "New Mexico",
    "newhampshire": "New Hampshire",
    "rhodeisland": "Rhode Island",
    "northdakota": "North Dakota",
    "southdakota": "South Dakota",
}

# Longest first, so 'West Virginia' wins over 'Virginia'.
STATES_BY_LENGTH = sorted(STATE_TO_HHS, key=len, reverse=True)

def mmwr_epiweek(date_obj):
    """MMWR epiweek via the epiweeks library (CDC system is the default)."""
    w = Week.fromdate(date_obj)
    return w.year, w.week

def main():
    summary = {}

    for season in SEASON_LABELS:
        with open(PROCESSED_DIR / f"qc_{season}.json") as f:
            records = json.load(f)

        mapped = []
        unmapped_states = {}

        for r in records:
            state = extract_state(r["geo_loc_name"])
            hhs_region = STATE_TO_HHS.get(state) if state else None

            if hhs_region is None:
                key = state or "NO_STATE_PARSED"
                unmapped_states[key] = unmapped_states.get(key, 0) + 1
                continue

            date_obj = datetime.fromisoformat(r["collection_date_parsed"])
            mmwr_year, mmwr_week = mmwr_epiweek(date_obj)

            r["state"] = state
            r["hhs_region"] = hhs_region
            r["mmwr_year"] = mmwr_year
            r["mmwr_week"] = mmwr_week
            r["epiweek"] = f"{mmwr_year}-W{mmwr_week:02d}"
            mapped.append(r)

        out_path = PROCESSED_DIR / f"mapped_{season}.json"
        with open(out_path, "w") as f:
            json.dump(mapped, f, indent=2)

        summary[season] = {
            "input": len(records),
            "mapped": len(mapped),
            "unmapped_states": unmapped_states,
        }

        print(f"\n{season}")
        print(f"  Input:  {len(records)}")
        print(f"  Mapped: {len(mapped)} ({100*len(mapped)/len(records):.1f}%)")
        if unmapped_states:
            print(f"  Unmapped states/territories: {unmapped_states}")

    with open(PROCESSED_DIR / "mapping_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved mapped files to data\\processed\\mapped_<season>.json")

if __name__ == "__main__":
    main()