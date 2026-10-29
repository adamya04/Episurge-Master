#!/usr/bin/env python3
"""
EpiSurge - Phase 1: Inspect raw geo_loc_name strings that failed state
extraction, to fix the parser against real formats instead of guessing.
"""

import json
import re
from pathlib import Path
from collections import Counter

PROCESSED_DIR = Path("data/processed")

def extract_state(geo_loc_name):
    match = re.match(r'USA:\s*([^,]+)', geo_loc_name)
    if match:
        return match.group(1).strip()
    return None

STATE_TO_HHS_KEYS = {
    "Connecticut","Maine","Massachusetts","New Hampshire","Rhode Island","Vermont",
    "New Jersey","New York","Puerto Rico","Virgin Islands",
    "Delaware","District of Columbia","Maryland","Pennsylvania","Virginia","West Virginia",
    "Alabama","Florida","Georgia","Kentucky","Mississippi","North Carolina","South Carolina","Tennessee",
    "Illinois","Indiana","Michigan","Minnesota","Ohio","Wisconsin",
    "Arkansas","Louisiana","New Mexico","Oklahoma","Texas",
    "Iowa","Kansas","Missouri","Nebraska",
    "Colorado","Montana","North Dakota","South Dakota","Utah","Wyoming",
    "Arizona","California","Hawaii","Nevada",
    "Alaska","Idaho","Oregon","Washington",
}

def main():
    for season in ["2022-23", "2023-24"]:
        with open(PROCESSED_DIR / f"qc_{season}.json") as f:
            records = json.load(f)

        raw_values = Counter()
        for r in records:
            state = extract_state(r["geo_loc_name"])
            if state is None or state not in STATE_TO_HHS_KEYS:
                raw_values[r["geo_loc_name"]] += 1

        print(f"\n{season} - raw geo_loc_name values that failed mapping (top 15):")
        for val, count in raw_values.most_common(15):
            print(f"  count={count:5d}  '{val}'")

if __name__ == "__main__":
    main()