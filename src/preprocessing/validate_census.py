#!/usr/bin/env python3
"""
EpiSurge - Phase 1: Validate Census -> HHS region aggregation before
it feeds the gravity graph.

Checks:
  1. All 50 states + DC present
  2. No state assigned to multiple HHS regions
  3. No Census jurisdiction silently dropped
  4. Region populations sum to national total
  5. Centroids inside CONUS bounds (flags PR/AK/HI effects)
  6. Exactly 10 HHS regions
  7. Puerto Rico inclusion flagged explicitly (affects Region 2 centroid)
  8. Region 2 centroid recomputed with and without PR, for comparison
"""

import csv
import json
import sys
from pathlib import Path

CENSUS_DIR = Path("data/raw/census")

EXPECTED_STATES = {
    "Alabama","Alaska","Arizona","Arkansas","California","Colorado","Connecticut",
    "Delaware","District of Columbia","Florida","Georgia","Hawaii","Idaho","Illinois",
    "Indiana","Iowa","Kansas","Kentucky","Louisiana","Maine","Maryland","Massachusetts",
    "Michigan","Minnesota","Mississippi","Missouri","Montana","Nebraska","Nevada",
    "New Hampshire","New Jersey","New Mexico","New York","North Carolina","North Dakota",
    "Ohio","Oklahoma","Oregon","Pennsylvania","Rhode Island","South Carolina",
    "South Dakota","Tennessee","Texas","Utah","Vermont","Virginia","Washington",
    "West Virginia","Wisconsin","Wyoming",
}

STATE_TO_HHS = {
    "Connecticut":1,"Maine":1,"Massachusetts":1,"New Hampshire":1,"Rhode Island":1,"Vermont":1,
    "New Jersey":2,"New York":2,
    "Delaware":3,"District of Columbia":3,"Maryland":3,"Pennsylvania":3,"Virginia":3,"West Virginia":3,
    "Alabama":4,"Florida":4,"Georgia":4,"Kentucky":4,"Mississippi":4,
    "North Carolina":4,"South Carolina":4,"Tennessee":4,
    "Illinois":5,"Indiana":5,"Michigan":5,"Minnesota":5,"Ohio":5,"Wisconsin":5,
    "Arkansas":6,"Louisiana":6,"New Mexico":6,"Oklahoma":6,"Texas":6,
    "Iowa":7,"Kansas":7,"Missouri":7,"Nebraska":7,
    "Colorado":8,"Montana":8,"North Dakota":8,"South Dakota":8,"Utah":8,"Wyoming":8,
    "Arizona":9,"California":9,"Hawaii":9,"Nevada":9,
    "Alaska":10,"Idaho":10,"Oregon":10,"Washington":10,
}

STATE_CENTROIDS = {
    "New Jersey":(40.298904,-74.521011), "New York":(42.165726,-74.948051),
    "Puerto Rico":(18.220833,-66.590149),
}

failures, warnings = [], []


def check(cond, msg, hard=True):
    if not cond:
        (failures if hard else warnings).append(msg)
    return cond


def main():
    with open(CENSUS_DIR / "state_populations.json") as f:
        state_pops = json.load(f)

    with open(CENSUS_DIR / "hhs_region_population.csv") as f:
        regions = list(csv.DictReader(f))

    print(f"Census jurisdictions: {len(state_pops)}")
    print(f"HHS region rows:      {len(regions)}\n")

    # 1 - all expected states present
    missing = EXPECTED_STATES - set(state_pops)
    check(not missing, f"Missing states: {sorted(missing)}")

    # 2 - no state in multiple regions (dict keys enforce this, verify anyway)
    check(len(STATE_TO_HHS) == len(set(STATE_TO_HHS)), "Duplicate state keys in mapping")

    # 3 - nothing silently dropped (PR excluded by design - see DECISIONS.md)
    INTENTIONALLY_EXCLUDED = {"Puerto Rico"}
    unmapped = set(state_pops) - set(STATE_TO_HHS) - INTENTIONALLY_EXCLUDED
    check(not unmapped, f"Census jurisdictions not mapped to any HHS region: {sorted(unmapped)}")
    if INTENTIONALLY_EXCLUDED & set(state_pops):
        print(f"Intentionally excluded: {sorted(INTENTIONALLY_EXCLUDED & set(state_pops))} "
              f"(see DECISIONS.md)")

    # 4 - populations sum correctly
    mapped_total = sum(p for s, p in state_pops.items() if s in STATE_TO_HHS)
    region_total = sum(int(r["population"]) for r in regions)
    check(mapped_total == region_total,
          f"Population mismatch: states={mapped_total:,} vs regions={region_total:,}")
    print(f"Total mapped population: {region_total:,}")

    # 5 - exactly 10 regions, numbered 1-10
    ids = sorted(int(r["hhs_region"]) for r in regions)
    check(ids == list(range(1, 11)), f"Expected regions 1-10, got {ids}")

    # 6 - centroid sanity
    for r in regions:
        rid = int(r["hhs_region"])
        lat, lon = float(r["centroid_lat"]), float(r["centroid_lon"])
        check(-180 <= lon <= 180 and -90 <= lat <= 90,
              f"Region {rid} centroid out of valid range: ({lat}, {lon})")
        if not (24 <= lat <= 50 and -125 <= lon <= -66):
            warnings.append(f"Region {rid} centroid outside CONUS bounds: ({lat}, {lon}) "
                            f"- expected for regions containing AK/HI/PR")

    # 7/8 - Puerto Rico impact on Region 2
    pr_pop = state_pops.get("Puerto Rico")
    if pr_pop:
        print(f"\n--- Puerto Rico impact on Region 2 ---")
        ny, nj = state_pops["New York"], state_pops["New Jersey"]

        def centroid(items):
            tot = sum(p for _, p in items)
            lat = sum(STATE_CENTROIDS[s][0] * p for s, p in items) / tot
            lon = sum(STATE_CENTROIDS[s][1] * p for s, p in items) / tot
            return lat, lon, tot

        with_pr = centroid([("New York", ny), ("New Jersey", nj), ("Puerto Rico", pr_pop)])
        without_pr = centroid([("New York", ny), ("New Jersey", nj)])

        print(f"  With PR:    pop={with_pr[2]:>12,}  centroid=({with_pr[0]:.3f}, {with_pr[1]:.3f})")
        print(f"  Without PR: pop={without_pr[2]:>12,}  centroid=({without_pr[0]:.3f}, {without_pr[1]:.3f})")
        print(f"  Latitude shift caused by PR: {abs(with_pr[0]-without_pr[0]):.3f} degrees")
        warnings.append(
            "Puerto Rico EXCLUDED from Region 2 (see DECISIONS.md). Comparison "
            "above documents the magnitude of that exclusion."
        )

    # --- report ---
    print(f"\n{'='*60}")
    if failures:
        print(f"FAILED ({len(failures)}):")
        for m in failures:
            print(f"  [FAIL] {m}")
    else:
        print("All hard checks PASSED")

    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for m in warnings:
            print(f"  [WARN] {m}")

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()