#!/usr/bin/env python3
"""
EpiSurge - Phase 1: Quick sanity check on raw NCBI fetch output.
Run this before building the QC pipeline, to catch field-population
issues early (geo_loc_name / collection_date were buggy in prior work).
"""

import json
from pathlib import Path

RAW_DIR = Path("data/raw/ncbi")

def inspect(file_path):
    with open(file_path) as f:
        records = json.load(f)

    total = len(records)
    has_geo = sum(1 for r in records if r["geo_loc_name"].strip())
    has_collection_date = sum(1 for r in records if r["collection_date"].strip())
    has_create_date = sum(1 for r in records if r["create_date"].strip())

    print(f"\n{file_path.name}")
    print(f"  Total records:         {total}")
    print(f"  Has geo_loc_name:      {has_geo} ({100*has_geo/total:.1f}%)")
    print(f"  Has collection_date:   {has_collection_date} ({100*has_collection_date/total:.1f}%)")
    print(f"  Has create_date:       {has_create_date} ({100*has_create_date/total:.1f}%)")

    print("  Sample record:")
    print(f"    {records[0]}")

def main():
    for file_path in sorted(RAW_DIR.glob("ncbi_*.json")):
        inspect(file_path)

if __name__ == "__main__":
    main()