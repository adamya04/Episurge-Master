#!/usr/bin/env python3
"""
EpiSurge - Phase 1: Inspect which collection_date formats are failing
to parse, so we can fix the parser instead of silently dropping data.
"""

import json
from pathlib import Path
from collections import Counter
from datetime import datetime

RAW_DIR = Path("data/raw/ncbi")

def try_parse(date_str):
    date_str = date_str.strip()
    for fmt in ("%d-%b-%Y", "%b-%Y", "%Y"):
        try:
            datetime.strptime(date_str, fmt)
            return True
        except ValueError:
            continue
    return False

def main():
    for season in ["2019-20", "2022-23", "2023-24"]:
        with open(RAW_DIR / f"ncbi_{season}.json") as f:
            records = json.load(f)

        failed_formats = Counter()
        failed_examples = {}

        for r in records:
            date_str = r["collection_date"].strip()
            if date_str and not try_parse(date_str):
                # bucket by rough shape, not exact string, to find patterns
                shape = "".join("D" if c.isdigit() else ("A" if c.isalpha() else c) for c in date_str)
                failed_formats[shape] += 1
                if shape not in failed_examples:
                    failed_examples[shape] = date_str

        print(f"\n{season} - unparseable date shapes:")
        for shape, count in failed_formats.most_common(10):
            print(f"  {shape:20s} count={count:5d}  example='{failed_examples[shape]}'")

if __name__ == "__main__":
    main()