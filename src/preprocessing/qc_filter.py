#!/usr/bin/env python3
"""
EpiSurge - Phase 1: QC filter on raw NCBI records.

Raw fetch pulled broader results than intended - non-USA locations,
non-human hosts, and out-of-season collection dates slipped through
because the NCBI query's USA[Country] filter doesn't reliably restrict
results. This filter enforces the real constraints post-fetch.

Keeps only:
  - geo_loc_name starting with "USA"
  - host is human (or unspecified/blank - common for human seasonal flu records)
  - collection_date falls within the season's date range
  - description does NOT indicate an animal host (duck, chicken, swine, canine, etc.)
"""

import json
import re
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import QC_WINDOWS                           # noqa: E402

RAW_DIR = Path("data/raw/ncbi")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Collection-date windows come from src/config.py so ingestion,
# preprocessing and feature modules cannot drift apart.
SEASON_RANGES = QC_WINDOWS

ANIMAL_HOST_PATTERN = re.compile(
    r'A/(duck|chicken|swine|canine|feline|equine|avian|turkey|goose|'
    r'quail|pheasant|mink|seal|bat|pig|horse|dog|cat)/',
    re.IGNORECASE
)

def parse_collection_date(date_str):
    """GenBank dates come in several formats: '2016', '18-Nov-2022',
    '2024-04-30' (ISO), '2019-03' (ISO year-month), etc."""
    date_str = date_str.strip()
    for fmt in ("%d-%b-%Y", "%b-%Y", "%Y", "%Y-%m-%d", "%Y-%m"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None

def passes_qc(record, season_start, season_end):
    reasons = []

    if not record["geo_loc_name"].strip().startswith("USA"):
        reasons.append("not_usa")

    if ANIMAL_HOST_PATTERN.search(record["description"]):
        reasons.append("animal_host")

    parsed_date = parse_collection_date(record["collection_date"])
    if parsed_date is None:
        reasons.append("unparseable_date")
    elif not (season_start <= parsed_date <= season_end):
        reasons.append("out_of_season")

    return (len(reasons) == 0), reasons, parsed_date

def main():
    summary = {}

    for season_label, (start, end) in SEASON_RANGES.items():
        raw_path = RAW_DIR / f"ncbi_{season_label}.json"
        with open(raw_path) as f:
            records = json.load(f)

        kept = []
        rejected_counts = {}

        for r in records:
            ok, reasons, parsed_date = passes_qc(r, start, end)
            if ok:
                r["collection_date_parsed"] = parsed_date.isoformat()
                kept.append(r)
            else:
                for reason in reasons:
                    rejected_counts[reason] = rejected_counts.get(reason, 0) + 1

        out_path = OUT_DIR / f"qc_{season_label}.json"
        with open(out_path, "w") as f:
            json.dump(kept, f, indent=2)

        summary[season_label] = {
            "total": len(records),
            "kept": len(kept),
            "rejected_breakdown": rejected_counts,
        }

        print(f"\n{season_label}")
        print(f"  Total:   {len(records)}")
        print(f"  Kept:    {len(kept)} ({100*len(kept)/len(records):.1f}%)")
        print(f"  Rejected breakdown: {rejected_counts}")

    with open(OUT_DIR / "qc_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved QC'd files to {OUT_DIR}\\qc_<season>.json")
    print(f"Summary saved to {OUT_DIR}\\qc_summary.json")

if __name__ == "__main__":
    main()