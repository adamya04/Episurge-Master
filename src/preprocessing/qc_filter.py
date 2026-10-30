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

# Non-human host detection.
#
# This was a blocklist of 18 animal names. It missed 146 distinct hosts -
# cattle, dairy cow, bovine, and every wild bird species - letting 2,213
# records (6.3%) through, 1,743 of them H5N1 from the 2024 cattle and avian
# panzootic. The damage was concentrated in 2022-23 (1,012 records) and
# 2023-24 (772), so the contamination was season-specific and would have
# produced spurious between-season variation in the composition features.
# Measured effect: the "other" subtype share was 16.0% in 2022-23 and 6.1%
# in 2023-24, and fell to 0.0% in both once animal hosts were excluded -
# i.e. the entire category was contamination, not a human subtype.
#
# Replaced with a STRUCTURAL rule from influenza strain nomenclature:
#     human   A/<place>/<id>/<year>            -> 4 slash-separated fields
#     animal  A/<host>/<place>/<id>/<year>     -> 5 or more
# A named host means a non-human isolate. Verified against all 146 hosts
# found in the data (cattle, mallard, environment, bos taurus, ...) with
# zero human false positives.
STRAIN_NAME = re.compile(r'\((A/[^)]+)\)')

# Some strains name the host explicitly as "human" - A/human/Texas/.../2024 -
# chiefly the 2024 H5N1 spillover cases. The structural rule alone flagged
# 2,230 of these as non-human, which is wrong: the host IS human. Whether a
# human H5N1 spillover case belongs in a SEASONAL composition is a separate
# question, handled by subtype rather than by host.
HUMAN_HOST_LABELS = {"human", "homo sapiens", "h. sapiens", "sapiens"}


def named_host(description):
    """
    Return the host name if the strain is from a non-human host, else None.

    Influenza strain nomenclature:
        human   A/<place>/<id>/<year>          -> 4 slash-separated fields
        animal  A/<host>/<place>/<id>/<year>   -> 5 or more
    A named host means non-human, EXCEPT when that host is explicitly human.
    """
    m = STRAIN_NAME.search(description)
    if not m:
        return None
    parts = m.group(1).split("/")
    if len(parts) < 5:
        return None
    host = parts[1].strip()
    if host.lower() in HUMAN_HOST_LABELS:
        return None
    return host

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

    host = named_host(record["description"])
    if host:
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
        rejected_hosts = {}

        for r in records:
            ok, reasons, parsed_date = passes_qc(r, start, end)
            if ok:
                r["collection_date_parsed"] = parsed_date.isoformat()
                kept.append(r)
            else:
                for reason in reasons:
                    rejected_counts[reason] = rejected_counts.get(reason, 0) + 1
                if "animal_host" in reasons:
                    h = named_host(r["description"]).lower()
                    rejected_hosts[h] = rejected_hosts.get(h, 0) + 1

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
        if rejected_hosts:
            top = sorted(rejected_hosts.items(), key=lambda kv: -kv[1])[:5]
            print(f"  Non-human hosts excluded: {len(rejected_hosts)} distinct, "
                  f"top: {', '.join(f'{h} ({n})' for h, n in top)}")

    with open(OUT_DIR / "qc_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved QC'd files to {OUT_DIR}\\qc_<season>.json")
    print(f"Summary saved to {OUT_DIR}\\qc_summary.json")

if __name__ == "__main__":
    main()