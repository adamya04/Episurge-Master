#!/usr/bin/env python3
"""
EpiSurge - Phase 1: NCBI Influenza A HA Sequence Fetcher

Fetches real GenBank records (not FASTA headers) to get verified
geo_loc_name, collection_date and create_date per sequence.

Field-name bugs found in prior work, fixed and not to be regressed:
  - Use [PDAT], NOT [Collection Date]        ([Collection Date] is not a
    valid field and returns zero results silently)
  - Use geo_loc_name, NOT country            (NCBI renamed the qualifier)
  - Parse fetched records, never the search index, for geography

FETCH-WINDOW CORRECTION (2026-10-29)
------------------------------------
This script previously searched publication dates over the same Oct-May
months that qc_filter.py uses for COLLECTION dates. Because publication
lags collection by a median of ~76 days, every sequence collected late in
a season was published after the window closed and was never retrieved.

The censoring was visible in the retrieved data: maximum observed delay
tracked days-remaining-to-window-close month by month in all three
seasons, and April/May collections were near zero (2019-20 had none at
all) in seasons whose core evaluation window runs to mid-May.

Publication windows now extend FETCH_TAIL_MONTHS (24) past the collection
window, so a sequence collected on the last day of the season still has
two years to appear. Run verify_fetch_window.py after this completes - it
tests whether the censoring signature is actually gone rather than
assuming the wider window was enough.

Seasons and windows come from src/config.py. Do not redefine them here.

Resumable: a season whose output file already exists is skipped. Delete
the file to refetch it.

Run (from the repository root, inside the venv):
    python src/ingestion/fetch_ncbi.py --email you@example.com
    python src/ingestion/fetch_ncbi.py --email you@example.com --seasons 2015-16 2016-17
"""

import argparse
import json
import sys
import time
from pathlib import Path

from Bio import Entrez, SeqIO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import SEASON_LABELS, fetch_window          # noqa: E402

OUT_DIR = Path("data/raw/ncbi")

QUERY_TEMPLATE = (
    '"Influenza A virus"[Organism] AND HA[Gene] AND USA[Country] '
    'AND ("{start}"[PDAT] : "{end}"[PDAT])'
)

RETMAX = 100000          # raised from 50000: wider windows return more ids
BATCH_SIZE = 200


def search_ids(start, end):
    query = QUERY_TEMPLATE.format(start=start, end=end)
    handle = Entrez.esearch(db="nucleotide", term=query, retmax=RETMAX)
    record = Entrez.read(handle)
    handle.close()
    ids = record["IdList"]
    total = int(record["Count"])
    if total > len(ids):
        print(f"  [WARN] {total} records match but only {len(ids)} ids returned; "
              f"raise RETMAX above {RETMAX}")
    return ids


def fetch_records(id_list, season_label, pause):
    all_records = []
    for i in range(0, len(id_list), BATCH_SIZE):
        batch = id_list[i:i + BATCH_SIZE]
        print(f"  [{season_label}] fetching {i}-{i + len(batch)} / {len(id_list)}")
        for attempt in range(3):
            try:
                handle = Entrez.efetch(db="nucleotide", id=batch,
                                       rettype="gb", retmode="text")
                for record in SeqIO.parse(handle, "genbank"):
                    src = next((f for f in record.features if f.type == "source"), None)
                    all_records.append({
                        "accession": record.id,
                        "geo_loc_name": (src.qualifiers.get("geo_loc_name", [""])[0]
                                         if src else ""),
                        "collection_date": (src.qualifiers.get("collection_date", [""])[0]
                                            if src else ""),
                        # NOTE: this is the record's LAST-MODIFIED date, not the
                        # submission date. True submission dates come from
                        # fetch_submission_dates.py, which parses the REFERENCE
                        # block. Kept only for provenance.
                        "create_date": record.annotations.get("date", ""),
                        "organism": record.annotations.get("organism", ""),
                        "sequence_length": len(record.seq),
                        "description": record.description,
                    })
                handle.close()
                break
            except Exception as e:
                wait = 5 * (attempt + 1)
                print(f"    [retry {attempt + 1}/3] {type(e).__name__}: {e} "
                      f"- sleeping {wait}s")
                time.sleep(wait)
        else:
            print(f"    [FAIL] batch at {i} skipped after 3 attempts")
        time.sleep(pause)
    return all_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True, help="contact email for NCBI Entrez")
    ap.add_argument("--api-key", default=None,
                    help="NCBI API key; raises the rate limit from 3/s to 10/s")
    ap.add_argument("--seasons", nargs="*", default=None,
                    help="subset of season labels; default is all in config")
    ap.add_argument("--force", action="store_true",
                    help="refetch seasons whose output file already exists")
    args = ap.parse_args()

    Entrez.email = args.email
    if args.api_key:
        Entrez.api_key = args.api_key
    pause = 0.12 if args.api_key else 0.40

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = args.seasons or SEASON_LABELS
    unknown = [s for s in targets if s not in SEASON_LABELS]
    if unknown:
        sys.exit(f"Unknown season label(s): {unknown}. Known: {SEASON_LABELS}")

    print("EpiSurge - NCBI fetch")
    print("=" * 68)
    print(f"Seasons: {', '.join(targets)}")
    print("Publication windows extend 24 months past each collection window "
          "so the\ndelay distribution is not truncated. Collection dates are "
          "filtered later,\nin qc_filter.py.\n")

    for label in targets:
        out_path = OUT_DIR / f"ncbi_{label}.json"
        if out_path.exists() and not args.force:
            with open(out_path) as f:
                n = len(json.load(f))
            print(f"[{label}] already fetched: {n} records in {out_path.name} "
                  f"(--force to refetch)\n")
            continue

        start, end = fetch_window(label)
        print(f"[{label}] searching publication dates {start} .. {end}")
        ids = search_ids(start, end)
        print(f"  found {len(ids)} sequence ids")

        records = fetch_records(ids, label, pause)
        with open(out_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"  saved {len(records)} records -> {out_path}\n")

    print("Done. Next: fetch_submission_dates.py, then verify_fetch_window.py")


if __name__ == "__main__":
    main()