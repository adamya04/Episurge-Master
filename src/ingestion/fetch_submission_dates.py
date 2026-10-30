#!/usr/bin/env python3
"""
EpiSurge - Phase 1 fix: Fetch TRUE submission dates for mapped accessions.

record.annotations["date"] is GenBank's LAST-MODIFIED date, not the date
the sequence became public. Using it would inflate the measured sequencing
delay and bias N5. The real date is in the REFERENCE section:
  JOURNAL  Submitted (DD-MON-YYYY) Lab Name, Address

This re-fetches only the 13,807 already-mapped accessions and extracts it.
"""

import json
import time
import re
from pathlib import Path
from Bio import Entrez, SeqIO

Entrez.email = "adamyasharma7476@gmail.com"   # same as before

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
PROCESSED_DIR = Path("data/processed")
from config import SEASON_LABELS as SEASONS

SUBMITTED_PATTERN = re.compile(r'Submitted\s*\((\d{2}-[A-Z]{3}-\d{4})\)', re.IGNORECASE)

def extract_submission_date(record):
    """Pull the earliest 'Submitted (DD-MON-YYYY)' from REFERENCE entries."""
    dates = []
    for ref in record.annotations.get("references", []):
        journal = getattr(ref, "journal", "") or ""
        match = SUBMITTED_PATTERN.search(journal)
        if match:
            dates.append(match.group(1))
    return sorted(dates)[0] if dates else ""

def main():
    # collect all accessions we actually care about
    accessions = []
    for season in SEASONS:
        with open(PROCESSED_DIR / f"mapped_{season}.json") as f:
            for r in json.load(f):
                accessions.append(r["accession"])
    accessions = list(dict.fromkeys(accessions))     # dedupe, keep order

    # INCREMENTAL: keep dates already fetched and request only what is new.
    # A full refetch of ~35k accessions takes hours, and any change upstream
    # (a parser fix adding a few hundred records) previously forced one.
    out_path = PROCESSED_DIR / "submission_dates.json"
    submission_dates = {}
    if out_path.exists():
        with open(out_path) as f:
            submission_dates = json.load(f)
        # drop blanks so a previously failed lookup is retried
        submission_dates = {k: v for k, v in submission_dates.items() if v}

    todo = [a for a in accessions if a not in submission_dates]
    print(f"Accessions in scope: {len(accessions)}")
    print(f"  already have dates: {len(accessions) - len(todo)}")
    print(f"  to fetch:           {len(todo)}")
    if not todo:
        print("\nNothing to fetch - all accessions already have submission dates.")
        report(submission_dates, accessions, out_path, write=False)
        return

    accessions = todo
    batch_size = 200

    for i in range(0, len(accessions), batch_size):
        batch = accessions[i:i + batch_size]
        print(f"  {i}-{i+len(batch)} / {len(accessions)}")
        try:
            handle = Entrez.efetch(db="nucleotide", id=batch, rettype="gb", retmode="text")
            for record in SeqIO.parse(handle, "genbank"):
                submission_dates[record.id] = extract_submission_date(record)
            handle.close()
        except Exception as e:
            print(f"    [WARN] batch failed: {e}")
        time.sleep(0.4)

    report(submission_dates, accessions, out_path, write=True)


def report(submission_dates, requested, out_path, write=True):
    if write:
        with open(out_path, "w") as f:
            json.dump(submission_dates, f, indent=2)

    found = sum(1 for v in submission_dates.values() if v)
    total = len(submission_dates) or 1
    print(f"\n{'Saved' if write else 'Have'} {len(submission_dates)} entries "
          f"-> {out_path}")
    print(f"  With submission date: {found} ({100*found/total:.1f}%)")
    print(f"  Missing:              {len(submission_dates) - found}")


if __name__ == "__main__":
    main()