#!/usr/bin/env python3
"""
EpiSurge - Phase 1: NCBI Influenza A HA Sequence Fetcher
Fetches real GenBank records (not FASTA headers) to get verified
geo_loc_name, collection_date, and create_date per sequence.

Two known field-name bugs already found in prior work, fixed here:
  - Use [PDAT], NOT [Collection Date]
  - Use geo_loc_name, NOT country
"""

import time
import json
from pathlib import Path
from datetime import datetime
from Bio import Entrez, SeqIO

Entrez.email = "adamyasharma7476@gmail.com"  # required by NCBI

OUT_DIR = Path("data/raw/ncbi")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEASONS = {
    "2019-20": ("2019/10/01", "2020/05/31"),
    "2022-23": ("2022/10/01", "2023/05/31"),
    "2023-24": ("2023/10/01", "2024/05/31"),
}

QUERY_TEMPLATE = (
    '"Influenza A virus"[Organism] AND HA[Gene] AND USA[Country] '
    'AND ("{start}"[PDAT] : "{end}"[PDAT])'
)

def search_ids(start, end):
    query = QUERY_TEMPLATE.format(start=start, end=end)
    handle = Entrez.esearch(db="nucleotide", term=query, retmax=50000)
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]

def fetch_records(id_list, season_label, batch_size=200):
    all_records = []
    for i in range(0, len(id_list), batch_size):
        batch = id_list[i:i + batch_size]
        print(f"  [{season_label}] fetching {i}-{i+len(batch)} / {len(id_list)}")
        handle = Entrez.efetch(db="nucleotide", id=batch, rettype="gb", retmode="text")
        for record in SeqIO.parse(handle, "genbank"):
            source_feature = next((f for f in record.features if f.type == "source"), None)
            geo_loc = source_feature.qualifiers.get("geo_loc_name", [""])[0] if source_feature else ""
            collection_date = source_feature.qualifiers.get("collection_date", [""])[0] if source_feature else ""

            all_records.append({
                "accession": record.id,
                "geo_loc_name": geo_loc,
                "collection_date": collection_date,
                "create_date": record.annotations.get("date", ""),
                "organism": record.annotations.get("organism", ""),
                "sequence_length": len(record.seq),
                "description": record.description,
            })
        handle.close()
        time.sleep(0.4)  # NCBI rate limit courtesy delay
    return all_records

def main():
    for season_label, (start, end) in SEASONS.items():
        print(f"Searching season {season_label} ({start} to {end})...")
        ids = search_ids(start, end)
        print(f"  Found {len(ids)} sequence IDs")

        records = fetch_records(ids, season_label)

        out_path = OUT_DIR / f"ncbi_{season_label}.json"
        with open(out_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"  Saved {len(records)} records -> {out_path}\n")

if __name__ == "__main__":
    main()