#!/usr/bin/env python3
"""
EpiSurge - feasibility check: can the study be extended to earlier seasons?

E2 showed that two rolling-origin folds produce opposite conclusions
(climatological skill at h=8 is -8.9% in fold 1 and +56.7% in fold 2),
because the three seasons peak at MMWR weeks 06, 47 and 52 and a
climatology built from one season lands on the wrong week in the next.
Between-season peak-timing variability dominates every model difference,
and with three seasons it is estimated from a sample of two folds.

Extending to 2015-16 .. 2018-19 would give seven seasons and six folds.
Surveillance, climate, population and boundary data all cover that period.
The open question is genomic volume: NCBI sequencing of US influenza A HA
was lower before 2019, and if the earlier seasons are too sparse the
extension buys folds for the baselines but not for the genomic comparison,
which is the one that matters.

This script COUNTS ONLY. It downloads no records and writes no data. It
issues esearch requests with retmax=0, which return a count and nothing
else, so it is fast and safe to re-run.

Decision guide (applied to the per-season counts it prints):
    >= 1500 seqs/season   extension is worthwhile for the genomic arm
    500 - 1500            marginal; extension helps baselines, genomic
                          coverage per region-week will be thin
    < 500                 do not extend for genomic purposes; keep three
                          seasons and change the claim instead

Run (from the repository root, inside the venv):
    python src/analysis/check_season_extension.py --email you@example.com
"""

import argparse
import sys
import time

from Bio import Entrez

# Same query form as src/ingestion/fetch_ncbi.py. [PDAT] is publication
# date - the only date field NCBI indexes reliably (see DECISIONS.md:
# [Collection Date] is not a valid field and returns zero silently).
QUERY = ('"Influenza A virus"[Organism] AND HA[Gene] AND USA[Country] '
         'AND ("{start}"[PDAT] : "{end}"[PDAT])')

# Season s runs collection weeks Jul(s) -> Jun(s+1). Publication lags
# collection by roughly 2-4 months, so the publication window is shifted
# and deliberately generous. Windows therefore OVERLAP between adjacent
# seasons and the counts are NOT additive - they indicate volume, not a
# partition.
SEASONS = [
    ("2015-16", "2015/07/01", "2016/12/31"),
    ("2016-17", "2016/07/01", "2017/12/31"),
    ("2017-18", "2017/07/01", "2018/12/31"),
    ("2018-19", "2018/07/01", "2019/12/31"),
    ("2019-20", "2019/07/01", "2020/12/31"),   # already in the study
    ("2022-23", "2022/07/01", "2023/12/31"),   # already in the study
    ("2023-24", "2023/07/01", "2024/12/31"),   # already in the study
]

IN_STUDY = {"2019-20", "2022-23", "2023-24"}

# Clean calendar-year counts, for the volume trend without window overlap.
YEARS = list(range(2014, 2025))


def count(term):
    handle = Entrez.esearch(db="nucleotide", term=term, retmax=0)
    rec = Entrez.read(handle)
    handle.close()
    return int(rec["Count"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True, help="contact email for NCBI Entrez")
    ap.add_argument("--api-key", default=None,
                    help="optional NCBI API key (raises the rate limit)")
    args = ap.parse_args()

    Entrez.email = args.email
    if args.api_key:
        Entrez.api_key = args.api_key
    pause = 0.15 if args.api_key else 0.40

    print("EpiSurge - season extension feasibility check")
    print("=" * 68)
    print("Counts only. No records are downloaded.\n")

    print("-" * 68)
    print("Influenza A HA, USA, by publication year")
    print("-" * 68)
    year_counts = {}
    for y in YEARS:
        try:
            n = count(QUERY.format(start=f"{y}/01/01", end=f"{y}/12/31"))
        except Exception as e:
            print(f"  {y}: ERROR {e}")
            time.sleep(2)
            continue
        year_counts[y] = n
        bar = "#" * min(60, n // 400)
        flag = "  <- COVID era" if y in (2020, 2021) else ""
        print(f"  {y}: {n:>7,}  {bar}{flag}")
        time.sleep(pause)

    print("\n" + "-" * 68)
    print("Per-season publication windows (overlapping, NOT additive)")
    print("-" * 68)
    season_counts = {}
    for label, start, end in SEASONS:
        try:
            n = count(QUERY.format(start=start, end=end))
        except Exception as e:
            print(f"  {label}: ERROR {e}")
            time.sleep(2)
            continue
        season_counts[label] = n
        tag = "already in study" if label in IN_STUDY else "CANDIDATE"
        print(f"  {label}  {start} .. {end}  {n:>7,}   {tag}")
        time.sleep(pause)

    # ── verdict ──────────────────────────────────────────────────────
    print("\n" + "-" * 68)
    print("Assessment")
    print("-" * 68)

    candidates = [l for l, _, _ in SEASONS if l not in IN_STUDY]
    have = [season_counts[l] for l in candidates if l in season_counts]
    if not have:
        sys.exit("No candidate-season counts retrieved; cannot assess.")

    lo, hi = min(have), max(have)
    print(f"  candidate seasons: {', '.join(candidates)}")
    print(f"  publication-window counts range {lo:,} to {hi:,}")

    ref = [season_counts[l] for l in IN_STUDY if l in season_counts]
    if ref:
        print(f"  for reference, seasons already in the study range "
              f"{min(ref):,} to {max(ref):,}")
        print(f"  (the study currently holds 29,115 raw records across those three)")

    print()
    if lo >= 1500:
        print("  VERDICT: extend. Every candidate season clears the 1,500 threshold,")
        print("  so the genomic arm gains real folds, not just the baselines.")
    elif lo >= 500:
        print("  VERDICT: marginal. Extension gives six folds for the baselines, but")
        print("  per-region-week genomic coverage in the thin seasons will be low.")
        print("  Consider extending surveillance/climate across all seven seasons and")
        print("  reporting the genomic comparison on the subset that supports it,")
        print("  stating the differing denominators explicitly.")
    else:
        print("  VERDICT: do not extend for genomic purposes. Keep three seasons and")
        print("  change the CLAIM instead: report per-fold results always, attach")
        print("  block-bootstrap intervals, and state that a genomic effect cannot be")
        print("  separated from between-season variability at this sample size.")
        print("  Under Master Plan D1 and D4 that is a legitimate reported outcome.")

    print("\n  Note: these are publication-date windows and overlap between adjacent")
    print("  seasons, so they indicate volume only. True per-season collection counts")
    print("  come out of qc_filter.py and map_region_epiweek.py after a real fetch.")
    print("\n  Cost if you extend: four more seasons through fetch_ncbi.py and")
    print("  fetch_submission_dates.py (hours, unattended), eight more ERA5 half-year")
    print("  requests (~40 min each, queue overnight), and FluView/vintages/Census")
    print("  re-run over the wider window (minutes).")


if __name__ == "__main__":
    main()