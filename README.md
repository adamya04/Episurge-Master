# EpiSurge

**Does pathogen genomic composition carry forecasting signal for regional
influenza incidence beyond what case history already provides — and at which
forecast horizons?**

Team 18 · B.Tech Artificial Intelligence and Data Science · Amrita Vishwa
Vidyapeetham, Amritapuri · Course 23AID498

This is a **measurement study**, not a model-building exercise. The answer may
be no, and that remains a result. The architecture is deliberately
conservative; the contribution is the experiment.

---

## Status

Phase 1 (data foundation) and Phase 2 (E1, E2) are complete and verified.
No model has been trained yet.

| | |
|---|---|
| Seasons | 6 — 2016-17 … 2023-24, excluding 2020-21, 2021-22, 2015-16 |
| Regions | 10 US HHS regions |
| Target | Weekly weighted ILI % (CDC FluView) |
| Horizons | 1, 2, 3, 4, 6, 8 weeks |
| Master table | 3,120 rows × 43 columns; 1,980 core-season |
| Evaluation | Rolling origin, 5 folds, expanding window |

Results: `RESULTS.md` · Decisions and rationale: `DECISIONS.md` · Roadmap:
`FUTURE_WORK.md`

---

## Setup

Python 3.11+ on Windows (the pipeline is developed and run on Windows
Command Prompt; paths in script output use backslashes).

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Credentials

Four sources need credentials. None are stored in the repository.

| source | requirement | how |
|---|---|---|
| NCBI Entrez | contact email; API key optional but ~3× faster | pass `--email`, optionally `--api-key` |
| Delphi Epidata | API key for the vintage fetch (anonymous hits HTTP 429) | `set DELPHI_API_KEY=<key>` |
| Copernicus CDS | Personal Access Token | `%USERPROFILE%\.cdsapirc` with `url:` and `key:` lines, and the ERA5 dataset licence accepted on its Download tab |
| US Census | API key | `set CENSUS_API_KEY=<key>` |

---

## Running the pipeline

`src/tools/run_pipeline.py` runs every stage in order with dependency checks.
A stage whose inputs are missing is skipped with a reason rather than left to
read a stale file and report plausible numbers from the wrong data.

```
python src/tools/run_pipeline.py --list              # show stages and readiness
python src/tools/run_pipeline.py                     # everything except ingestion
python src/tools/run_pipeline.py --include-ingestion # including the slow fetches
python src/tools/run_pipeline.py --only analysis     # just E1 and E2
python src/tools/run_pipeline.py --from qc_filter    # resume partway
```

Ingestion is excluded by default: it needs credentials, takes hours, and is
resumable on its own.

### Equivalent manual sequence

```
python src/ingestion/fetch_ncbi.py --email you@example.com
python src/ingestion/fetch_fluview.py
python src/ingestion/fetch_fluview_vintages.py
python src/ingestion/fetch_census.py
python src/ingestion/fetch_gadm.py
python src/ingestion/fetch_era5.py

python src/preprocessing/qc_filter.py
python src/preprocessing/map_region_epiweek.py
python src/ingestion/fetch_submission_dates.py
python src/preprocessing/unzip_era5.py
python src/preprocessing/era5_to_region_week.py

python src/features/build_oracle_composition.py
python src/features/build_asof_composition.py
python src/features/build_asof_ili.py

python src/preprocessing/build_master_table.py
python src/preprocessing/validate_all_sources.py

python src/analysis/delay_coverage_analysis.py     # E1
python src/analysis/baselines.py                   # E2
```

**Expected runtimes.** NCBI fetch and submission dates are hours on a first
run (both resumable; submission dates are incremental thereafter). ERA5 queues
on Copernicus servers — roughly 10 minutes per half-year request, one at a
time, so plan for several hours. Everything else is seconds to minutes.

### Verification

```
python src/analysis/verify_fetch_window.py   # is the delay distribution truncated?
python src/analysis/diagnose_fetch.py        # censoring test + state-parsing audit
python src/preprocessing/validate_all_sources.py
```

`validate_all_sources.py` is the gate: it exits non-zero on any hard failure
and must pass before the master table is trusted.

---

## Repository layout

```
src/config.py                  single source of truth for seasons and windows
src/ingestion/                 one fetcher per external source
src/preprocessing/             QC, mapping, climate aggregation, the join, validation
src/features/                  composition and surveillance feature builders
src/analysis/                  experiments (E1, E2) and diagnostics
src/tools/                     pipeline driver and development scripts
data/raw/                      immutable source snapshots (gitignored)
data/processed/                QC'd, mapped and joined data
data/features/                 model-ready feature tables
data/reports/                  result tables and regression baselines
figures/                       generated figures
```

### Configuration

**All season and window configuration lives in `src/config.py`.** Nothing else
defines it. Adding or removing a season is one edit there, followed by a
re-run from `qc_filter.py`.

---

## Data sources

| source | content | access |
|---|---|---|
| NCBI GenBank | Influenza A HA sequences, USA | Entrez E-utilities |
| CDC FluView | Weekly wILI per HHS region, plus revision vintages | Delphi Epidata API |
| ERA5 | 2 m temperature and dewpoint | Copernicus CDS |
| US Census PEP | State populations | Census API |
| GADM v4.1 | State boundary polygons | direct download |

---

## Reproducibility

- **Raw data is gitignored** and not committed — ERA5 alone is ~300 MB.
- **The NCBI snapshot is not re-fetchable.** GenBank gains records
  retroactively, so a refetch returns a different dataset. The snapshot *is*
  the dataset. Back it up independently of this repository.
- Regression baselines live in `data/reports/coverage_baseline.json`. E1
  compares against them and fails loudly on drift; delete the file to
  re-accept new numbers after an intentional data change.
- `data/reports/baseline_results.csv` is the full audit trail behind every E2
  number — one row per model × fold × horizon × seed × stratum.

---

## Standing methodological rules

Established by experiment and recorded in `DECISIONS.md`. Do not reverse
without evidence.

1. **Human hosts only.** Non-human influenza is excluded by a structural rule
   on strain nomenclature, not a blocklist.
2. **As-of, not final.** Inputs use values as published at the forecast
   origin; only the target uses final revised values.
3. **Paired evaluation** on identical row sets, verified in code.
4. **Per-fold reporting.** Pooled-across-fold figures are not reportable; a
   horizon whose sign flips across folds supports no claim.
5. **Phase stratification** on every genomic result, because as-of coverage is
   concentrated after the epidemic peak.