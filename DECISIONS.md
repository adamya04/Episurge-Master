| 2026-10-01 | Dropped 1,058 records where geo_loc_name="USA" with no state specified (cannot map to HHS region). Kept all records with abbreviation/lowercase/county-prefixed state info via extract_state() fix. Final mapped total: 13,807 / 14,865 QC'd records (92.9%). | Honest data limitation, not a parsing bug — verified via debug_geoloc.py inspection of raw strings. |

| 2026-10-01 | True submission dates (GenBank REFERENCE) show median collection->submission delay of 76 days (~11wk). Realtime retention: k=0:0%, k=4:6.2%, k=8:31.8%. Dropped k=0/k=2 variants as non-viable; added k=12/k=16. Added lagged-composition features as the operationally realistic genomic input. | Real-time genomic composition is effectively unavailable at nowcast horizons — this is the N5 finding, not a pipeline failure. |

| 2026-10-01 | As-of coverage ceiling is ~47% of region-week cells across all windows. Retained L4-12 and L8-16 as ablation variants; dropped L1-8 (12.4% coverage) and L12-24 (dominated). | Structural sparsity in US flu sequencing — validates H3, makes coverage flag/masked loss mandatory rather than optional. |

| 2026-10-01 | Excluded Puerto Rico from HHS Region 2. Including it displaced the Region 2 centroid 2.334 degrees south (into the Atlantic), distorting gravity-model distances. PR was also absent from the genomic state mapping and is reported separately by CDC ILINet. Final mapped population: 334,914,895 across 10 regions. | Geographic consistency across population, genomic, and target layers. |

| 2026-10-01 | Region centroids use state GEOGRAPHIC centroids weighted by population, not true population centroids. Region 8 lands at (41.12, -106.86) despite Colorado holding ~46% of its population, because Montana/Wyoming land area pulls it. | Acceptable simplification for a 10-node graph; stated assumption for the model card. |

| 2026-10-02 | CDS returns ZIP archives (magic bytes PK\x03\x04) when a request has multiple variables, despite the .nc extension and "unarchived" download option. Added unzip_era5.py; extracted files are one-variable-per-file (t2m and d2m separately) in data/raw/era5/extracted/. | Downstream code must read from extracted/, not the raw .nc files. |

| 2026-10-02 | Heavy model training runs on Kaggle/Colab. Training notebooks committed to notebooks/; trained model artifacts pulled back into the repo. Final deliverable includes a production-grade web application (Phase 6). | Local machine insufficient for larger training runs; app is the eventual showcase. |

| 2026-10-02 | Climate: DC (pop. 678,972) absent from Region 3 weighting - too small for any 0.25-degree ERA5 cell centre to fall inside it. Region 3 climate is represented by surrounding MD/VA cells. Alaska and Hawaii excluded by CONUS bounding box as planned. | Resolution artifact; negligible effect since DC is enclosed by Region 3 states. |

| 2026-10-02 | F1: Fetched FluView revision history via the Delphi `issues` parameter - 156 issues, 109,070 rows, 0 empty. Convention fixed: TARGET = final revised wILI (CDC FluSight scoring convention); INPUTS = wILI as published at the forecast-origin issue. Backfill magnitude measured at lag 1: mean absolute revision 0.091 pct-points, median 0.022, p90 0.265, max 2.541, 19.9% of cells revised by >0.1. | Prevents backfill leakage - using final revised values as model inputs would hand the model information a real forecaster lacked, the same flaw N5 exposes on the genomic side. Median correction is small but the 2.541 max shows individual weeks can be revised substantially. |


| 2026-10-02 | F2: Replaced the hand-rolled MMWR epiweek function with the `epiweeks` library across map_region_epiweek.py, build_oracle_composition.py, build_asof_composition.py and era5_to_region_week.py. The old function used shifted ISO logic that mis-keys years where 1 January falls on a Thursday (assigns ISO week 1 where MMWR assigns week 53 of the prior year). Verified neutral on current data: 13,807 records, 84/84 epiweek overlap, 734 joinable cells, 734 oracle rows, 76-day median submission delay and 1,590 climate region-weeks all unchanged. | Master Plan specified the epiweeks library. Latent bug - no effect on 2019-2024 but would silently mis-key any future rerun, including 2026. |


| 2026-10-02 | F3: As-of composition origin grid now spans all 156 season-window weeks rather than only the 84 weeks in which sequences were collected. Valid cells rose from 371 to 538 (L4-12) and 398 to 702 (L8-16). | The previous grid omitted origin weeks that had no same-week collections but did have published sequences inside their trailing window, understating as-of genomic coverage. |


| 2026-10-02 | F4: Deleted superseded feature files - composition_realtime_k*.csv (k-offset approach) and asof_composition_L1-8, L9-20, L12-24. | The k-offset approach anchored on collection week rather than forecast origin and was replaced by the as-of approach. L1-8 dropped at 12.4% coverage; L9-20 and L12-24 dominated by L8-16 on both coverage and freshness. |


| 2026-10-02 | F6/F7: Deleting build_composition.py removed the only means of regenerating composition_oracle.csv, which was also still built with the pre-F2 epiweek function. Replaced with build_oracle_composition.py using the epiweeks library and without the dead k-offset code. Oracle is explicitly the N5 upper bound, not a usable forecasting feature - it leaks future information by construction. | Reproducibility requirement; oracle is the ceiling against which as-of degradation is measured, so N5 cannot be quantified without it. |


| 2026-10-02 | F8: Delphi reports a first-publication lag of 0 (issue W contains epiweek W), but CDC publishes ILINet on Fridays for the week ending the prior Saturday, which implies lag 1. Could not resolve from the data whether Delphi labels issues by data week or publication week. Decision: use lag >= 1 only - at origin T, use issue T restricted to epiweeks <= T-1. | Correct under either interpretation. Costs one week of recency; eliminates a leakage risk that cannot otherwise be ruled out. Lag >= 0 to be run later as a sensitivity check. |


| 2026-10-02 | F9: As-of coverage percentages computed over the full week 27-26 window include summer weeks with minimal influenza sequencing. Headline coverage is core-season only (weeks 40-20). Confirmed in the master table: core-season coverage 47.4% (L4-12) and 53.7% (L8-16) versus full-window 34.5% and 45.0%. | The full-window percentage understates usable coverage - summer weeks are outside the forecast window and never evaluated. |


| 2026-10-02 | Verified vintage coverage: 156 issues present, 156 origins needed, no missing or extra issues, every origin has usable lag>=1 history. | Feature engineering requires issue T to exist for every origin T; a missing issue would silently drop that origin from the modelling set. |


| 2026-10-02 | Built as-of ILI lag features (data/features/asof_ili.csv): 1,560 rows (156 origins x 10 regions), lags 1-8, zero missing values across all lag columns. Filtering uses Delphi's own `lag` column rather than computing T-k locally. | Epiweek arithmetic across year boundaries is a known off-by-one source (202301 minus 1 is 202252 or 202253 depending on the year); Delphi has already resolved this correctly. |


| 2026-10-02 | Built master region-week table (data/processed/master_region_week.csv): 1,560 rows x 37 columns, 990 core-season rows, all modelling-ready with zero missing target, population, climate or surveillance values. Genomic coverage over core season - oracle 73.9%, as-of L8-16 53.7%, as-of L4-12 47.4%. | Table holds joined raw values only; no derived features, so feature choices can change without re-running the join. The oracle-to-as-of gap is the quantified N5 result. |


| 2026-10-02 | Fixed join bug: composition_oracle.csv carries both `epiweek_int` (key) and `epiweek` (display string), so renaming the former collided with the latter. Loader now drops the string column before renaming. | Duplicate column labels break pandas merges. |


| 2026-10-02 | Process: adopted standing practice of proactively flagging every suspected flaw at the moment of detection, however minor, rather than deferring. Flaw register maintained with severity and impact per item. | Two prior project resets originated in unexamined assumptions; the cost of raising a non-issue is far below the cost of carrying a real one undetected. |

| 2026-10-02 | F10: Investigated an apparent 732 vs 734 oracle cell discrepancy. Not a defect - 732 is the core-season count (weeks 40-20) while 734 is the all-rows count, which matched exactly. Two oracle cells fall in shoulder weeks 21-39, inside the season window but outside the evaluation window. Verified: zero oracle cells missing from the master table, zero cells with n_total <= 0. | Raised from comparing a core-season figure against an all-rows baseline. No action required. |

| 2026-10-02 | F11: Replaced the hardcoded 5-sequence validity threshold with empirical-Bayes shrinkage toward the pooled national composition (k=5), retaining sequence count as a feature. A proportion from n sequences has SE sqrt(p(1-p)/n) - at n=5, p=0.5 that is +/-22pp, so any hard cutoff is arbitrary and discarding n<5 both loses real signal and biases the retained sample toward high-sequencing regions, which correlate with population and surveillance capacity. Verified monotonic: mean shrinkage 0.214 at n<=2 falling to 0.002 at n>100. Core-season coverage rose 47.4%->55.8% (L4-12) and 53.7%->60.3% (L8-16). | Sharpens H3 from "the model knows which cells have genomic data" to "the model knows how much genomic evidence each cell carries". No cell is dropped. |


| 2026-10-02 | Master table rebuilt with shrunk composition features: 1,560 rows x 43 columns, 990 core-season rows, all modelling-ready. N5 gap quantified - oracle 73.9% vs as-of L8-16 60.3% core-season genomic coverage. | Oracle is the retrospective-study assumption; as-of is what a real forecaster has. The gap is the contribution. |