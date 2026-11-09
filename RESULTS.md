# EpiSurge — Results Log

Verified experimental results. Every number here came from a script run that
was checked. Planned or expected results do not belong in this file.

**Current scope:** 6 seasons (2016-17 … 2023-24, excluding 2020-21, 2021-22
and 2015-16), 10 HHS regions, 1,980 core-season rows, 5 rolling-origin folds.

**Provenance rule.** Every figure in this file must be traceable to a
committed artifact in `data/reports/`. When a script is re-run, this file is
updated in the SAME commit. This rule exists because it was broken once: the
E1 section was written at commit `27ede32`, E1 was re-run and its CSVs
committed at `1d9a03a`, and the numbers here were left at their earlier
values — eleven of them wrong, one by 24 percentage points. Corrected
2026-11-07 against `coverage_baseline.json`, `coverage_by_peak_side.csv`,
`coverage_by_week.csv` and `delay_summary.csv`. The claims survived; only the
numbers were wrong.

---

## E1 — Delay and coverage timing
**Run:** 2026-10-30 · `src/analysis/delay_coverage_analysis.py`
**Source artifacts:** `data/reports/coverage_*.csv`, `delay_summary.csv`,
`coverage_baseline.json`

### Core-season genomic coverage

| variant | coverage |
|---|---|
| oracle (collection-dated) | **91.5 %** |
| as-of, 8–16 wk window | **64.4 %** |
| as-of, 4–12 wk window | **63.1 %** |

Baseline recorded in `data/reports/coverage_baseline.json` (91.46 / 64.44 /
63.13). E1 compares against it and fails loudly on drift.

### The headline result

Publication delay does not attenuate genomic coverage uniformly. It removes
coverage **specifically before the epidemic peak**, and none after:

| | pre-peak | peak or after |
|---|---|---|
| oracle | **97.1 %** | 86.6 % |
| as-of 8–16 wk | **34.8 %** | 89.6 % |
| as-of 4–12 wk | 37.4 % | 85.0 % |

(Row-weighted; season means are 97.5 / 29.9 / 32.2 and 85.7 / 92.5 / 89.0.)

The gap before the peak is **62.3 percentage points**. After the peak it is
**−3.0 points** — as-of coverage is marginally *higher* than oracle there, so
the loss is entirely pre-peak.

Two splits are used and must not be conflated. The table above is
**peak_side**, relative to each season's own peak. The script's own checks
report the **phase** split, calendar weeks 40–52 against weeks 1–20: oracle
early-minus-late **+8.6 pp**, as-of late-minus-early **+79.0 pp**.

Every season has **8–11 consecutive onset weeks with zero usable genomic
data**, beginning at MMWR week 40 in every case:

| season | consecutive zero weeks | MMWR range |
|---|---|---|
| 2016-17 | 11 | 40–50 |
| 2017-18 | 8 | 40–47 |
| 2018-19 | 9 | 40–48 |
| 2019-20 | 10 | 40–49 |
| 2022-23 | 9 | 40–48 |
| 2023-24 | 9 | 40–48 |

> **Framing note.** Under the earlier censored fetch window this read as a
> *crossover* — oracle falling late while as-of rose. Correcting the window
> showed most of that oracle decline was an artifact of truncated late-season
> collections. Oracle early-minus-late is only +10.5 pp. The claim is
> one-sided loss, not relocation, and is simpler for it.

### Collection-to-submission delay

Overall median **86 days (12.3 weeks)**, n = 32,141.

| season | n | median | p90 | % over 12 wk |
|---|---|---|---|---|
| 2016-17 | 3,096 | 63 | 238 | 38.6 % |
| 2017-18 | 4,057 | 89 | 256 | 52.2 % |
| 2018-19 | 5,434 | 82 | 142 | 46.3 % |
| 2019-20 | 2,321 | 71 | 111 | 23.3 % |
| 2022-23 | 5,183 | 100 | 226 | 69.2 % |
| 2023-24 | 12,050 | 94 | 273 | 55.7 % |

Range of season medians is 63–100 days. Quote as a range, not a point.

---

## E2 — Baselines B0–B2
**Run:** 2026-10-30 · `src/analysis/baselines.py` · 5 rolling-origin folds
**Metrics:** MAE and RMSE only. WIS enters at Phase 5 with conformal intervals.
**Source artifacts:** `data/reports/baseline_results.csv`,
`baseline_summary.csv`

Verified against `baseline_summary.csv` on 2026-11-07: every figure in this
section reproduces to within rounding.

### Claims that survive every fold

**1. Persistence is unbeatable at h = 1.** B1 loses in all five folds
(−11.6 % to −172.6 %).

**2. Climatology beats persistence at h = 6 and h = 8 in all five folds.**

| fold | test season | h6 | h8 |
|---|---|---|---|
| 1 | 2017-18 | +54.6 | +61.5 |
| 2 | 2018-19 | +40.0 | +45.5 |
| 3 | 2019-20 | +46.0 | +52.7 |
| 4 | 2022-23 | +3.5 | +19.4 |
| 5 | 2023-24 | +48.5 | +55.9 |

Sign-consistent at h1 (always negative), h6 and h8 (always positive).
**Inconsistent at h2, h3, h4** — no claim available there.

**3. Gradient boosting beats persistence at every horizon in four of five
folds**, failing only on fold 4.

| fold | h1 | h2 | h3 | h4 | h6 | h8 |
|---|---|---|---|---|---|---|
| 1 | 17.5 | 37.1 | 47.0 | 53.5 | 60.1 | 63.5 |
| 2 | 14.7 | 20.1 | 26.8 | 32.4 | 41.6 | 40.3 |
| 3 | 27.6 | 35.6 | 41.1 | 47.8 | 54.6 | 58.2 |
| 4 | **−29.1** | **−38.1** | **−42.5** | **−35.9** | **−35.8** | **−20.9** |
| 5 | 18.3 | 22.2 | 30.1 | 34.9 | 43.9 | 49.9 |

### Fold 4 is the COVID discontinuity, not noise

Fold 4 trains on 2016-17 … 2019-20 (all pre-COVID) and tests on 2022-23, the
first post-COVID season. Peak timing explains it: pre-COVID seasons peak at
MMWR weeks 05–07, 2022-23 peaked at **week 47** — roughly ten weeks earlier
than anything in training. Every model leaning on seasonal shape fails there.

Reportable as a finding. Pre-COVID folds and the gap-crossing fold must be
reported separately rather than pooled.

### Pooled table (for completeness; per-fold is the reportable view)

| h | B0 | B1 | B2 | B1 skill | B2 skill |
|---|---|---|---|---|---|
| 1 | **0.768** | 1.179 | 0.688 | −53.5 % | +10.4 % |
| 2 | 1.064 | 1.201 | **0.893** | −12.9 % | +16.1 % |
| 3 | 1.335 | 1.221 | **1.060** | +8.5 % | +20.6 % |
| 4 | 1.593 | 1.238 | **1.171** | +22.3 % | +26.5 % |
| 6 | 2.051 | **1.260** | 1.369 | +38.6 % | +33.3 % |
| 8 | 2.408 | **1.256** | 1.443 | +47.8 % | +40.1 % |

### Notes

- Evaluation is paired: identical row sets per (fold, horizon, stratum) across
  all three models, verified by the script's own check.
- B2 seed sd is 0.0 because `HistGradientBoostingRegressor` is deterministic
  given fixed data. Not evidence of stability; the 5-seed protocol bites on
  the GNN.
- B1 rests on 1–5 prior seasons depending on fold. Fold 1's climatology is a
  single season.
- Baseline MAE on genomically-covered rows is close to all-rows MAE (2.267 vs
  2.408 at h=8), so the covered subset is no longer a markedly easier problem
  — it was on three seasons. Pairing is still enforced.

---

## Derived genomic features GF2–GF5
**Built:** 2026-11-07 · `src/features/build_genomic_features.py`
**Source artifact:** `data/features/genomic_derived.csv` (3,120 × 23)

Not an experiment — a feature-availability measurement, recorded here because
it bounds which E3 arms are answerable. Coverage on the 1,980 core-season
rows, as-of 8–16 wk variant:

| family | features | core rows defined |
|---|---|---|
| GF2 change | `js_div_4w`, `d_prop_h3n2_4w` | 1,036 (52.3 %) |
| GF3 effort | `seq_per_100k`, `n_total_log_ratio` | 1,980 (100 %) |
| GF4 spatial | `js_vs_national`, `js_vs_others` | 1,276 (64.4 %) |
| GF5 novelty | `frac_offdominant` | 1,216 (61.4 %) |

GF2 requires both endpoints of the 4-week comparison populated, so arms
carrying it (A3, A4, A6) see more missingness than arms carrying GF1 alone.
`HistGradientBoostingRegressor` handles NaN natively, so no row is dropped and
pairing holds — but the asymmetry belongs in the write-up.

**One defect found and fixed during the build.** `n_total_rel` as an expanding
mean within the region-season was a seasonal clock in disguise: season windows
open at MMWR week 27 (July), so near-zero summer weeks held the denominator
down and the ratio climbed through the autumn — core-season median 2.2 with
37 % of rows above 3. A genomic feature carrying season position would have
inflated any apparent genomic contribution, and the model is already given
season position honestly as `season_week_idx`. Replaced with
`n_total_log_ratio`, a log2 ratio against a trailing 8-week mean: core median
+0.02, sd 1.32, defined on all 1,980 rows.

GF2 and GF5 were hand-verified week by week against the raw subtype counts
for region 4, season 2017-18 before being accepted.

---

## Closed decisions

**2015-16 excluded (2026-10-30).** Median submission delay 298 days against
63–100 for every other season, p10 255 days, 99.7 % of sequences over 12 weeks
late, and 0.0 % core-season as-of coverage across all 33 weeks. The profile is
a retrospective batch deposit rather than surveillance submission. Only 773 of
3,570 QC-passed records (21.7 %) also carried a mappable state. Raw data
retained; recorded in `config.EXCLUDED_SEASONS` as a stated criterion.

Effect: 7 seasons → 6, 6 folds → 5. Oracle coverage 85.0 → 91.5 %, as-of
55.2 → 64.4 %, delay median 91 → 86 days with the outlier removed. The h=6/h=8
climatology result survived the lost fold.

---

## Outstanding before Phase 3 -> Now cmpleted bro

- Update the Master Plan and project documents: delay 76 → 86 days, coverage
  figures, row counts, season count, fold count. The specific §-by-§ edits are
  listed in `PHASE3_PLAN.md` §2.
- Regenerate the seven existing figures through `src/analysis/plotstyle.py`
  so typography and DPI are consistent across the report.
- E3 (genomic representation ablation) is the next experiment —
  `PHASE3_PLAN.md` §4. It is the gate: steps 5–10 of the Phase 3 plan are
  conditioned on its outcome.

  ---

## E3 — Genomic ablation on incidence forecasting
**Run:** 2026-11-09 · `src/analysis/genomic_ablation.py` · 8 arms × 5 folds × 6 horizons
**Source artifacts:** `data/reports/ablation_{results,contrasts,delay_cost}.csv`

**Result: null.** No arm produces a sign-consistent improvement at any
horizon. The primary endpoint — the delay cost A6−A4 — FLIPs at all six.
Where genomics is sign-consistent it is consistently *negative*: A4−A0 is
−8.2 % at h=4 and −7.3 % at h=6; A6−A0 is −5.5 % and −5.7 %.

**The null is controlled, not a bug.**

| control | h4 | h6 |
|---|---|---|
| positive (target + noise) | **+50.2 %** | **+52.9 %** |
| pure noise, 7 columns | FLIP | −2.5 % |
| missingness flag alone | FLIP | FLIP |
| A4 genomic, 7 columns | −8.2 % | −7.3 % |

The positive control is +37.6 % to +56.0 %, every fold, every horizon. The
missingness flag does nothing, so the harm is not the NaN pattern.

**Detectable effect size** (median bootstrap half-width / A0 MAE): 5.5 % at
h=1 rising to 10.8 % at h=6. A delay cost above that band would have been
resolved. 8 of 30 fold × horizon CIs exclude zero.

**Conclusion.** At subtype resolution, composition carries no recoverable
signal for incidence forecasting — oracle or as-of. This bounds the claim:
it is about H3N2-vs-H1N1 composition, not about genomics in general. Clade
or antigenic features are untestable here because only metadata was stored.

**Note on the COVID fold.** Fold 4 reverses sign (A4−A0 is +7.1 to +12.6 %).
One fold, opposite regime, no claim either way — recorded so it is not
mistaken for a finding later.

---

## E3b — Subtype-dominance forecasting and the cost of publication delay
**Run:** 2026-11-09 · `src/analysis/dominance_forecast.py`
**Source artifacts:** `data/reports/dominance_{results,delay_cost,decomposition}.csv`
**Target:** `oracle_prop_h3n2(t+h)` — the eventual H3N2 share, collection-dated

**The delay cost, sign-consistent at every horizon in every fold:**

| contrast | h1 | h2 | h3 | h4 | h6 | h8 |
|---|---|---|---|---|---|---|
| **national, oracle vs as-of (PRIMARY)** | **+43.5** | **+45.4** | **+47.2** | **+48.8** | **+49.7** | **+49.8** |
| regional, oracle vs as-of | +35.7 | +38.3 | +38.9 | +39.7 | +40.9 | +40.4 |
| regional vs national, deployable | −8.4 | −7.2 | −7.0 | −6.9 | −5.5 | −4.4 |
| regional vs national, oracle | −23.1 | −19.7 | −22.7 | −24.1 | −22.5 | −22.3 |
| deployable genomics vs trivial rule | +29.8 | +27.9 | +25.3 | +23.2 | +17.8 | FLIP |
| oracle genomics vs trivial rule | +61.6 | +62.2 | +62.1 | +62.0 | +59.9 | +57.3 |

All 30 fold × horizon bootstrap CIs on the primary endpoint exclude zero,
including the COVID fold (+44.2 % to +49.4 %).

| predictor | h1 accuracy | h1 MAE | h8 accuracy |
|---|---|---|---|
| national, collection-dated | **0.734** | **0.180** | 0.603 |
| national, published by origin | 0.499 | 0.346 | 0.410 |
| predict training seasons' majority | 0.513 | 0.490 | 0.508 |

**Baseline discipline.** The trivial rule is train-majority at 0.51, not
"always predict this season's majority", which scores 0.661–0.666 and would
have made the oracle result look like an artifact. Dominance flips hard
between seasons — H3N2 led 94.5 % of region-weeks in 2016-17 and 23.2 % in
2019-20 — so no fixed rule survives.

**Staleness, not absence.** On rows where both variants carry real data the
accuracy gap is 21.3–24.8 pp of the full 21.4–26.3 pp: roughly 85–95 % of
the damage is that the available sequences are too old.

**Independent corroboration of E1.** The delay cost is 49.9–55.5 % before
the peak against 25.7–36.2 % after it. E1 measured the mechanism; E3b
measures the consequence on a different target. Neither was fitted to the
other.

**No model helps.** Gradient boosting does not beat carry-forward (FLIP or
negative at every horizon). Dominance forecasting is an information-
availability problem, not a modelling problem.