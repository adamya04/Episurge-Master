# EpiSurge — Results Log

Verified experimental results. Every number here came from a script run that
was checked. Planned or expected results do not belong in this file.

**Current scope:** 6 seasons (2016-17 … 2023-24, excluding 2020-21, 2021-22
and 2015-16), 10 HHS regions, 1,980 core-season rows, 5 rolling-origin folds.

---

## E1 — Delay and coverage timing
**Run:** 2026-10-30 · `src/analysis/delay_coverage_analysis.py`

### Core-season genomic coverage

| variant | coverage |
|---|---|
| oracle (collection-dated) | **91.8 %** |
| as-of, 8–16 wk window | **64.4 %** |
| as-of, 4–12 wk window | **63.1 %** |

Baseline recorded in `data/reports/coverage_baseline.json`.

### The headline result

Publication delay does not attenuate genomic coverage uniformly. It removes
coverage **specifically before the epidemic peak**, and almost none after:

| | pre-peak | peak or after |
|---|---|---|
| oracle | **97.3 %** | 87.1 % |
| as-of 8–16 wk | **34.8 %** | 89.6 % |
| as-of 4–12 wk | 37.4 % | 85.0 % |

(Row-weighted; season means are 97.6 / 29.9 / 32.2 and 86.1 / 92.5 / 89.0.)

The gap before the peak is **62.5 percentage points**. After the peak it is
essentially zero. As-of late-minus-early is **+79.0 pp**.

Every season has **8–11 consecutive onset weeks with zero usable genomic
data** (MMWR weeks 40–47 through 40–50).

> **Framing note.** Under the earlier censored fetch window this read as a
> *crossover* — oracle falling late while as-of rose. Correcting the window
> showed most of that oracle decline was an artifact of truncated late-season
> collections. Oracle early-minus-late is now only +8.3 pp. The claim is
> one-sided loss, not relocation, and is simpler for it.
>
> **`fig_coverage_crossover.png` and `fig_coverage_vs_peak.png` are still
> titled for the crossover reading and must be retitled before use.**

### Collection-to-submission delay

Overall median **90 days (12.9 weeks)**, n = 34,117.

| season | n | median | p90 | % over 12 wk |
|---|---|---|---|---|
| 2016-17 | 3,209 | 65 | 256 | 40.7 % |
| 2017-18 | 4,128 | 90 | 282 | 53.1 % |
| 2018-19 | 5,441 | 82 | 143 | 46.3 % |
| 2019-20 | 2,327 | 71 | 112 | 23.5 % |
| 2022-23 | 6,195 | 110 | 734 | 74.2 % |
| 2023-24 | 12,817 | 95 | 300 | 56.5 % |

Range of season medians is 65–110 days. Quote as a range, not a point.

---

## E2 — Baselines B0–B2
**Run:** 2026-10-30 · `src/analysis/baselines.py` · 5 rolling-origin folds
**Metrics:** MAE and RMSE only. WIS enters at Phase 5 with conformal intervals.

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
| 1 | **0.768** | 1.179 | 0.688 | −53.5 % | +10.5 % |
| 2 | 1.064 | 1.201 | **0.893** | −12.9 % | +16.0 % |
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

## Closed decisions

**2015-16 excluded (2026-10-30).** Median submission delay 298 days against
65–110 for every other season, p10 255 days, 99.7 % of sequences over 12 weeks
late, and 0.0 % core-season as-of coverage across all 33 weeks. The profile is
a retrospective batch deposit rather than surveillance submission. Only 773 of
3,570 QC-passed records (21.7 %) also carried a mappable state. Raw data
retained; recorded in `config.EXCLUDED_SEASONS` as a stated criterion.

Effect: 7 seasons → 6, 6 folds → 5. Oracle coverage 85.0 → 91.8 %, as-of
55.2 → 64.4 %, delay median 91 → 90 days with the outlier removed. The h=6/h=8
climatology result survived the lost fold.

---

## Outstanding before Phase 3

- Retitle the two coverage figures (still say "crossover").
- Update the Master Plan and project documents: delay 76 → 90 days, coverage
  figures, row counts, season count, fold count.
- E3 (genomic representation ablation) is the next experiment.