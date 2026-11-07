# EpiSurge — Future Work

Two separate things live in this file, and they are not interchangeable.

**Part 1 — Pending work** is inside the agreed scope and planned. It is what
remains to finish the project.

**Part 2 — Future research directions** is outside current scope. Per the
Master Plan §2 scope-change rule, nothing moves from Part 2 into the build
unless an experiment proves a current assumption wrong. Ideas are recorded
here precisely so they do not leak into the build on the grounds of sounding
valuable.

Nothing already implemented and verified appears in this file. For what is
done, see `RESULTS.md`; for why, see `DECISIONS.md`.

---

# Part 1 — Pending work (in scope)

## 1.1 Immediate — before E3

| item | why it blocks |
|---|---|
| Execute the F4 deletions | Four superseded feature files with the obsolete pre-F11 `is_valid` schema are still in `data/features/`. Anything globbing `asof_composition_*.csv` would pick up dropped variants. |
| Record raw-data provenance | `data/raw/` is gitignored, and the NCBI snapshot is **not re-fetchable** — GenBank gains records retroactively, so a refetch next year returns a different dataset. Without checksums and recorded request parameters, nobody can verify they reconstructed the same inputs. |
| Update the Master Plan and project documents | They still quote 3 seasons, 990 rows, 76-day delay, and 73.9% oracle coverage. All superseded. |
| Amend Master Plan §7.3 | It lists transfer entropy as mandatory. It is not supportable on this data — see 1.2. |

## 1.2 Phase 2 completion — E3 and E4

**E3 — genomic representation ablation.** The first experiment that tests the
research question. Gradient boosting with and without genomic features,
paired on identical rows, phase-stratified, reported per fold.

Two constraints already known:

- Composition is now **binary** (H3N2 share). The R1 (raw proportions) and R3
  (CLR) variants are monotone transforms of each other on two categories and
  will give near-identical results. Only R2 (JS divergence, i.e. change)
  is genuinely distinct. Running a four-way ablation where two arms are the
  same thing would waste effort and look careless in the paper.
- **Fold 4 will dominate.** It broke every baseline. Decide in advance whether
  the headline is "all folds" or "pre-COVID folds, with the COVID
  discontinuity reported separately".

**E4 — model-free evidence.** Panel Granger causality, composition against
wILI. Three requirements established by the feasibility work:

- **Pooled panel only.** Usable segments are 14–24 contiguous points per
  region-season. Per-region Granger is dead beyond lag 1 (at p=2 you fit 5
  parameters on 12 rows).
- **Must run on differences.** wILI is non-stationary within season — median
  ADF p = 0.545, only 7% stationary at 0.05.
- **Phase-stratified**, or the horizon effect is confounded with seasonal
  phase.

**Transfer entropy — demoted, deliberately.** TE needs hundreds of points per
series for a stable estimate and the segments are 14–24. Pooled at lag 1 with
coarse binning it is computable, but bias-dominated and requiring a
permutation null just to interpret. Run it as a secondary check expecting
"inconclusive"; do not build an evidence line on it.

## 1.3 Phases 3–6 (unchanged from the Master Plan)

- **Phase 3** — EpiGraph-GNN: GRU encoder, hand-written tri-relational
  GraphSAGE, learned relation gates, multi-horizon heads, MLflow tracking.
  The 5-seed protocol becomes meaningful here; it is vacuous for the
  deterministic baselines.
- **Phase 4** — full ablation grid including the density-matched random-graph
  control (G5), which is non-negotiable.
- **Phase 5** — conformal prediction and coverage analysis. **N7 needs
  narrowing**: per-tier calibration counts were 129 / 27 / 2 / 2, so a
  Mondrian *correction* is not fittable. The *diagnosis* is — report test
  coverage conditional on tier, showing marginal coverage hides tier-dependent
  miscalibration. Decide this before Phase 5, not during it.
- **Phase 6** — FastAPI service, dashboard, weekly scheduler, model card.

---

# Part 2 — Future research directions (out of scope)

## 2.1 Genomic depth — the strongest upgrade path

**Per-region genetic distance matrices.** The repository stores sequence
*metadata* only: `fetch_ncbi.py` keeps `len(record.seq)` and discards
`record.seq`. No nucleotides exist anywhere in `data/`. Consequently
phylogenetic trees, mutation analysis, sequence similarity and clade
assignment are all **currently impossible**, not merely unprioritised.

This matters because composition is now a single number, which makes the
genomic-similarity graph relation (H2) thin — cosine similarity between two
2-vectors that each sum to 1 reduces to "how close are their H3N2 shares".
Actual sequence divergence between regions' viral populations would be a far
better foundation.

Feasible if scoped properly: subsample ~50 sequences per region-season
(≈3,000 total, not 35,000), align, compute pairwise distance. That is
tractable. Full phylogeny over 35,000 sequences is a different project.

**Revisit only if E3 shows the genomic block carries signal.** If it does not,
a better graph relation would not have rescued it.

**Clade and subclade resolution.** Would restore a genuinely multi-dimensional
composition instead of a binary one. Requires sequence data and a clade
assignment tool. Same precondition.

## 2.2 Additional experimental studies

- **Lag ≥ 0 sensitivity check** — F8 chose lag ≥ 1 conservatively because the
  Delphi issue convention could not be resolved from the data. Running lag ≥ 0
  would quantify what that caution costs. Low effort, explicitly deferred at
  the time.
- **2020-21 and 2021-22 as a held-out regime test** — excluded for COVID
  suppression, but fold 4 showed the discontinuity is itself informative.
  Testing a model trained on normal seasons against suppressed ones would
  characterise failure under regime change.
- **NASA POWER climate cross-check** — Master Plan D8's documented fallback,
  never exercised. Would test sensitivity to the climate source. Note the
  constraint: it must be applied to **all** seasons or it introduces a
  source change at a season boundary, which is exactly the confound the study
  is built to avoid.
- **Alternative graph relations** — commuting flows, airline passenger volume.
  Both would be better-grounded than a gravity model but need external data.

## 2.3 Statistical and robustness work

- **Block-bootstrap confidence intervals** on fold-level results. With 5 folds,
  sign consistency is currently the only inference tool available; bootstrap
  intervals would quantify uncertainty rather than just direction.
- **Regional heterogeneity.** All results are pooled across 10 regions.
  Whether genomic value differs by region — plausible, since sequencing
  capacity varies — is untested.
- **Sequencing-capacity confound.** Coverage correlates with population and
  surveillance capacity. A region-level analysis controlling for capacity
  would separate "genomics helps" from "well-resourced regions are easier to
  forecast".

## 2.4 Model and graph extensions

Deliberately excluded from the current scope and listed here only to record
that they were considered: graph transformers, neural ODEs, hypergraphs,
diffusion models, learned graph structure. All would overfit ~1,980 rows and
none addresses the research question. Attention-based relation weighting is
the one plausible extension, and only if the learned relation gates turn out
to be informative in Phase 4.

## 2.5 Explainability

- SHAP at feature-block level is already planned for Phase 5.
- Relation-gate values are a planned Phase 4 readout.
- Beyond those: per-region attribution, and counterfactual analysis of what a
  forecast would have looked like with timely genomic data — which connects
  directly to the N5 finding and would be a strong closing figure.

## 2.6 Deployment and scale

- Operational weekly runs against live Delphi issues.
- Extension to state level — 50 nodes instead of 10 would materially improve
  the graph-learning case, but surveillance data at state level is sparser and
  the genomic coverage problem would worsen.
- Other respiratory pathogens. **Not a second disease in this study** — that
  framing was rejected at the mid-term review as confounding disease with
  country, surveillance system and data density. Only ever as a separate
  study.

---

## Explicitly rejected — do not reintroduce

Recorded so these are not proposed again. Each was evaluated and rejected on
evidence; see `DECISIONS.md` and the Master Plan §2.

| rejected | reason |
|---|---|
| Dengue or any second disease | Confounds disease with country, surveillance system and data density — the mid-term review objection |
| Genetic / evolutionary algorithms | Audit of eight optimisation problems found no fit; dataset far too small for heavy search |
| KAN as a core architecture | Not novel; documented weakness on noisy functions. Retained only as a single ablation row |
| Offline RL on OxCGRT | Circular environment; COVID→influenza domain mismatch |
| Five-agent MARIO orchestration | No implemented multi-agent epidemic decision-support system exists in the literature; adds no evidence for the question |
| Genomic encoder trained from scratch | Influ-BERT is trained on ~900,000 sequences; ours would have ~32,000 |
| PyTorch Geometric | At 10 nodes, mean-aggregation message passing is ~20 lines; the dependency is a known Windows failure point |
| Newly invented evaluation metrics | Reviewers discount single-paper metrics |

ChatGPT Suggestions:

A. §1.1 table — two new rows (after the “Amend Master Plan §7.3” row):

| Add a shared plotting style module | The 7 existing figures each set their own fonts, sizes and DPI. One `src/analysis/plotstyle.py` imported everywhere gives consistent typography across the report. ~30 min. |
| Geographic coverage figure | E1 characterises coverage in time only. No figure shows which regions are sequencing-dark, though Region 2 needed a 172-record recovery to be usable at all. Uses existing data. |

B. §1.2 — third E3 constraint (after the “Fold 4 will dominate” bullet):

- **A sampling-effort control arm is mandatory.** `n_total` is in the master
  table and used by nothing. Sequencing volume plausibly tracks outbreak size,
  so a positive composition result is open to the objection that it measures
  surveillance intensity rather than genomic signal. E3 therefore runs **four
  arms**: none / `n_total` only / raw composition / JS divergence. If the
  `n_total` arm matches the composition arms, the finding is effort, not
  genomics, and must be reported as such. This is the in-scope counterpart to
  the regional confound analysis in 2.3.

C. §2.1 — new bullet at the end:

**Metadata-only genomic features.** Without sequences, the remaining
unexploited signal in the 32,141 records is: Shannon or Simpson diversity on
the subtype mix, subtype turnover rate per region-week (an antigenic-drift
proxy), and sequencing intensity per capita. All are derivable today. They are
listed here rather than in Part 1 because none is justified unless E3 shows
the existing composition block carries signal.

D. §2.3 — new bullet:

- **Hyperparameter sensitivity for B2.** The `HistGradientBoostingRegressor`
  settings were chosen once and never varied. A reviewer will ask whether the
  baseline was tuned as carefully as the genomic model, and right now the
  answer is no for both.