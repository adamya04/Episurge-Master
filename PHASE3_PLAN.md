# EpiSurge — Phase 3 Plan

**Status:** proposed, not yet implemented. Supersedes Master Plan §4–§7 where
they conflict; the specific Master Plan edits are listed in §2.

Phase 3 turns the completed data foundation (Phase 1) and the delay/baseline
measurements (Phase 2: E1, E2) into the evidence base for the paper. It
contains four experiments, one model family, eight tables and thirteen
figures.

---

## 1. What Phase 3 claims — and what it does not

### The three claims

**C1 — Delay-aware evaluation of genomic forecasting features.**
Published work that uses pathogen genomics for incidence forecasting indexes
sequences by *collection* date. That is an oracle: at forecast origin T, a
sequence collected before T but published after T is not available. E1
measured what the oracle is worth — 97.1 % coverage before the epidemic peak
against 34.8 % under publication-dating, a 62.3-point gap, with 8–11
consecutive onset weeks at exactly zero in every season. Phase 3 measures what
that gap costs a forecast. **This is the primary contribution.**

**C2 — Horizon-resolved marginal value of composition.**
Whether subtype composition adds skill over case history, separately at
h = 1, 2, 3, 4, 6, 8, phase-stratified, paired, per fold.

**C3 — Which graph relation carries information.**
A tri-relational graph over 10 HHS regions (gravity, climate similarity,
genomic similarity) evaluated against a density-matched random-graph control.
The question is not "is the GNN accurate" but "does any relation beat random".

### What Phase 3 does not claim

**No accuracy superiority over operational forecasting systems.** The honest
comparator for regional wILI is the CDC FluSight ensemble. A hand-written
10-node GNN trained on 1,980 rows will not beat it, and a reviewer will make
that comparison whether we invite it or not. Framing the paper around beating
state of the art is the fastest route to rejection. The contribution is the
measurement, not the model.

**No transmission claims.** Subtype composition similarity between two regions
is not evidence of transmission between them. Every genomic result is phrased
as predictive association.

**No guarantee of a positive result.** E3 can land in three worlds:

| world | result | paper |
|---|---|---|
| 1 | oracle composition helps, as-of does not | **strongest.** "Genomic composition carries forecasting signal; publication delay destroys it." E1 becomes the mechanism. |
| 2 | neither helps | Bounds what a genomic forecasting system can deliver at subtype resolution. Weaker, still reportable with the coverage work. |
| 3 | both help | Good for a forecasting system, weaker for the paper — the delay story shrinks and the contest moves to accuracy, which we lose. |

All three are publishable. Only world 1 is publishable *well*. We do not know
which we are in, and nothing downstream of E3 should be built until we do.

---

## 2. Master Plan edits required

Apply these locally and push, so the plan and the build agree.

| § | change |
|---|---|
| §2 | Add: "No accuracy-superiority claim is made against operational forecasting systems. The contribution is measurement." |
| §4.2 | Add the five genomic feature families GF1–GF5 (§3 below). The existing feature list covers GF1 and part of GF3 only. |
| §4.3 | Unchanged in substance (gravity / climate-similarity / genomic-similarity), but record that the genomic relation is built from **raw** proportions with count-based down-weighting, per DECISIONS 2026-10-29. |
| §5 | Reframe Phase 3 from "build EpiGraph-GNN" to "evaluate genomic signal, then evaluate graph relations". The GNN is the instrument for C3, not an accuracy play. Its construction is **contingent on E3** — see §9. |
| §7.1 | Record that B1 climatological, not B0 persistence, is the long-horizon reference (already in DECISIONS, not yet in the plan). |
| §7.2 | Replace the 4-arm E3 with the 7-arm design in §4. |
| §7.3 | Transfer entropy: remove "mandatory", replace with the surrogate-based design in §6. An inconclusive result is a declared, reportable outcome. |
| §7.4 | Add panel Granger as a primary evidence line (§5), pooled on first differences, phase-stratified, with a permutation null. |
| new §7.6 | Multiplicity policy: one pre-declared primary endpoint, Holm correction within families, sign-consistency across folds as the primary inference tool. |

Also: delete the dead line `_, _, _ = None, None, None` in
`config.verify_no_censoring`, and remove the stale "retitle the coverage
figures" warning from `RESULTS.md` — both titles were corrected in
`delay_coverage_analysis.py:332,364`.

---

## 3. Genomic feature expansion (GF1–GF5)

The repository stores sequence **metadata** only — 32,141 human records with
accession, state, collection date, submission date, subtype and length.
`fetch_ncbi.py:100` keeps `len(record.seq)` and discards the sequence.
Phylogenetics, mutation analysis and sequence-similarity work are therefore
impossible without a fresh download; that is recorded in FUTURE_WORK §2.1 and
is not revisited here.

What *is* derivable from metadata, and is currently unexploited:

### GF1 — composition level *(have)*
`prop_h1n1`, `prop_h3n2`, `prop_other`, plus shrunk variants. Oracle and
as-of. Already in the master table.

### GF2 — composition change *(new)*
- `js_div_4w` — Jensen–Shannon divergence between the composition at the
  origin window and the window four weeks earlier.
- `d_prop_h3n2_4w` — signed change in H3N2 share.

Composition is binary, so level and CLR are monotone transforms of each other
and carry the same information. **Change is the only genuinely distinct
transform available**, which is why it gets its own arm.

### GF3 — sampling effort *(new, partly have)*
- `n_total` *(have)*
- `seq_per_100k` — `n_total / population × 1e5`
- `n_total_rel` — `n_total` over the region's season-mean `n_total`

These are the **confound controls**. Sequencing volume plausibly tracks
outbreak size, so without them a positive composition result is open to the
objection that it measures surveillance intensity, not genomic signal.

### GF4 — spatial composition divergence *(new)*
- `js_vs_national` — JS divergence between the region's composition and the
  pooled national composition that week.
- `js_vs_neighbours` — JS divergence against the gravity-weighted mean
  composition of the other nine regions.

This is the only **spatial** genomic signal obtainable without sequences, and
it is the quantity the genomic-similarity graph relation is built from. It
makes GF4 and C3 the same measurement seen two ways.

### GF5 — subtype novelty proxy *(new)*
- `frac_offdominant` — fraction of the region-week's sequences whose subtype
  differs from that region's dominant subtype over the preceding four weeks.

A crude "something different is arriving" signal. **Stated limitation:** at
subtype resolution this detects H1N1↔H3N2 turnover only, not antigenic drift
or clade emergence. It is a proxy and must be labelled one.

**Missingness.** Every GF feature is NaN where coverage is absent, which under
as-of is most of the onset window. `HistGradientBoostingRegressor` handles NaN
natively; the GNN needs explicit imputation plus a missingness indicator.
Imputation strategy is itself an ablation axis — do not silently zero-fill.

---

## 4. E3 — genomic ablation *(the gate)*

**Research question.** Does genomic composition add forecasting skill over
case history, and how much of any gain is destroyed by publication delay?

**Design.** Seven arms. Each adds a feature block to B2's epidemiological
features (8 as-of ILI lags, region, season-week index). Gradient boosting,
paired rows, 5 rolling-origin folds, 6 horizons, 2 phase strata.

| arm | features added | answers |
|---|---|---|
| A0 | none *(= existing B2)* | reference |
| A1 | GF3 effort only | is any gain just surveillance intensity? |
| A2 | GF1 as-of | deployable level |
| A3 | GF1 + GF2 as-of | deployable level + change |
| A4 | GF1+GF2+GF4+GF5 as-of | **deployable, full** |
| A5 | GF1 oracle | literature-equivalent |
| A6 | GF1+GF2+GF4+GF5 oracle | **ceiling** |

**The four comparisons that matter.**

| contrast | reads as |
|---|---|
| A6 − A4 | **the delay cost — primary endpoint (C1)** |
| A5 − A0, A6 − A0 | does the oracle carry signal at all |
| A4 − A0 | does deployable genomics carry signal (C2) |
| A4 − A1 | genomics, or effort |

**Primary endpoint, declared before running:** mean skill difference
A6 − A4 in MAE, pooled across horizons, pre-peak stratum, reported per fold
with sign-consistency. Everything else is secondary and Holm-corrected within
its family.

**Inference.** Five folds is too few for a t-test. Primary tool is
sign-consistency across folds (the rule already standing in DECISIONS: a
horizon whose sign flips supports no claim). Secondary: block-bootstrap CIs,
resampling region-season blocks to respect autocorrelation.

**Fold 4 (test 2022-23) is reported separately.** It is the COVID
discontinuity — pre-COVID seasons peak at MMWR weeks 05–07, 2022-23 peaked at
week 47. It broke every baseline in E2 and it will break this. Headline is the
four pre-COVID folds; fold 4 is reported as a regime-change finding in its own
right.

**Runtime.** 7 arms × 5 folds × 6 horizons × 2 strata on 1,980 rows. Under a
minute. No Colab.

**Acceptance criteria.** Pairing verified in code; every arm evaluated on
identical row sets per (fold, horizon, stratum); `baseline_results.csv`-style
audit trail with one row per arm × fold × horizon × seed × stratum.

---

## 5. E4 — panel Granger causality

**Research question.** Does composition change lead incidence change, model
free, without a fitted forecaster?

**Design.**
- First differences. wILI is non-stationary within season — median ADF
  p = 0.545, only 7 % stationary at 0.05.
- **Pooled panel with region fixed effects.** Usable segments are 14–24
  contiguous points per region-season; per-region Granger is dead beyond
  lag 1 (at p = 2 you fit five parameters on twelve rows). Per-region results
  appear only as a supplementary heatmap, explicitly labelled underpowered.
- Lag order by BIC on the pooled panel, capped at p = 2.
- **Both directions.** ΔwILI → Δcomposition is run as a control: if the
  reverse direction is equally strong, we have a shared seasonal driver, not a
  lead, and must say so.
- Phase-stratified.
- **Oracle and as-of both.** The contrast is the result, exactly as in E3.
- **Permutation null:** shuffle region labels within week, 1,000 reps. This
  destroys the region-level pairing while preserving week-level dynamics,
  which the asymptotic F-distribution does not account for in a short panel.

**Reported.** F statistic, asymptotic p, permutation p, partial R², per
direction × stratum × lag × composition variant.

**Runtime.** Seconds.

---

## 6. E5 — transfer entropy

Transfer entropy was demoted in FUTURE_WORK §1.2 for good reason: it needs
hundreds of points per series and ours are 14–24. It is retained here in a
form that is defensible at this sample size, because there is a version that
contributes and a version that does not.

**What does not work:** a per-region TE heatmap with bare numbers. At 20
points per series the estimator is bias-dominated; the picture would look
authoritative and mean nothing.

**What does work:** TE as a second, model-free channel for the *same*
oracle-versus-as-of contrast that E3 and E4 test.

**Design.**
- Pooled across region-seasons, lag 1, three-bin equiprobable discretisation.
- `TE(Δcomposition → ΔwILI)` and the reverse.
- **Surrogates:** circular time-shift within region-season, 1,000 reps. Report
  the observed estimate against the surrogate distribution, never alone.
- **Bias floor:** report `mean(TE_surrogate)` explicitly;
  `TE_corrected = TE_observed − mean(TE_surrogate)`.
- Stratified by oracle / as-of.

**Primary readout:** (a) does the observed estimate exceed the surrogate 95th
percentile, and (b) is `TE_oracle > TE_asof` consistently across strata? A
consistent ordering is evidence even when both sit inside wide bands.

**Declared acceptable outcome:** "information-theoretic analysis at this
sample size cannot resolve a composition → incidence channel; the surrogate
distribution overlaps the observed estimate." That is an honest, reportable
result and reviewers respect it far more than a confident heatmap built on
twenty points. **Do not build an evidence line on E5.** It corroborates E4 or
it says nothing.

**Runtime.** Minutes, dominated by surrogates.

---

## 7. E6 — model ladder and relation ablation

**Research question.** Does a graph help, which relation carries information,
and does any of it beat a random graph of the same density?

**Ladder.** Multi-horizon heads (one model predicts all six horizons), so the
fit count stays tractable.

| model | description | isolates |
|---|---|---|
| M0 | HistGBR, epi only *(= B2)* | tabular reference |
| M1 | HistGBR + best genomic block from E3 | tabular ceiling |
| M2 | GRU encoder, no graph | does the graph add anything? |
| M3a | GRU + gravity relation only | |
| M3b | GRU + climate-similarity only | which relation? |
| M3c | GRU + genomic-similarity only | |
| M4 | GRU + tri-relational, learned relation gates | full EpiGraph-GNN |
| M5 | GRU + density-matched random graph | **the G5 control — non-negotiable** |

**M5 is the result that makes M4 meaningful.** At 10 nodes almost any
aggregation helps a little by smoothing. If M4 does not beat M5, the graph is
doing smoothing, not carrying relational information, and the paper says so.

**Relation gates.** The learned gate values are a reportable readout, not
decoration: they say which relation the model actually used, per fold.

**Protocol.** 5 seeds (meaningful here; vacuous for the deterministic
baselines), 5 folds, 8 variants = 200 fits. Early stopping on a held-out tail
of the training span, never on the test season.

**Runtime.** Minutes per fit on CPU at 10 nodes. 3–10 hours total; overnight
locally, or Colab if preferred. No GPU required.

**Contingency.** If E3 world 2 (no genomic signal at all), M3c and the genomic
relation have nothing to carry. M2/M3a/M3b/M4/M5 still answer C3, and that
remains worth running — but the paper's genomic claim becomes the negative
result plus the delay mechanism, and the GNN's role shrinks accordingly. See §9.

---

## 8. Deliverables

### Tables

| id | content | supports |
|---|---|---|
| T1 | Dataset characteristics — seasons, regions, rows, coverage by variant, delay quantiles by season | setup |
| T2 | E3 skill % by arm × horizon, per fold, with CIs and a sign-consistency column | C2 |
| T3 | **The delay-cost table** — A6 − A4 by horizon and stratum | **C1** |
| T4 | Effort controls — A1 and A4 − A1, showing the result is not surveillance intensity | C2 |
| T5 | Granger — direction × stratum × lag × variant, with F, p, permutation p, partial R² | C1, C2 |
| T6 | Transfer entropy with surrogate bands, oracle vs as-of | corroboration |
| T7 | Model ladder — MAE and RMSE by horizon, per fold, M0–M5 | C3 |
| T8 | Relation ablation — each relation against M5, plus learned gate values | **C3** |

### Figures

Four already exist. Nine are new. Each is tied to a table; none is decorative.

| id | figure | status |
|---|---|---|
| F1 | Coverage vs weeks-from-peak, oracle vs as-of | have |
| F2 | Onset gap per season | have |
| F3 | Collection-to-publication delay distribution | have |
| F4 | Baseline skill by fold | have |
| F5 | **E3 headline — skill delta vs horizon, oracle and as-of overlaid, bootstrap CI ribbons** | new |
| F6 | Seven-arm comparison with effort controls separated | new |
| F7 | E3 phase-stratified — pre-peak vs peak-or-after | new |
| F8 | Geographic coverage choropleth — which regions are sequencing-dark (uses GADM, already fetched) | new |
| F9 | Granger effect by lag and stratum, with permutation null band | new |
| F10 | TE observed vs surrogate distribution, oracle vs as-of | new |
| F11 | Model ladder — MAE by horizon, one line per model | new |
| F12 | Relation gate values per fold | new |
| F13 | M4 vs M5 skill distributions across seeds and folds | new |

**Production standard.** One `src/analysis/plotstyle.py` imported everywhere:
shared fonts, sizes, DPI, colour cycle, colourblind-safe palette. The seven
existing figures each set their own and must be regenerated once the module
exists.

---

## 9. Execution order

| # | step | new files | time |
|---|---|---|---|
| 1 | Plot style module; regenerate the 7 existing figures; clear the stale RESULTS.md warning and the dead config line | `src/analysis/plotstyle.py` | 1 h |
| 2 | Genomic feature families GF2–GF5; join into master table; extend validation | `src/features/build_genomic_features.py`, edits to `build_master_table.py`, `validate_all_sources.py` | 1 day |
| 3 | **E3 — run it. This is the gate.** | `src/analysis/genomic_ablation.py` | 0.5 day |
| 4 | **Decision point.** Read E3. Which world are we in? Everything after this is conditioned on the answer. | — | — |
| 5 | E4 Granger | `src/analysis/granger_panel.py` | 0.5 day |
| 6 | E5 transfer entropy | `src/analysis/transfer_entropy.py` | 0.5 day |
| 7 | Graph construction — three relations plus the density-matched random control | `src/models/graphs.py` | 1 day |
| 8 | GRU encoder, hand-written GraphSAGE, relation gates, trainer | `src/models/` | 2 days |
| 9 | E6 model ladder and relation ablation | `src/analysis/model_ladder.py`, `src/analysis/relation_ablation.py` | 1 day |
| 10 | Table emitter (markdown + LaTeX); final figure regeneration | `src/analysis/make_tables.py` | 1 day |

Roughly two focused weeks. Steps 1–3 are three days and settle whether the
rest is worth building in its planned form.

**Why E3 first.** It is cheap, it is decisive, and every later design choice
depends on it. Building the GNN before knowing whether composition carries any
signal risks a finished architecture that demonstrates nothing.

---

## 10. Stopping conditions

Recorded in advance so they are not renegotiated under deadline pressure.

- **E3 shows no signal in any arm, oracle included.** Do not build the genomic
  graph relation. The paper becomes the delay mechanism plus a bounded
  negative result. M3c is dropped; C3 is answered with two relations.
- **A4 − A1 ≈ 0 while A4 − A0 > 0.** The gain is sequencing effort, not
  genomics. Report it as such; it is a real and useful finding about surveillance
  intensity as a forecasting covariate.
- **M4 does not beat M5.** The graph smooths rather than relates. Say so; do
  not tune until it wins.
- **Reverse-direction Granger matches the forward direction.** Shared seasonal
  driver. No lead claim.

Any of these is a result. None is a reason to change the question.