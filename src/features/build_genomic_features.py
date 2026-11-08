#!/usr/bin/env python3
"""
EpiSurge - Phase 3: derived genomic feature families GF2-GF5.

The repository stores sequence METADATA only. fetch_ncbi.py keeps
len(record.seq) and discards record.seq, so no nucleotides exist anywhere
in data/. Phylogenetics, mutation analysis and sequence-similarity work are
therefore impossible without a fresh download (FUTURE_WORK 2.1) and are not
attempted here.

What IS derivable from metadata, and was unexploited until now:

  GF1  composition level          already in the master table
  GF2  composition CHANGE         js_div_4w, d_prop_h3n2_4w
  GF3  sampling EFFORT            seq_per_100k, n_total_log_ratio
  GF4  SPATIAL divergence         js_vs_national, js_vs_others
  GF5  subtype NOVELTY proxy      frac_offdominant

Why GF2 exists. Composition is binary in practice (H3N2 vs H1N1, with
`other` near zero), so raw proportions and a CLR transform are monotone
functions of each other and carry identical information. Change is the only
genuinely distinct transform available on two categories, which is why it
gets its own ablation arm rather than being folded into GF1.

Why GF3 exists. Sequencing volume plausibly tracks outbreak size. Without
an effort control, a positive composition result is open to the objection
that it measures surveillance intensity rather than genomic signal. GF3 is
the control arm, not a feature we expect to help.

Why GF4 exists. It is the only SPATIAL genomic signal obtainable without
sequences, and it is the same quantity the genomic-similarity graph
relation is built from - so GF4 and the Phase 3 relation ablation are one
measurement seen two ways.

GF5 is a proxy and is labelled one: at subtype resolution it detects
H1N1<->H3N2 turnover only, never antigenic drift or clade emergence.

Design decisions recorded here because they are not obvious
-----------------------------------------------------------
RAW, NOT SHRUNK. GF2 and GF4 are built from raw proportions. DECISIONS
2026-10-29 established that shrinking toward the weekly national mean
removes exactly the between-region variation the genomic-similarity
relation exists to measure; the same argument applies to a spatial
divergence (GF4). For GF2 it applies in time: shrinkage toward a national
mean that itself moves week to week would damp genuine composition change.

TWO DIFFERENT DENOMINATORS. In the upstream composition files `n_total`
INCLUDES n_unknown, while prop_* are computed over H1N1+H3N2+other only.
This module follows that: composition features use n_class (excluding
unknown), effort features use n_total (including it, because an unsubtyped
sequence still represents sequencing effort).

WITHIN-SEASON SHIFTS ONLY. A 4-week lag is taken inside (region, season).
Season windows run MMWR week 27 to week 26, so the four NaN weeks this
creates per region-season are weeks 27-30 - outside the core evaluation
window (40-20). No core row is lost.

UNWEIGHTED NEIGHBOURS, FOR NOW. js_vs_others compares a region against the
unweighted pool of the other nine. The gravity-weighted version arrives
with the graph module in Phase 3 step 7; this file must not depend on it,
or step 2 cannot run before step 7.

Inputs:
    data/features/composition_oracle.csv
    data/features/asof_composition_L4-12.csv
    data/features/asof_composition_L8-16.csv
    data/raw/census/hhs_region_population.csv

Output:
    data/features/genomic_derived.csv     one row per (hhs_region, epiweek)

Run (from the repository root, inside the venv):
    python src/features/build_genomic_features.py

Must run BEFORE build_master_table.py, which joins this output. It reads
only the composition files and the census file - never the master table -
so there is no cycle.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from epiweeks import Week

_sys_path = str(Path(__file__).resolve().parents[1])
if _sys_path not in sys.path:
    sys.path.insert(0, _sys_path)
from config import SEASON_WINDOWS, assign_season          # noqa: E402

RAW = Path("data/raw")
FEATURES = Path("data/features")

OUT = FEATURES / "genomic_derived.csv"

CHANGE_LAG_WEEKS = 4        # GF2 comparison window
DOMINANT_WINDOW = 4         # GF5 look-back for the dominant subtype
EFFORT_WINDOW = 8           # GF3 trailing window for relative effort
EFFORT_MIN_PERIODS = 4      # below this the trailing mean is NaN, not noise

# variant tag -> source file. The tag becomes the column prefix, matching
# the naming build_master_table.py already uses for GF1.
VARIANTS = {
    "oracle": FEATURES / "composition_oracle.csv",
    "asof_L4_12": FEATURES / "asof_composition_L4-12.csv",
    "asof_L8_16": FEATURES / "asof_composition_L8-16.csv",
}

COUNT_COLS = ["n_h1n1", "n_h3n2", "n_other"]

DERIVED_COLS = [
    "js_div_4w", "d_prop_h3n2_4w",          # GF2
    "seq_per_100k", "n_total_log_ratio",    # GF3
    "js_vs_national", "js_vs_others",       # GF4
    "frac_offdominant",                     # GF5
]


# ──────────────────────────────────────────────────────────────────────
# information-theoretic helpers
# ──────────────────────────────────────────────────────────────────────

def _entropy(p):
    """Shannon entropy, base 2, over the last axis. 0*log0 treated as 0."""
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log2(p), 0.0)
    return -terms.sum(axis=-1)


def js_divergence(p, q):
    """
    Jensen-Shannon divergence, base 2, so the result is in [0, 1].

    JSD(p,q) = H((p+q)/2) - (H(p) + H(q))/2

    Rows where either distribution is all-NaN or sums to zero return NaN -
    an absent composition is not a composition of zero divergence.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    out = np.full(p.shape[0] if p.ndim > 1 else 1, np.nan, dtype=float)

    p2 = np.atleast_2d(p)
    q2 = np.atleast_2d(q)
    ok = (np.isfinite(p2).all(axis=1) & np.isfinite(q2).all(axis=1)
          & (p2.sum(axis=1) > 0) & (q2.sum(axis=1) > 0))
    if not ok.any():
        return out if p.ndim > 1 else float(out[0])

    pn = p2[ok] / p2[ok].sum(axis=1, keepdims=True)
    qn = q2[ok] / q2[ok].sum(axis=1, keepdims=True)
    m = 0.5 * (pn + qn)
    val = _entropy(m) - 0.5 * (_entropy(pn) + _entropy(qn))
    # floating point can push a true zero marginally negative
    out[ok] = np.clip(val, 0.0, 1.0)
    return out if p.ndim > 1 else float(out[0])


# ──────────────────────────────────────────────────────────────────────
# grid and inputs
# ──────────────────────────────────────────────────────────────────────

def full_grid(regions):
    """
    Every (region, epiweek) in the configured season windows, whether or
    not any sequence exists there.

    The composition files contain only POPULATED cells. Shifting a 4-week
    lag over a gappy index would silently compare t to t-6 wherever a week
    was missing, so the series is reindexed onto the complete grid first.
    """
    weeks = []
    for start, end in SEASON_WINDOWS:
        w = Week(start // 100, start % 100)
        while w.year * 100 + w.week <= end:
            weeks.append(w.year * 100 + w.week)
            w = w + 1
    grid = pd.MultiIndex.from_product([sorted(regions), weeks],
                                      names=["hhs_region", "epiweek"])
    g = grid.to_frame(index=False)
    g["season"] = g["epiweek"].apply(assign_season)
    if g["season"].isna().any():
        sys.exit("grid contains epiweeks outside every configured season")
    return g


def load_variant(tag, path):
    """Counts per (region, epiweek) for one composition variant."""
    if not path.exists():
        sys.exit(f"Missing {path} - run the composition builders first")
    df = pd.read_csv(path)

    if "origin_epiweek_int" in df.columns:          # as-of files
        df = df.rename(columns={"origin_epiweek_int": "epiweek"})
    else:                                           # oracle file
        df = df.drop(columns=["epiweek"], errors="ignore")
        df = df.rename(columns={"epiweek_int": "epiweek"})

    missing = [c for c in COUNT_COLS + ["n_total"] if c not in df.columns]
    if missing:
        sys.exit(f"{path.name} lacks expected columns: {missing}")

    df = df[["hhs_region", "epiweek"] + COUNT_COLS + ["n_total"]].copy()
    if df.duplicated(["hhs_region", "epiweek"]).any():
        sys.exit(f"{path.name} has duplicate (region, epiweek) keys")

    # n_total INCLUDES n_unknown upstream; n_class excludes it and is the
    # denominator every composition proportion uses.
    df["n_class"] = df[COUNT_COLS].sum(axis=1)
    return df.rename(columns={c: f"{c}" for c in COUNT_COLS})


def load_population():
    path = RAW / "census" / "hhs_region_population.csv"
    if not path.exists():
        sys.exit(f"Missing {path} - run fetch_census.py first")
    pop = pd.read_csv(path)[["hhs_region", "population"]]
    if pop["population"].isna().any() or (pop["population"] <= 0).any():
        sys.exit("census file has missing or non-positive populations")
    return pop


# ──────────────────────────────────────────────────────────────────────
# feature families
# ──────────────────────────────────────────────────────────────────────

def props(df):
    """Raw composition proportions as an (n, 3) array. NaN where n_class=0."""
    n = df["n_class"].values.astype(float)
    arr = df[COUNT_COLS].values.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = arr / n[:, None]
    p[n == 0] = np.nan
    return p


def gf2_change(df):
    """
    Composition change over CHANGE_LAG_WEEKS, within (region, season).

    The grid is contiguous per region-season, so a positional shift of 4 is
    exactly a 4-week lag. Shifts never cross a season boundary.
    """
    p = props(df)
    df = df.copy()
    for i, c in enumerate(COUNT_COLS):
        df[f"_p{i}"] = p[:, i]

    pcols = [f"_p{i}" for i in range(3)]
    lagged = (df.sort_values(["hhs_region", "season", "epiweek"])
                .groupby(["hhs_region", "season"], sort=False)[pcols]
                .shift(CHANGE_LAG_WEEKS))
    lagged = lagged.reindex(df.index)

    df["js_div_4w"] = js_divergence(df[pcols].values, lagged[pcols].values)
    df["d_prop_h3n2_4w"] = df["_p1"] - lagged["_p1"]        # index 1 = h3n2
    return df[["js_div_4w", "d_prop_h3n2_4w"]]


def gf3_effort(df, pop):
    """
    Sampling effort. n_total (including unsubtyped sequences) is the right
    numerator: an unsubtyped sequence is still sequencing that happened.

    n_total_log_ratio is log2((n_total(t) + 1) / (trailing mean + 1)), where
    the trailing mean is taken over the EFFORT_WINDOW weeks STRICTLY BEFORE
    t within the region-season. Zero means this week matches its recent
    baseline, +1 means double, -1 means half.

    An expanding mean was tried first and is wrong. Season windows open at
    MMWR week 27 (July), so a dozen near-zero summer weeks hold the
    denominator down and the ratio climbs monotonically through the autumn:
    measured, the core-season median was 2.2 and 37% of rows exceeded 3.
    That is not relative effort, it is a seasonal clock in disguise - and
    the model is already given season position honestly, as
    season_week_idx. A genomic feature that smuggles it in a second time
    would inflate any apparent genomic contribution.

    A plain ratio against the trailing mean was tried next and is also
    wrong: with min_periods=4 the denominator can be a fraction, and the
    measured maximum reached 288 with sd 8.8 against a median of 1.4. Trees
    bin features so the tail would not have hurt E3, but the same column
    feeds the GNN and the Granger work, where it would. The +1 inside the
    log removes the near-zero denominator entirely.
    """
    df = df.merge(pop, on="hhs_region", how="left")
    if df["population"].isna().any():
        sys.exit("a region in the composition grid has no census population")

    out = pd.DataFrame(index=df.index)
    out["seq_per_100k"] = df["n_total"].astype(float) / df["population"] * 1e5

    ordered = df.sort_values(["hhs_region", "season", "epiweek"])
    prior = (ordered.groupby(["hhs_region", "season"], sort=False)["n_total"]
                    .shift(1))
    trail = (prior.groupby([ordered["hhs_region"], ordered["season"]], sort=False)
                  .rolling(EFFORT_WINDOW, min_periods=EFFORT_MIN_PERIODS).mean()
                  .reset_index(level=[0, 1], drop=True))
    trail = trail.reindex(df.index)

    out["n_total_log_ratio"] = (np.log2(df["n_total"].astype(float) + 1.0)
                                - np.log2(trail + 1.0))
    return out


def gf4_spatial(df):
    """
    Divergence of a region's composition from the rest of the country, per
    epiweek.

    js_vs_national compares against the pooled national composition, which
    INCLUDES the region itself - for a region supplying most of that week's
    sequences the comparison is partly self-referential, and that is why
    js_vs_others exists alongside it.

    js_vs_others pools only the other nine regions, so it is the cleaner
    quantity and the one the graph relation corresponds to.
    """
    p = props(df)

    nat = df.groupby("epiweek")[COUNT_COLS].transform("sum").values.astype(float)
    own = df[COUNT_COLS].values.astype(float)
    others = nat - own

    out = pd.DataFrame(index=df.index)
    out["js_vs_national"] = js_divergence(p, nat)
    out["js_vs_others"] = js_divergence(p, others)
    return out


def gf5_novelty(df):
    """
    Fraction of this week's sequences whose subtype differs from the
    region's dominant subtype over the DOMINANT_WINDOW weeks STRICTLY
    BEFORE t.

    Strictly before matters: including t would make the measure partly
    self-defining, since t's own sequences would help choose the baseline
    they are then compared against.

    NaN where t has no classified sequences, or where the preceding window
    has none to establish a baseline. Proxy only - at subtype resolution
    this is H1N1<->H3N2 turnover, not drift or clade emergence.
    """
    d = df.sort_values(["hhs_region", "season", "epiweek"]).copy()
    g = d.groupby(["hhs_region", "season"], sort=False)

    prior = {}
    for i, c in enumerate(COUNT_COLS):
        shifted = g[c].shift(1)
        prior[i] = (shifted.groupby([d["hhs_region"], d["season"]], sort=False)
                           .rolling(DOMINANT_WINDOW, min_periods=1).sum()
                           .reset_index(level=[0, 1], drop=True))

    prior_arr = np.column_stack([prior[i].reindex(d.index).values for i in range(3)])
    prior_tot = prior_arr.sum(axis=1)

    dominant = np.where(prior_tot > 0, prior_arr.argmax(axis=1), -1)
    own = d[COUNT_COLS].values.astype(float)
    n_class = d["n_class"].values.astype(float)

    frac = np.full(len(d), np.nan)
    valid = (n_class > 0) & (prior_tot > 0)
    if valid.any():
        idx = np.where(valid)[0]
        on_dominant = own[idx, dominant[idx]]
        frac[idx] = 1.0 - on_dominant / n_class[idx]

    return pd.Series(frac, index=d.index).reindex(df.index).to_frame("frac_offdominant")


# ──────────────────────────────────────────────────────────────────────

def build_variant(tag, path, grid, pop):
    src = load_variant(tag, path)
    df = grid.merge(src, on=["hhs_region", "epiweek"], how="left")
    if len(df) != len(grid):
        sys.exit(f"{tag}: join changed row count")

    # absent cell = no sequences, which is a count of zero, not unknown.
    # The composition PROPORTIONS stay NaN (props() handles that) - only
    # the counts are filled, so sums and dominance look-backs are correct.
    for c in COUNT_COLS + ["n_total", "n_class"]:
        df[c] = df[c].fillna(0).astype(float)

    feats = pd.concat(
        [gf2_change(df), gf3_effort(df, pop), gf4_spatial(df), gf5_novelty(df)],
        axis=1,
    )
    feats = feats[DERIVED_COLS]
    feats.columns = [f"{tag}_{c}" for c in DERIVED_COLS]

    out = pd.concat([df[["hhs_region", "epiweek"]], feats], axis=1)
    populated = int((df["n_class"] > 0).sum())
    return out, populated


def main():
    print("EpiSurge Phase 3 - derived genomic features GF2-GF5")
    print("=" * 70)

    pop = load_population()

    regions = None
    for tag, path in VARIANTS.items():
        if path.exists():
            regions = sorted(pd.read_csv(path)["hhs_region"].unique())
            break
    if regions is None:
        sys.exit("no composition file found in data/features/")

    grid = full_grid(regions)
    print(f"\nGrid: {len(regions)} regions x "
          f"{grid['epiweek'].nunique()} epiweeks = {len(grid)} cells")

    merged = grid[["hhs_region", "epiweek"]].copy()
    for tag, path in VARIANTS.items():
        feats, populated = build_variant(tag, path, grid, pop)
        merged = merged.merge(feats, on=["hhs_region", "epiweek"], how="left")
        print(f"\n  {tag:<11} populated cells: {populated:>5} "
              f"({100 * populated / len(grid):.1f}% of grid)")
        for c in DERIVED_COLS:
            col = f"{tag}_{c}"
            v = merged[col]
            n = int(v.notna().sum())
            if n:
                print(f"      {c:<18} n={n:>5}  "
                      f"mean={v.mean():+.4f}  sd={v.std():.4f}  "
                      f"min={v.min():+.4f}  max={v.max():+.4f}")
            else:
                print(f"      {c:<18} n=0   [!] all missing")

    # ── checks ────────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("Checks")
    print("-" * 70)
    failures = []

    for tag in VARIANTS:
        for c in ["js_div_4w", "js_vs_national", "js_vs_others"]:
            v = merged[f"{tag}_{c}"].dropna()
            if not v.empty and (v.min() < -1e-9 or v.max() > 1 + 1e-9):
                failures.append(f"{tag}_{c} outside [0,1]: "
                                f"[{v.min():.6f}, {v.max():.6f}]")
        v = merged[f"{tag}_d_prop_h3n2_4w"].dropna()
        if not v.empty and (v.min() < -1 - 1e-9 or v.max() > 1 + 1e-9):
            failures.append(f"{tag}_d_prop_h3n2_4w outside [-1,1]")
        v = merged[f"{tag}_frac_offdominant"].dropna()
        if not v.empty and (v.min() < -1e-9 or v.max() > 1 + 1e-9):
            failures.append(f"{tag}_frac_offdominant outside [0,1]")

    print(f"  divergences within [0,1]: {'yes' if not failures else 'NO'}")

    # the four NaN weeks GF2 creates per region-season must be pre-core
    core_mask = merged["epiweek"].apply(lambda e: (e % 100) >= 40 or (e % 100) <= 20)
    for tag in VARIANTS:
        pop_cells = merged[core_mask].copy()
        print(f"  {tag:<11} core-season rows with js_div_4w defined: "
              f"{int(pop_cells[f'{tag}_js_div_4w'].notna().sum()):>5}")

    if merged.duplicated(["hhs_region", "epiweek"]).any():
        failures.append("output has duplicate (region, epiweek) keys")
    print(f"  one row per (region, epiweek): "
          f"{'yes' if not merged.duplicated(['hhs_region', 'epiweek']).any() else 'NO'}")

    FEATURES.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT, index=False)
    print(f"\n  Saved -> {OUT}")
    print(f"  Shape: {merged.shape[0]} rows x {merged.shape[1]} columns")

    if failures:
        print("\n  FAILURES:")
        for f in failures:
            print(f"    [FAIL] {f}")
        sys.exit(1)
    print("\n  All checks passed. Next: build_master_table.py")


if __name__ == "__main__":
    main()