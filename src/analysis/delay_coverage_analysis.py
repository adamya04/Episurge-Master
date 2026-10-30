#!/usr/bin/env python3
"""
EpiSurge - Phase 2a / E1: delay and coverage-timing analysis.

The headline N5 statistic so far has been a level difference: core-season
genomic coverage is 73.9% under oracle indexing and 60.3% under as-of
indexing. That understates the finding.

Coverage is not uniformly attenuated by publication delay - it is
RELOCATED. Oracle coverage is high early in the season (sequences are
collected when the season starts) and falls later. As-of coverage is the
mirror image: near zero early, near complete late, because an early-season
forecast origin has an empty trailing window (nothing collected in summer,
and nothing collected recently has been published yet).

The operational consequence is the actual contribution: genomic information
arrives AFTER the part of the season a forecast could change a decision
about. This script quantifies that two ways -

  1. by calendar phase   (weeks 40-52 vs weeks 1-20) - simple, reproducible
  2. by position relative to each season's own epidemic peak - this is the
     epidemiologically meaningful version and the headline figure

It also re-derives the collection-to-submission delay distribution, overall
and per season, so the 76-day median has per-season evidence behind it.

This script is descriptive only. It establishes WHEN genomic data is
available. It says nothing about whether genomic data improves a forecast -
that is E3 (representation ablation) and E4 (panel Granger).

Inputs:
    data/processed/master_region_week.csv
    data/reports/delay_distribution.csv

Outputs:
    data/reports/coverage_by_phase.csv
    data/reports/coverage_by_week.csv
    data/reports/coverage_by_peak_offset.csv
    data/reports/delay_summary.csv
    figures/fig_coverage_crossover.png
    figures/fig_coverage_vs_peak.png
    figures/fig_onset_gap.png
    figures/fig_delay_distribution.png

Run (from the repository root, inside the venv):
    python src/analysis/delay_coverage_analysis.py

Requires matplotlib:
    pip install matplotlib
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")           # no display needed; write files only
import matplotlib.pyplot as plt

PROCESSED = Path("data/processed")
REPORTS = Path("data/reports")
FIGURES = Path("figures")

MASTER = PROCESSED / "master_region_week.csv"
DELAYS = REPORTS / "delay_distribution.csv"

import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from config import SEASON_LABELS as SEASONS



# The three genomic availability variants carried in the master table.
# oracle leaks by construction and is the N5 ceiling, not a usable feature.
VARIANTS = [
    ("oracle",     "has_genomic_oracle",        "Oracle (collection-dated)"),
    ("asof_L8_16", "has_genomic_asof_L8_16",    "As-of, 8-16 wk window"),
    ("asof_L4_12", "has_genomic_asof_L4_12",    "As-of, 4-12 wk window"),
]

# Calendar phase split. Core season is MMWR weeks 40-20; the calendar-year
# boundary is the natural dividing line and needs no tuning.
EARLY_LABEL = "early (wk 40-52)"
LATE_LABEL = "late (wk 1-20)"

PLOT_COLOURS = {
    "oracle": "#1f4e79",        # dark blue
    "asof_L8_16": "#c0504d",    # brick red
    "asof_L4_12": "#e8a33d",    # amber
}


# ──────────────────────────────────────────────────────────────────────
# loading and derived columns
# ──────────────────────────────────────────────────────────────────────

def load_master():
    if not MASTER.exists():
        sys.exit(f"Missing {MASTER} - run build_master_table.py first")
    df = pd.read_csv(MASTER)

    core = df[df["in_core_season"] == True].copy()      # noqa: E712
    if core.empty:
        sys.exit("No core-season rows found in the master table")

    core["week"] = core["epiweek"] % 100
    core["phase"] = np.where(core["week"] >= 40, EARLY_LABEL, LATE_LABEL)

    # Position within the season, 0 = first core week. Derived by ranking the
    # season's own epiweeks rather than by arithmetic on week numbers, so an
    # MMWR 53-week year cannot shift the index.
    core["season_week_idx"] = (
        core.groupby("season")["epiweek"].rank(method="dense").astype(int) - 1
    )

    for key, col, _ in VARIANTS:
        if col not in core.columns:
            sys.exit(f"Master table is missing expected column {col}")
        core[col] = core[col].astype(bool)

    return core


def add_peak_offset(core):
    """
    Weeks relative to each season's own epidemic peak.

    The peak is the epiweek with the highest mean wILI across the ten
    regions. A single national peak per season is used rather than a
    per-region peak so that the x-axis is comparable across regions; the
    per-region peaks sit within a week or two of it.
    """
    peaks = {}
    for season, grp in core.groupby("season"):
        national = grp.groupby("epiweek")["wili_final"].mean()
        peak_ew = int(national.idxmax())
        peak_idx = int(grp.loc[grp["epiweek"] == peak_ew, "season_week_idx"].iloc[0])
        peaks[season] = {
            "peak_epiweek": peak_ew,
            "peak_week": peak_ew % 100,
            "peak_idx": peak_idx,
            "peak_wili": float(national.max()),
        }

    core["peak_idx"] = core["season"].map(lambda s: peaks[s]["peak_idx"])
    core["weeks_from_peak"] = core["season_week_idx"] - core["peak_idx"]
    core["peak_side"] = np.where(core["weeks_from_peak"] < 0, "pre-peak", "peak or after")
    return core, peaks


# ──────────────────────────────────────────────────────────────────────
# tables
# ──────────────────────────────────────────────────────────────────────

def coverage_by_phase(core):
    rows = []
    for key, col, label in VARIANTS:
        for season in SEASONS:
            s = core[core["season"] == season]
            for phase in (EARLY_LABEL, LATE_LABEL):
                p = s[s["phase"] == phase]
                rows.append({
                    "variant": key, "variant_label": label,
                    "season": season, "phase": phase,
                    "n_rows": len(p),
                    "n_covered": int(p[col].sum()),
                    "pct_covered": round(100 * p[col].mean(), 1) if len(p) else np.nan,
                })
        # Pooled two ways. Seasons contribute unequal numbers of rows to each
        # phase (the peak falls at a different week each season), so the
        # row-weighted pool is dominated by whichever season has the longest
        # run in that phase. The unweighted season mean is reported alongside
        # it; where the two disagree, quote both.
        for phase in (EARLY_LABEL, LATE_LABEL):
            p = core[core["phase"] == phase]
            rows.append({
                "variant": key, "variant_label": label,
                "season": "ALL (row-weighted)", "phase": phase,
                "n_rows": len(p),
                "n_covered": int(p[col].sum()),
                "pct_covered": round(100 * p[col].mean(), 1),
            })
            per_season = [100 * core[(core.season == s) & (core.phase == phase)][col].mean()
                          for s in SEASONS]
            rows.append({
                "variant": key, "variant_label": label,
                "season": "ALL (season mean)", "phase": phase,
                "n_rows": len(p),
                "n_covered": int(p[col].sum()),
                "pct_covered": round(float(np.mean(per_season)), 1),
            })
    return pd.DataFrame(rows)


def coverage_by_peak_side(core):
    rows = []
    for key, col, label in VARIANTS:
        for season in SEASONS:
            s = core[core["season"] == season]
            for side in ("pre-peak", "peak or after"):
                p = s[s["peak_side"] == side]
                rows.append({
                    "variant": key, "variant_label": label,
                    "season": season, "peak_side": side,
                    "n_rows": len(p),
                    "n_covered": int(p[col].sum()),
                    "pct_covered": round(100 * p[col].mean(), 1) if len(p) else np.nan,
                })
        # See coverage_by_phase: pre-peak run length differs sharply by season
        # (18 / 7 / 12 weeks), so the row-weighted pool is ~49% 2019-20, which
        # is also the best-covered season. Both pooling methods are reported.
        for side in ("pre-peak", "peak or after"):
            p = core[core["peak_side"] == side]
            per_season = [100 * core[(core.season == s) & (core.peak_side == side)][col].mean()
                          for s in SEASONS]
            for lab, val in [("ALL (row-weighted)", round(100 * p[col].mean(), 1)),
                             ("ALL (season mean)", round(float(np.mean(per_season)), 1))]:
                rows.append({
                    "variant": key, "variant_label": label,
                    "season": lab, "peak_side": side,
                    "n_rows": len(p),
                    "n_covered": int(p[col].sum()),
                    "pct_covered": val,
                })
    return pd.DataFrame(rows)


def coverage_by_week(core):
    recs = []
    for (season, idx), grp in core.groupby(["season", "season_week_idx"]):
        rec = {
            "season": season,
            "season_week_idx": idx,
            "epiweek": int(grp["epiweek"].iloc[0]),
            "week": int(grp["week"].iloc[0]),
            "weeks_from_peak": int(grp["weeks_from_peak"].iloc[0]),
            "n_regions": len(grp),
            "mean_wili": round(float(grp["wili_final"].mean()), 3),
        }
        for key, col, _ in VARIANTS:
            rec[f"{key}_n_covered"] = int(grp[col].sum())
        recs.append(rec)
    return pd.DataFrame(recs).sort_values(["season", "season_week_idx"])


def coverage_by_peak_offset(core):
    recs = []
    for off, grp in core.groupby("weeks_from_peak"):
        rec = {"weeks_from_peak": int(off), "n_rows": len(grp)}
        for key, col, _ in VARIANTS:
            rec[f"{key}_pct"] = round(100 * grp[col].mean(), 1)
        recs.append(rec)
    return pd.DataFrame(recs).sort_values("weeks_from_peak")


def delay_summary():
    if not DELAYS.exists():
        print(f"  [WARN] {DELAYS} not found - skipping delay tables and figure")
        return None, None
    d = pd.read_csv(DELAYS)
    d = d[d["delay_days"].notna()].copy()
    d["delay_days"] = d["delay_days"].astype(float)

    rows = []
    for season in SEASONS + ["ALL"]:
        s = d if season == "ALL" else d[d["season"] == season]
        if s.empty:
            continue
        v = s["delay_days"].values
        rows.append({
            "season": season, "n_sequences": len(v),
            "p10": int(np.percentile(v, 10)), "p25": int(np.percentile(v, 25)),
            "median": int(np.median(v)),
            "p75": int(np.percentile(v, 75)), "p90": int(np.percentile(v, 90)),
            "mean": round(float(np.mean(v)), 1),
            "median_weeks": round(float(np.median(v)) / 7, 1),
            "pct_over_8wk": round(100 * float((v > 56).mean()), 1),
            "pct_over_12wk": round(100 * float((v > 84).mean()), 1),
        })
    return d, pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# figures
# ──────────────────────────────────────────────────────────────────────

def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_crossover(byweek, core):
    """Coverage against position in the season, oracle vs as-of."""
    fig, ax = plt.subplots(figsize=(9, 5))

    labels = (core.sort_values("season_week_idx")
                  .drop_duplicates("season_week_idx")[["season_week_idx", "week"]])
    idx_to_week = dict(zip(labels.season_week_idx, labels.week))

    for key, _, label in VARIANTS:
        # faint per-season lines
        for season in SEASONS:
            s = byweek[byweek["season"] == season].sort_values("season_week_idx")
            if s.empty:
                continue
            ax.plot(s.season_week_idx, 100 * s[f"{key}_n_covered"] / s.n_regions,
                    color=PLOT_COLOURS[key], alpha=0.18, linewidth=1.0)
        # mean across seasons
        m = (byweek.groupby("season_week_idx")
                   .apply(lambda g: 100 * g[f"{key}_n_covered"].sum() / g.n_regions.sum(),
                          include_groups=False))
        ax.plot(m.index, m.values, color=PLOT_COLOURS[key],
                linewidth=2.4, label=label, marker="o", markersize=3)

    ax.set_xlabel("Position in season (MMWR week)")
    ax.set_ylabel("Region-weeks with genomic data (%)")
    ax.set_title("Publication delay relocates genomic coverage to the late season",
                 fontsize=12, loc="left")
    ticks = [i for i in sorted(idx_to_week) if i % 4 == 0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{idx_to_week[i]:02d}" for i in ticks])
    ax.set_ylim(-3, 103)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()
    out = FIGURES / "fig_coverage_crossover.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_vs_peak(byoffset):
    """Coverage against weeks from the season's epidemic peak - headline figure."""
    fig, ax = plt.subplots(figsize=(9, 5))
    keep = byoffset[byoffset["n_rows"] >= 10]      # drop thin tails
    for key, _, label in VARIANTS:
        ax.plot(keep.weeks_from_peak, keep[f"{key}_pct"],
                color=PLOT_COLOURS[key], linewidth=2.4, marker="o",
                markersize=3.5, label=label)
    ax.axvline(0, color="#444444", linestyle="--", linewidth=1.2)
    ax.annotate("epidemic peak", xy=(0, 8), xytext=(1.2, 8),
                fontsize=9, color="#444444", va="center")
    ax.axvspan(keep.weeks_from_peak.min(), 0, color="#000000", alpha=0.04)
    ax.set_xlabel("Weeks from epidemic peak  (negative = before the peak)")
    ax.set_ylabel("Region-weeks with genomic data (%)")
    ax.set_title("Genomic data is largely absent before the peak under realistic availability",
                 fontsize=12, loc="left")
    ax.set_ylim(-3, 103)
    ax.legend(frameon=False, fontsize=9, loc="center left")
    _style(ax)
    fig.tight_layout()
    out = FIGURES / "fig_coverage_vs_peak.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_onset(byweek, peaks):
    """Per-season week-by-week covered-region counts, as-of vs oracle."""
    fig, axes = plt.subplots(1, len(SEASONS), figsize=(13, 4), sharey=True)
    if len(SEASONS) == 1:
        axes = [axes]
    for ax, season in zip(axes, SEASONS):
        s = byweek[byweek["season"] == season].sort_values("season_week_idx")
        if s.empty:
            continue
        ax.bar(s.season_week_idx, s.oracle_n_covered, width=0.86,
               color=PLOT_COLOURS["oracle"], alpha=0.35, label="Oracle")
        ax.bar(s.season_week_idx, s.asof_L8_16_n_covered, width=0.52,
               color=PLOT_COLOURS["asof_L8_16"], label="As-of 8-16 wk")
        pk = peaks[season]["peak_idx"]
        ax.axvline(pk, color="#444444", linestyle="--", linewidth=1.1)

        zero = s[s.asof_L8_16_n_covered == 0]
        if not zero.empty:
            ax.axvspan(zero.season_week_idx.min() - 0.5,
                       zero.season_week_idx.max() + 0.5,
                       color="#c0504d", alpha=0.08)
            ax.text(zero.season_week_idx.min(), 10.4,
                    f"{len(zero)} wks\nno data", fontsize=8, color="#c0504d",
                    va="top")
        ax.set_title(f"{season}   (peak wk {peaks[season]['peak_week']:02d})",
                     fontsize=10, loc="left")
        ax.set_xlabel("Season week index")
        _style(ax)
    axes[0].set_ylabel("Regions with genomic data (of 10)")
    axes[0].set_ylim(0, 11)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Epidemic onset has no usable genomic data in any season",
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    out = FIGURES / "fig_onset_gap.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_delay(d):
    fig, ax = plt.subplots(figsize=(9, 5))
    v = d["delay_days"].values
    ax.hist(v, bins=np.arange(0, min(v.max(), 400) + 7, 7),
            color="#1f4e79", alpha=0.75, edgecolor="white", linewidth=0.4)
    med = float(np.median(v))
    for q, lab, ls in [(np.percentile(v, 10), "p10", ":"),
                       (med, f"median {med:.0f} d ({med/7:.1f} wk)", "-"),
                       (np.percentile(v, 90), "p90", ":")]:
        ax.axvline(q, color="#c0504d", linestyle=ls, linewidth=1.6)
    ax.annotate(f"median {med:.0f} days ({med/7:.1f} weeks)",
                xy=(med, ax.get_ylim()[1] * 0.92),
                xytext=(med + 20, ax.get_ylim()[1] * 0.92),
                color="#c0504d", fontsize=10, va="center")
    ax.set_xlabel("Collection to public submission (days)")
    ax.set_ylabel("Sequences")
    ax.set_title("Collection-to-publication delay, influenza A HA, USA",
                 fontsize=12, loc="left")
    _style(ax)
    fig.tight_layout()
    out = FIGURES / "fig_delay_distribution.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# ──────────────────────────────────────────────────────────────────────

def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    print("EpiSurge E1 - delay and coverage-timing analysis")
    print("=" * 66)

    core = load_master()
    core, peaks = add_peak_offset(core)
    print(f"\nCore-season rows: {len(core)}  "
          f"({core.season.nunique()} seasons x {core.hhs_region.nunique()} regions)")

    print("\nEpidemic peak per season (national mean wILI):")
    for s in SEASONS:
        p = peaks[s]
        print(f"  {s}: MMWR week {p['peak_week']:02d} "
              f"(epiweek {p['peak_epiweek']}), peak wILI {p['peak_wili']:.2f}")

    phase = coverage_by_phase(core)
    side = coverage_by_peak_side(core)
    byweek = coverage_by_week(core)
    byoffset = coverage_by_peak_offset(core)

    # ── console summary (this is the block to share) ──────────────────
    print("\n" + "-" * 66)
    print("TABLE 1  Coverage by calendar phase (%)")
    print("-" * 66)
    piv = phase.pivot_table(index=["variant", "phase"], columns="season",
                            values="pct_covered")
    print(piv.to_string())

    print("\n" + "-" * 66)
    print("TABLE 2  Coverage before vs after the epidemic peak (%)")
    print("-" * 66)
    piv2 = side.pivot_table(index=["variant", "peak_side"], columns="season",
                            values="pct_covered")
    print(piv2.to_string())

    print("\n" + "-" * 66)
    print("TABLE 3  Consecutive onset weeks with ZERO as-of genomic coverage")
    print("-" * 66)
    for s in SEASONS:
        sw = byweek[byweek.season == s].sort_values("season_week_idx")
        zero = sw[sw.asof_L8_16_n_covered == 0]
        if zero.empty:
            print(f"  {s}: none")
            continue
        wks = zero.week.tolist()
        last_off = int(zero.weeks_from_peak.max())
        # weeks_from_peak is negative before the peak and positive after, so
        # the sign has to be read out explicitly or the sentence inverts
        when = (f"{abs(last_off)} wk before the peak" if last_off < 0
                else "at the peak" if last_off == 0
                else f"{last_off} wk after the peak")
        print(f"  {s}: {len(zero)} weeks  (MMWR wk "
              f"{wks[0]:02d}-{wks[-1]:02d})   "
              f"peak is wk {peaks[s]['peak_week']:02d}; "
              f"last zero-coverage week falls {when}")

    d, dsum = delay_summary()
    if dsum is not None:
        print("\n" + "-" * 66)
        print("TABLE 4  Collection-to-submission delay (days)")
        print("-" * 66)
        print(dsum.to_string(index=False))

    # ── write tables ─────────────────────────────────────────────────
    phase.to_csv(REPORTS / "coverage_by_phase.csv", index=False)
    side.to_csv(REPORTS / "coverage_by_peak_side.csv", index=False)
    byweek.to_csv(REPORTS / "coverage_by_week.csv", index=False)
    byoffset.to_csv(REPORTS / "coverage_by_peak_offset.csv", index=False)
    if dsum is not None:
        dsum.to_csv(REPORTS / "delay_summary.csv", index=False)

    # ── figures ──────────────────────────────────────────────────────
    print("\n" + "-" * 66)
    print("Figures")
    print("-" * 66)
    made = [fig_crossover(byweek, core), fig_vs_peak(byoffset),
            fig_onset(byweek, peaks)]
    if d is not None:
        made.append(fig_delay(d))
    for m in made:
        print(f"  wrote {m}")

    # ── self-checks ──────────────────────────────────────────────────
    print("\n" + "-" * 66)
    print("Checks")
    print("-" * 66)
    fails = []

    if len(core) != 990:
        fails.append(f"expected 990 core rows, got {len(core)}")

    oracle_all = 100 * core.has_genomic_oracle.mean()
    a816 = 100 * core.has_genomic_asof_L8_16.mean()
    a412 = 100 * core.has_genomic_asof_L4_12.mean()
    print(f"  core-season coverage: oracle {oracle_all:.1f}%  "
          f"as-of L8-16 {a816:.1f}%  as-of L4-12 {a412:.1f}%")
    for name, got, want in [("oracle", oracle_all, 73.9),
                            ("as-of L8-16", a816, 60.3),
                            ("as-of L4-12", a412, 55.8)]:
        if abs(got - want) > 0.15:
            fails.append(f"{name} coverage {got:.1f}% != logged {want}%")

    early = core[core.phase == EARLY_LABEL]
    late = core[core.phase == LATE_LABEL]
    o_gap = 100 * early.has_genomic_oracle.mean() - 100 * late.has_genomic_oracle.mean()
    a_gap = 100 * late.has_genomic_asof_L8_16.mean() - 100 * early.has_genomic_asof_L8_16.mean()
    print(f"  oracle early-minus-late:  {o_gap:+.1f} pp")
    print(f"  as-of  late-minus-early:  {a_gap:+.1f} pp")
    if o_gap <= 0 or a_gap <= 0:
        fails.append("crossover not present - the central claim does not hold on this data")

    if fails:
        print("\n  FAILURES:")
        for f in fails:
            print(f"    [FAIL] {f}")
        sys.exit(1)
    print("\n  All checks passed.")
    print("\nE1 complete. Descriptive only - says nothing about forecast skill.")


if __name__ == "__main__":
    main()