#!/usr/bin/env python3
"""
EpiSurge - Phase 1 gate: validate every data source individually AND
verify cross-source consistency before the master table join.

Within-source bugs are usually loud. Cross-source bugs are silent and
destroy the join - mismatched region IDs, epiweek formats that look
compatible but aren't, date ranges that don't overlap.

Run this before building anything downstream.
"""

import csv
import json
import sys
from pathlib import Path
from collections import Counter

import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from config import SEASON_LABELS, SEASON_WINDOWS

PROCESSED = Path("data/processed")
RAW = Path("data/raw")

failures, warnings, notes = [], [], []

def fail(msg): failures.append(msg)
def warn(msg): warnings.append(msg)
def note(msg): notes.append(msg)

def section(title):
    print(f"\n{'='*64}\n{title}\n{'='*64}")


# ─────────────────────────────────────────────────────────────
# 1. GENOMIC
# ─────────────────────────────────────────────────────────────
def check_genomic():
    section("1. GENOMIC (NCBI)")
    seasons = SEASON_LABELS
    all_records, counts = [], {}

    for s in seasons:
        p = PROCESSED / f"mapped_{s}.json"
        if not p.exists():
            fail(f"Missing {p}")
            continue
        with open(p) as f:
            recs = json.load(f)
        counts[s] = len(recs)
        all_records.extend(recs)

    print(f"  Records per season: {counts}")
    print(f"  Total: {len(all_records)}")

    if not all_records:
        fail("No genomic records loaded")
        return []

    regions = Counter(r["hhs_region"] for r in all_records)
    print(f"  Regions present: {sorted(regions)}")
    if set(regions) != set(range(1, 11)):
        warn(f"Genomic data missing regions: {set(range(1,11)) - set(regions)}")

    print(f"  Sequences per region: "
          f"min={min(regions.values())} max={max(regions.values())}")

    # Valid MMWR years are derived from the configured season windows, not
    # hardcoded. A fixed 2019-2024 range failed 9,408 legitimate records the
    # moment earlier seasons were added.
    yr_lo = min(a for a, _ in SEASON_WINDOWS) // 100
    yr_hi = max(b for _, b in SEASON_WINDOWS) // 100
    bad_weeks = [r for r in all_records
                 if not (1 <= r["mmwr_week"] <= 53)
                 or not (yr_lo <= r["mmwr_year"] <= yr_hi)]
    if bad_weeks:
        fail(f"{len(bad_weeks)} genomic records have epiweeks outside "
             f"{yr_lo}-{yr_hi}")

    # Publication windows overlap by 24 months, so an accession can appear in
    # two season files. The composition builders deduplicate; reported here so
    # the count is visible rather than silently differing.
    accs = [r.get("accession") for r in all_records if r.get("accession")]
    dupes = len(accs) - len(set(accs))
    if dupes:
        print(f"  Duplicate accessions across overlapping fetch windows: {dupes} "
              f"(deduplicated downstream)")

    years = sorted({r["mmwr_year"] for r in all_records})
    print(f"  MMWR years: {years}")

    sub_path = PROCESSED / "submission_dates.json"
    if sub_path.exists():
        with open(sub_path) as f:
            subs = json.load(f)
        missing = [r["accession"] for r in all_records
                   if not subs.get(r["accession"], "").strip()]
        print(f"  Submission dates: {len(subs)} entries, {len(missing)} missing")
        if missing:
            fail(f"{len(missing)} records lack submission dates (breaks N5)")
    else:
        fail("submission_dates.json not found")

    return all_records


# ─────────────────────────────────────────────────────────────
# 2. FLUVIEW
# ─────────────────────────────────────────────────────────────
def check_fluview():
    section("2. FLUVIEW (CDC via Delphi)")
    p = RAW / "fluview" / "fluview.csv"
    if not p.exists():
        fail(f"Missing {p}")
        return []

    with open(p) as f:
        rows = list(csv.DictReader(f))

    print(f"  Rows: {len(rows)}")

    regions = sorted({r["region"] for r in rows})
    print(f"  Regions: {len(regions)}  {regions}")
    if len(regions) != 10:
        fail(f"Expected 10 FluView regions, got {len(regions)}")

    weeks = sorted({int(r["epiweek"]) for r in rows})
    print(f"  Epiweeks: {len(weeks)}  ({weeks[0]} to {weeks[-1]})")

    expected = len(regions) * len(weeks)
    if len(rows) != expected:
        warn(f"Row count {len(rows)} != regions*weeks {expected} - incomplete panel")

    wili_vals = []
    for r in rows:
        try:
            wili_vals.append(float(r["wili"]))
        except (ValueError, TypeError):
            pass
    if wili_vals:
        print(f"  wILI range: {min(wili_vals):.3f} to {max(wili_vals):.3f}")
        if max(wili_vals) > 30:
            warn(f"wILI max {max(wili_vals):.1f}% is implausibly high")
        if min(wili_vals) < 0:
            fail("Negative wILI values present")
    missing_wili = len(rows) - len(wili_vals)
    if missing_wili:
        warn(f"{missing_wili} rows have unparseable wili")

    return rows


# ─────────────────────────────────────────────────────────────
# 3. CENSUS
# ─────────────────────────────────────────────────────────────
def check_census():
    section("3. CENSUS")
    p = RAW / "census" / "hhs_region_population.csv"
    if not p.exists():
        fail(f"Missing {p}")
        return []

    with open(p) as f:
        rows = list(csv.DictReader(f))

    ids = sorted(int(r["hhs_region"]) for r in rows)
    total = sum(int(r["population"]) for r in rows)
    print(f"  Regions: {ids}")
    print(f"  Total population: {total:,}")

    if ids != list(range(1, 11)):
        fail(f"Census regions {ids} != 1-10")
    if not (320_000_000 < total < 345_000_000):
        warn(f"Total population {total:,} outside plausible US range")

    return rows


# ─────────────────────────────────────────────────────────────
# 4. ERA5
# ─────────────────────────────────────────────────────────────
def check_era5():
    section("4. ERA5")
    try:
        import xarray as xr
    except ImportError:
        warn("xarray not installed - skipping ERA5 content checks")
        return None

    files = sorted((RAW / "era5" / "extracted").glob("*.nc"))
    print(f"  Files: {len(files)}")
    # Expected count follows from the configured seasons: each season needs
    # H2 of its start year and H1 of its end year, two variables each.
    halves = set()
    for a, b in SEASON_WINDOWS:
        halves.add((a // 100, "H2"))
        halves.add((b // 100, "H1"))
    expected_files = len(halves) * 2
    # Only a SHORTFALL matters. Extra files are leftovers from seasons since
    # excluded (2015-16) and are simply not read by era5_to_region_week.py,
    # which selects periods by the configured season windows.
    if len(files) < expected_files:
        warn(f"Expected at least {expected_files} extracted ERA5 files "
             f"({len(halves)} half-years x 2 variables), found {len(files)}")
    elif len(files) > expected_files:
        print(f"  ({len(files) - expected_files} extra files present from "
              f"excluded seasons - not read)")

    date_ranges = []
    for f in files:
        try:
            ds = xr.open_dataset(f)
        except Exception as e:
            fail(f"ERA5 {f.name} will not open: {e}")
            continue

        tdim = "valid_time" if "valid_time" in ds.dims else "time"
        t = ds[tdim].values
        start, end = str(t[0])[:10], str(t[-1])[:10]
        date_ranges.append((start, end, f.name))

        vars_present = set(ds.data_vars)
        # After unzipping, CDS splits variables into separate files -
        # each file holds exactly one of t2m or d2m, not both.
        found = {"t2m", "d2m"} & vars_present
        if not found:
            fail(f"ERA5 {f.name} has neither t2m nor d2m - has {sorted(vars_present)}")

        for v in found:
            if bool(ds[v].isnull().all()):
                fail(f"ERA5 {f.name}: {v} is entirely NaN")

        print(f"    {f.name}: {start} to {end}, {ds.sizes.get(tdim)} days")
        ds.close()

    # verify both variables are present for every period
    periods = {}
    for f in files:
        tag = f.stem.split("_")[1]          # era5_2023a_... -> 2023a
        var = "t2m" if "2m_temperature" in f.stem else "d2m"
        periods.setdefault(tag, set()).add(var)

    print(f"\n  Variable coverage by period:")
    for tag in sorted(periods):
        have = periods[tag]
        status = "OK" if have == {"t2m", "d2m"} else f"INCOMPLETE - only {sorted(have)}"
        print(f"    {tag}: {status}")
        if have != {"t2m", "d2m"}:
            fail(f"ERA5 period {tag} missing a variable: has {sorted(have)}")

    return date_ranges


# ─────────────────────────────────────────────────────────────
# 5. GADM
# ─────────────────────────────────────────────────────────────
def check_gadm():
    section("5. GADM")
    p = RAW / "gadm" / "hhs_regions.geojson"
    if not p.exists():
        fail(f"Missing {p}")
        return None

    try:
        import geopandas as gpd
    except ImportError:
        warn("geopandas not installed - skipping GADM content checks")
        return None

    gdf = gpd.read_file(p)
    ids = sorted(gdf["hhs_region"].astype(int))
    print(f"  Regions: {ids}")
    print(f"  CRS: {gdf.crs}")

    if ids != list(range(1, 11)):
        fail(f"GADM regions {ids} != 1-10")

    invalid = gdf[~gdf.geometry.is_valid]
    if len(invalid):
        warn(f"{len(invalid)} GADM geometries are topologically invalid")

    note("Region 10 spans the antimeridian (Alaska) and Region 9 extends "
         "to lat 18.9 (Hawaii) - both excluded from the ERA5 CONUS box.")

    return gdf


# ─────────────────────────────────────────────────────────────
# 6. CROSS-SOURCE CONSISTENCY
# ─────────────────────────────────────────────────────────────
def check_cross_source(genomic, fluview, census, era5_ranges):
    section("6. CROSS-SOURCE CONSISTENCY")

    if not (genomic and fluview and census):
        warn("Skipping cross-source checks - a source failed to load")
        return

    gen_regions = {int(r["hhs_region"]) for r in genomic}
    flu_regions = {int(r["region"].replace("hhs", "")) for r in fluview}
    cen_regions = {int(r["hhs_region"]) for r in census}

    print(f"  Region IDs - genomic: {sorted(gen_regions)}")
    print(f"               fluview: {sorted(flu_regions)}")
    print(f"               census:  {sorted(cen_regions)}")

    if not (gen_regions <= flu_regions):
        fail(f"Genomic regions not covered by FluView: {gen_regions - flu_regions}")
    if flu_regions != cen_regions:
        fail(f"FluView/Census region mismatch: {flu_regions ^ cen_regions}")

    gen_weeks = {r["mmwr_year"] * 100 + r["mmwr_week"] for r in genomic}
    flu_weeks = {int(r["epiweek"]) for r in fluview}

    print(f"\n  Genomic epiweek range: {min(gen_weeks)} to {max(gen_weeks)}")
    print(f"  FluView epiweek range: {min(flu_weeks)} to {max(flu_weeks)}")

    overlap = gen_weeks & flu_weeks
    print(f"  Overlapping epiweeks: {len(overlap)} / {len(gen_weeks)} genomic weeks")

    orphans = gen_weeks - flu_weeks
    if orphans:
        warn(f"{len(orphans)} genomic epiweeks have no FluView row "
             f"(e.g. {sorted(orphans)[:5]}) - these rows will drop in the join")
    if not overlap:
        fail("ZERO epiweek overlap between genomic and FluView - "
             "formats are incompatible, join will produce nothing")

    gen_cells = {(int(r["hhs_region"]), r["mmwr_year"] * 100 + r["mmwr_week"])
                 for r in genomic}
    flu_cells = {(int(r["region"].replace("hhs", "")), int(r["epiweek"]))
                 for r in fluview}
    joinable = gen_cells & flu_cells

    print(f"\n  Genomic (region, week) cells: {len(gen_cells)}")
    print(f"  FluView (region, week) cells: {len(flu_cells)}")
    print(f"  Joinable cells:               {len(joinable)}")
    print(f"  Genomic coverage of FluView:  "
          f"{100*len(joinable)/len(flu_cells):.1f}%")

    if len(joinable) < 300:
        warn(f"Only {len(joinable)} joinable cells - may be too sparse for modelling")

    if era5_ranges:
        era_start = min(r[0] for r in era5_ranges)
        era_end = max(r[1] for r in era5_ranges)
        print(f"\n  ERA5 date span: {era_start} to {era_end}")

        flu_year_min = min(flu_weeks) // 100
        flu_year_max = max(flu_weeks) // 100
        if int(era_start[:4]) > flu_year_min:
            note(f"ERA5 starts {era_start[:4]} but FluView has data from "
                 f"{flu_year_min} - early weeks will lack climate features")
        if int(era_end[:4]) < flu_year_max:
            note(f"ERA5 ends {era_end[:4]} but FluView extends to {flu_year_max}")


# ─────────────────────────────────────────────────────────────
def main():
    genomic = check_genomic()
    fluview = check_fluview()
    census = check_census()
    era5_ranges = check_era5()
    check_gadm()
    check_cross_source(genomic, fluview, census, era5_ranges)

    section("SUMMARY")
    if notes:
        print("Notes:")
        for m in notes:
            print(f"  [NOTE] {m}")
    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for m in warnings:
            print(f"  [WARN] {m}")
    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for m in failures:
            print(f"  [FAIL] {m}")
        print("\nFix failures before the master table join.")
        sys.exit(1)

    print("\nAll hard checks PASSED - sources are consistent and joinable.")


if __name__ == "__main__":
    main()