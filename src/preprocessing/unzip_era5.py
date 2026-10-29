#!/usr/bin/env python3
"""
EpiSurge - Unwrap ERA5 downloads.

CDS returns a ZIP when a request contains multiple variables, even if
the .nc extension suggests otherwise. This extracts the real NetCDF
files and renames them to match their source request.
"""

import shutil
import zipfile
from pathlib import Path

RAW_DIR = Path("data/raw/era5")
EXTRACT_DIR = RAW_DIR / "extracted"
EXTRACT_DIR.mkdir(exist_ok=True)


def main():
    files = sorted(RAW_DIR.glob("era5_daily_*.nc"))
    print(f"Found {len(files)} downloaded files\n")

    for f in files:
        with open(f, "rb") as fh:
            magic = fh.read(4)

        if magic != b"PK\x03\x04":
            print(f"{f.name}: not a ZIP, skipping")
            continue

        tag = f.stem.replace("era5_daily_", "")   # 2019, 2023a, ...

        with zipfile.ZipFile(f) as z:
            members = [m for m in z.namelist() if m.endswith(".nc")]
            print(f"{f.name}: contains {len(members)} NetCDF file(s)")

            for m in members:
                # CDS names members by variable, e.g. 2m_temperature_0_daily-mean.nc
                var_part = Path(m).stem
                out_name = f"era5_{tag}_{var_part}.nc"
                out_path = EXTRACT_DIR / out_name

                with z.open(m) as src, open(out_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)

                print(f"    -> {out_name} ({out_path.stat().st_size/1e6:.1f} MB)")

    extracted = sorted(EXTRACT_DIR.glob("*.nc"))
    print(f"\nExtracted {len(extracted)} NetCDF files to {EXTRACT_DIR}")


if __name__ == "__main__":
    main()