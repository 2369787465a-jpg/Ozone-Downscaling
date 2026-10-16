"""Check required input files without importing the scientific Python stack."""

import os
from pathlib import Path
import sys


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = Path(os.environ.get("OZONE_DATA_DIR", project_root / "data")).expanduser().resolve()
    required = ["daily_mean.nc", "regrided.nc", "gba_pollutants_jan2020_complete.csv"]
    optional = ["humidity_regrid.nc", "radiation_regrid.nc"]
    print(f"Data directory: {data_dir}")
    missing = []
    for name in required:
        path = data_dir / name
        present = path.is_file() and path.stat().st_size > 0
        print(f"{'OK     ' if present else 'MISSING'} required: {name}")
        if not present:
            missing.append(name)
    for name in optional:
        path = data_dir / name
        present = path.is_file() and path.stat().st_size > 0
        print(f"{'OK     ' if present else 'ABSENT '} optional: {name}")
    print("This checks file presence only; validate variables, coordinates and units in data/README.md.")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
