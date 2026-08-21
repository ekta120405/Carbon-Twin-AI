"""
src/layer1_data/acquisition.py

Task 4, Step 3: NASA POWER hourly extraction.

Fetches independent weather data for BOTH candidate plant-location hypotheses
(Nasik and Gandikota — see config/plants.yaml), each expanded into a 3x3 grid
of neighbouring 1-degree cells, matching the CERES SYN1deg spatial resolution.
This directly produces the inputs for the Step 4 sensitivity analysis, which
decides (a) which candidate maps to which plant, and (b) how sensitive twin
accuracy is to the coordinate assumption.

Docs:
    https://power.larc.nasa.gov/docs/services/api/
    https://power.larc.nasa.gov/docs/services/api/temporal/hourly/
    https://power.larc.nasa.gov/parameters/

Usage:
    python -m src.layer1_data.acquisition --config config/plants.yaml
    python -m src.layer1_data.acquisition --config config/plants.yaml --site nasik
    python -m src.layer1_data.acquisition --config config/plants.yaml --dry-run

Output:
    data/raw/nasa_power/<site>_<lat>_<lon>.parquet   (UTC-indexed hourly)
    data/raw/nasa_power/manifest.json                (what was fetched, when)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import yaml

BASE_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"

PARAMETERS = [
    "ALLSKY_SFC_SW_DWN",
    "CLRSKY_SFC_SW_DWN",
    "ALLSKY_SFC_SW_DNI",
    "ALLSKY_SFC_SW_DIFF",
    "ALLSKY_KT",
    "T2M",
    "RH2M",
    "WS2M",
    "PS",
    "PRECTOTCORR",
]
assert len(PARAMETERS) <= 15, "Hourly API allows at most 15 parameters per call"

# Kaggle dataset window is 2020-05-15 to 2020-06-17. Pad by 3 days each side
# so downstream feature engineering (lags, clear-sky index smoothing) never
# runs out of context at the edges.
START_DATE = "20200512"
END_DATE = "20200620"

GRID_OFFSETS = [-1.0, 0.0, 1.0]   # degrees; matches CERES SYN1deg cell size
MAX_RETRIES = 4
RETRY_BACKOFF_S = 5
REQUEST_TIMEOUT_S = 90


@dataclass
class Cell:
    site: str          # candidate site name, e.g. "nasik"
    lat: float
    lon: float
    dlat: int           # grid offset index, for filenames/logging
    dlon: int


def build_grid(site: str, center_lat: float, center_lon: float) -> list[Cell]:
    cells = []
    for i, dlat in enumerate(GRID_OFFSETS):
        for j, dlon in enumerate(GRID_OFFSETS):
            cells.append(
                Cell(
                    site=site,
                    lat=round(center_lat + dlat, 4),
                    lon=round(center_lon + dlon, 4),
                    dlat=i - 1,
                    dlon=j - 1,
                )
            )
    return cells


def cache_path(cell: Cell, out_dir: Path) -> Path:
    tag = f"{cell.site}_{cell.lat:+.2f}_{cell.lon:+.2f}".replace("+", "p").replace("-", "m").replace(".", "_")
    return out_dir / f"{tag}.parquet"


def fetch_cell(cell: Cell) -> pd.DataFrame:
    """One NASA POWER hourly request, UTC explicit, with retry/backoff."""
    params = {
        "parameters": ",".join(PARAMETERS),
        "community": "RE",              # Renewable Energy community
        "latitude": cell.lat,
        "longitude": cell.lon,
        "start": START_DATE,
        "end": END_DATE,
        "format": "JSON",
        "time-standard": "UTC",         # NEVER omit — hourly defaults to Local Solar Time
    }

    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT_S)
            r.raise_for_status()
            payload = r.json()
            break
        except (requests.RequestException, ValueError) as e:
            last_err = e
            wait = RETRY_BACKOFF_S * attempt
            print(f"    attempt {attempt}/{MAX_RETRIES} failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    else:
        raise RuntimeError(f"NASA POWER fetch failed for {cell} after {MAX_RETRIES} attempts") from last_err

    param_data = payload["properties"]["parameter"]

    # NASA POWER returns {param: {timestamp_str: value, ...}, ...}
    # timestamp_str is YYYYMMDDHH in UTC (because we forced time-standard=UTC).
    df = pd.DataFrame(param_data)
    df.index = pd.to_datetime(df.index, format="%Y%m%d%H", utc=True)
    df.index.name = "timestamp_utc"
    df = df.sort_index()

    # NASA POWER uses -999 as a fill value for missing observations.
    df = df.replace(-999.0, pd.NA)

    df["site"] = cell.site
    df["lat"] = cell.lat
    df["lon"] = cell.lon
    df["dlat"] = cell.dlat
    df["dlon"] = cell.dlon

    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/plants.yaml")
    ap.add_argument("--out-dir", default="data/raw/nasa_power")
    ap.add_argument("--site", default=None, help="Fetch only this candidate site")
    ap.add_argument("--dry-run", action="store_true", help="Print the plan, fetch nothing")
    ap.add_argument("--force", action="store_true", help="Re-fetch even if cached")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # These are the two candidate sites from Task 4 Step 2 (§4.3). If your
    # satellite-refined coordinates differ from the city centroids, edit here.
    candidates = {
       "nasik":     (20.0024, 73.7945),
       "gandikota": (14.9164, 78.2919),
    }
    if args.site:
        candidates = {args.site: candidates[args.site]}

    all_cells: list[Cell] = []
    for site, (lat, lon) in candidates.items():
        all_cells.extend(build_grid(site, lat, lon))

    print(f"Plan: {len(all_cells)} cells across {len(candidates)} candidate site(s)")
    print(f"Window: {START_DATE} to {END_DATE} (UTC), {len(PARAMETERS)} parameters")
    for c in all_cells:
        print(f"  {c.site:10s}  ({c.lat:+.2f}, {c.lon:+.2f})  offset ({c.dlat:+d},{c.dlon:+d})"
              f"{'  [CENTER]' if c.dlat == 0 and c.dlon == 0 else ''}")

    if args.dry_run:
        print("\n--dry-run: no requests made")
        return 0

    manifest = {"fetched_at_utc": datetime.utcnow().isoformat(), "cells": []}

    for i, cell in enumerate(all_cells, 1):
        path = cache_path(cell, out_dir)
        print(f"\n[{i}/{len(all_cells)}] {cell.site} ({cell.lat:+.2f}, {cell.lon:+.2f})")

        if path.exists() and not args.force:
            print(f"    cached -> {path.name} (use --force to re-fetch)")
            manifest["cells"].append({"site": cell.site, "lat": cell.lat, "lon": cell.lon,
                                       "path": str(path), "cached": True})
            continue

        df = fetch_cell(cell)
        df.to_parquet(path)
        n_missing = int(df[PARAMETERS].isna().sum().sum())
        print(f"    fetched {len(df)} hourly rows, {n_missing} missing values -> {path.name}")
        manifest["cells"].append({"site": cell.site, "lat": cell.lat, "lon": cell.lon,
                                   "path": str(path), "cached": False, "n_rows": len(df),
                                   "n_missing": n_missing})

        time.sleep(1.0)  # be a polite API citizen between requests

    with open(out_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    print(f"\nDone. {len(all_cells)} cells written to {out_dir}/")
    print("Next: python -m src.layer1_data.coordinate_sensitivity")
    return 0


if __name__ == "__main__":
    sys.exit(main())
