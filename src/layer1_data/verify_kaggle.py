"""
src/layer1_data/verify_kaggle.py

Task 4, integrity check. Run immediately after downloading the Kaggle CSVs
(§4.2) and before touching NASA POWER. Confirms the files match the schema
this whole pipeline assumes.

Usage:
    python -m src.layer1_data.verify_kaggle
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

KAGGLE_DIR = Path("data/raw/kaggle")

EXPECTED_GEN_COLS = {"DATE_TIME", "PLANT_ID", "SOURCE_KEY", "DC_POWER", "AC_POWER",
                      "DAILY_YIELD", "TOTAL_YIELD"}
EXPECTED_WEATHER_COLS = {"DATE_TIME", "PLANT_ID", "SOURCE_KEY",
                          "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"}


def parse_datetime(series: pd.Series) -> pd.Series:
    for fmt in ("%d-%m-%Y %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return pd.to_datetime(series, format=fmt)
        except ValueError:
            continue
    return pd.to_datetime(series)  # last resort, let pandas infer


def check_generation(plant: int) -> bool:
    path = KAGGLE_DIR / f"Plant_{plant}_Generation_Data.csv"
    ok = True
    print(f"\n--- Plant {plant} generation: {path.name} ---")
    if not path.exists():
        print("  MISSING FILE"); return False

    df = pd.read_csv(path)
    missing_cols = EXPECTED_GEN_COLS - set(df.columns)
    if missing_cols:
        print(f"  FAIL: missing columns {missing_cols}"); ok = False

    n_inverters = df["SOURCE_KEY"].nunique()
    print(f"  rows: {len(df)}  |  inverters: {n_inverters}")
    if n_inverters != 22:
        print(f"  WARNING: expected 22 inverters, found {n_inverters}")

    ts = parse_datetime(df["DATE_TIME"])
    span_days = (ts.max() - ts.min()).days
    print(f"  date range: {ts.min()} to {ts.max()}  (~{span_days} days)")
    if not (30 <= span_days <= 40):
        print(f"  WARNING: expected ~34 days, got {span_days}")

    neg = (df["AC_POWER"] < 0).sum()
    if neg:
        print(f"  NOTE: {neg} negative AC_POWER rows (expected — clip in Layer 1.4)")

    null_frac = df[["DC_POWER", "AC_POWER"]].isna().mean().mean()
    print(f"  null fraction (DC/AC power): {null_frac:.4f}")

    return ok


def check_weather(plant: int) -> bool:
    path = KAGGLE_DIR / f"Plant_{plant}_Weather_Sensor_Data.csv"
    ok = True
    print(f"\n--- Plant {plant} weather: {path.name} ---")
    if not path.exists():
        print("  MISSING FILE"); return False

    df = pd.read_csv(path)
    missing_cols = EXPECTED_WEATHER_COLS - set(df.columns)
    if missing_cols:
        print(f"  FAIL: missing columns {missing_cols}"); ok = False

    n_sensors = df["SOURCE_KEY"].nunique()
    print(f"  rows: {len(df)}  |  weather sources: {n_sensors} (expect 1)")

    print(f"  AMBIENT_TEMPERATURE range: {df['AMBIENT_TEMPERATURE'].min():.1f} "
          f"to {df['AMBIENT_TEMPERATURE'].max():.1f}")
    print(f"  MODULE_TEMPERATURE range:  {df['MODULE_TEMPERATURE'].min():.1f} "
          f"to {df['MODULE_TEMPERATURE'].max():.1f}")
    print(f"  IRRADIATION range:         {df['IRRADIATION'].min():.4f} "
          f"to {df['IRRADIATION'].max():.4f}")

    if df["MODULE_TEMPERATURE"].max() < df["AMBIENT_TEMPERATURE"].max():
        print("  WARNING: module temp should exceed ambient temp under sun — check units/parse")

    return ok


def main() -> int:
    if not KAGGLE_DIR.exists():
        print(f"ERROR: {KAGGLE_DIR} does not exist. Run the Kaggle download first (§4.2).")
        return 1

    results = []
    for plant in (1, 2):
        results.append(check_generation(plant))
        results.append(check_weather(plant))

    print("\n" + "=" * 50)
    if all(results):
        print("ALL CHECKS PASSED — proceed to Step 2 (coordinate resolution)")
        return 0
    print("SOME CHECKS FAILED — resolve before proceeding")
    return 1


if __name__ == "__main__":
    sys.exit(main())
