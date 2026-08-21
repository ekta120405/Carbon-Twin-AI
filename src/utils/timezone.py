"""
src/utils/timezone.py

Layer 1.2: the timezone contract (Overview v2 SS4.3).

Single source of truth. Every module that touches a raw timestamp imports
from here rather than re-implementing localisation -- that's what prevents
the 5.5-hour IST/UTC misalignment bug from creeping back in somewhere.
"""
from __future__ import annotations

import pandas as pd

SOURCE_TZ = "Asia/Kolkata"   # Kaggle timestamps are naive local time
PIPELINE_TZ = "UTC"          # everything downstream of this module is UTC
SOLAR_NOON_TOLERANCE_MIN = 90


def parse_kaggle_datetime(series: pd.Series) -> pd.Series:
    """Parse Kaggle's DATE_TIME column, handling both known formats."""
    for fmt in ("%d-%m-%Y %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return pd.to_datetime(series, format=fmt)
        except ValueError:
            continue
    return pd.to_datetime(series)  # last-resort inference; log a warning upstream


def to_utc(naive_ist: pd.Series) -> pd.Series:
    """Localise naive IST timestamps to tz-aware UTC. Apply exactly once per source."""
    return naive_ist.dt.tz_localize(SOURCE_TZ).dt.tz_convert(PIPELINE_TZ)


def assert_solar_noon_sane(power_series: pd.Series, longitude: float,
                            tolerance_min: int = SOLAR_NOON_TOLERANCE_MIN) -> None:
    """
    Validation gate: the empirical daily generation peak must fall within
    `tolerance_min` of computed solar noon, or the pipeline aborts. This is
    what catches a silent timezone bug before it contaminates every layer
    downstream.
    """
    daily_peak_hour_utc = (
        power_series.groupby(power_series.index.date)
        .apply(lambda s: s.idxmax().hour + s.idxmax().minute / 60)
    )
    mean_peak_utc = daily_peak_hour_utc.mean()

    # Solar noon (UTC) approx = 12:00 local solar time - longitude/15 offset from UTC.
    # For longitude in degrees East, local solar noon in UTC hours:
    expected_solar_noon_utc = 12.0 - (longitude / 15.0)

    diff_min = abs(mean_peak_utc - expected_solar_noon_utc) * 60
    assert diff_min <= tolerance_min, (
        f"Solar-noon sanity check FAILED: mean generation peak at "
        f"{mean_peak_utc:.2f}h UTC, expected ~{expected_solar_noon_utc:.2f}h UTC "
        f"(longitude {longitude}), difference {diff_min:.0f} min exceeds "
        f"{tolerance_min} min tolerance. Check timezone localisation before proceeding."
    )
