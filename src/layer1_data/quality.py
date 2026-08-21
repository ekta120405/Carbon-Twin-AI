"""
src/layer1_data/quality.py

Layer 1.4: physics ceiling QC (Overview v2 SS4.5).

This is how "clean data" gets defined WITHOUT circularity. We do not use
statistics on the claimed generation itself to decide what's clean (that
begs the question the whole project exists to answer). Instead we compute
the physically maximum possible AC output from independent weather + basic
system assumptions, and anything claiming more than that is definitionally
impossible -- excluded from training, flagged at maximum severity in Layer 3.

Also handles: night-time masking, stuck-sensor detection, negative power.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_cell_temperature(t2m_c: pd.Series, ws2m: pd.Series, poa_ghi: pd.Series,
                              noct_c: float = 45.0) -> pd.Series:
    """
    Simple NOCT-based cell temperature model (a lighter alternative to pvlib's
    Faiman/SAPM models -- swap in pvlib.temperature.faiman for the production
    twin if a closer physical match is needed; this is sufficient for QC).

        T_cell = T_ambient + (NOCT - 20) / 800 * GHI
    """
    return t2m_c + (noct_c - 20.0) / 800.0 * poa_ghi.clip(lower=0)


def compute_physics_ceiling(
    poa_ghi: pd.Series,
    t2m_c: pd.Series,
    ws2m: pd.Series,
    ac_nameplate_kw: float,
    gamma_pdc: float = -0.004,
    headroom: float = 1.05,
) -> pd.Series:
    """
    Physical maximum AC output. Simple single-diode-free PVWatts-style model:

        P_ceiling = P_nameplate * (GHI/1000) * (1 + gamma*(T_cell-25)) * headroom

    `headroom` (default 5%) accounts for irradiance measurement uncertainty
    and brief over-irradiance (cloud-edge enhancement) -- without it, genuine
    clear moments would be falsely excluded.
    """
    t_cell = compute_cell_temperature(t2m_c, ws2m, poa_ghi)
    temp_factor = (1.0 + gamma_pdc * (t_cell - 25.0)).clip(lower=0.5, upper=1.05)
    ceiling = ac_nameplate_kw * (poa_ghi.clip(lower=0) / 1000.0) * temp_factor * headroom
    return ceiling.clip(lower=0)


def apply_daylight_mask(df: pd.DataFrame, latitude: float, longitude: float,
                         min_elevation_deg: float = 10.0) -> pd.Series:
    """
    Boolean mask: True where solar elevation exceeds the threshold.
    Requires pvlib for solar position (see downscale.py for the same pattern).
    """
    import pvlib

    solpos = pvlib.solarposition.get_solarposition(df.index, latitude, longitude)
    elevation = 90.0 - solpos["apparent_zenith"]
    return elevation >= min_elevation_deg


def flag_stuck_sensor(series: pd.Series, run_length_threshold: int = 8) -> pd.Series:
    """
    Boolean mask: True where a value repeats for >= `run_length_threshold`
    consecutive samples. Applied only within daylight -- a flat night-time
    zero is not a sensor fault, it's correct.
    """
    same_as_prev = series.eq(series.shift())
    # run-length: count consecutive True's, reset on False
    run_id = (~same_as_prev).cumsum()
    run_len = same_as_prev.groupby(run_id).transform("size")
    return run_len >= run_length_threshold


def build_clean_mask(
    df: pd.DataFrame,
    ac_nameplate_kw: float,
    latitude: float,
    longitude: float,
    ceiling_col: str = "ac_power",
    poa_col: str = "ghi_15min",
    t2m_col: str = "t2m",
    ws2m_col: str = "ws2m",
    ceiling_tolerance: float = 1.05,
    stuck_run_length: int = 8,
    min_elevation_deg: float = 10.0,
) -> pd.DataFrame:
    """
    Combines every QC test into one output table with individual boolean
    columns PLUS a final `is_clean` = all checks passed. Individual columns
    are kept because a ceiling violation is itself a maximum-severity Layer 3
    flag, not just a training-exclusion -- Layer 3 will read this table too.
    """
    ceiling = compute_physics_ceiling(
        df[poa_col], df[t2m_col], df[ws2m_col], ac_nameplate_kw, headroom=ceiling_tolerance
    )
    is_daylight = apply_daylight_mask(df, latitude, longitude, min_elevation_deg)
    exceeds_ceiling = (df[ceiling_col] > ceiling) & is_daylight
    is_negative = df[ceiling_col] < 0
    is_stuck = flag_stuck_sensor(df[ceiling_col], stuck_run_length) & is_daylight

    is_clean = is_daylight & ~exceeds_ceiling & ~is_negative & ~is_stuck

    return pd.DataFrame({
        "physics_ceiling_kw": ceiling,
        "is_daylight": is_daylight,
        "exceeds_ceiling": exceeds_ceiling,
        "is_negative": is_negative,
        "is_stuck": is_stuck,
        "is_clean": is_clean,
    }, index=df.index)
