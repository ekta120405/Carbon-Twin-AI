"""
src/layer3_anomaly/aggregate.py

Layer 3.6: risk aggregation. Combines 3.1-3.5's outputs into a
documented, non-arbitrary project-level risk score.

FLAG-TYPE-AWARE SEVERITY -- DECISION, VERIFIED NOT ASSUMED:
"Cumulative unexplained excess energy" cannot be a single generic
computation across all flag types. 3.1's excess_energy_kwh is
DAYLIGHT-ONLY by construction (see residual.py's compute_daily_energy_residual
and flag_night_injection docstrings -- this is exactly why A3 needed a
separate guardrail in the first place: r_day is structurally blind to
night claims). Using excess_energy_kwh as the severity measure for a
flagged_night_injection day would silently sum unrelated ordinary
daytime variation, not the actual fabricated night energy. Similarly, a
flagged_stuck_sensor day's excess_energy_kwh (computed over the WHOLE
day) dilutes a short frozen run across many ordinary timesteps -- the
same day-wide-dilution problem already found and corrected for 3.4's
autoencoder (day-mean/day-max reconstruction error hid A8 entirely;
only checking the exact frozen window revealed anything).

Resolution: three separate severity functions, one per flag mechanism,
each using the magnitude definition that's actually meaningful for that
anomaly type:
    flagged_overclaim        -> 3.1's excess_energy_kwh (correct as-is,
                                 this IS what it measures)
    flagged_night_injection  -> actual energy claimed during genuinely-
                                 dark (twin_clearsky_ghi-gated) timesteps
    flagged_stuck_sensor     -> |claimed - expected_q50| integrated over
                                 ONLY the frozen run's duration, not the
                                 whole day
    fault_underclaim/normal  -> 0.0 (no fraud severity to report)
"""
from __future__ import annotations

import pandas as pd

from src.layer3_anomaly.residual import (
    NIGHT_INJECTION_DARK_GHI_THRESHOLD,
    flag_stuck_sensor,
)


def compute_night_injection_severity_kwh(
    ts_table: pd.DataFrame, dt_hours: float = 0.25,
) -> pd.Series:
    """
    Day-level severity (kWh) for flagged_night_injection days: total
    energy actually claimed during genuinely-dark timesteps (the same
    twin_clearsky_ghi gate flag_night_injection uses), NOT 3.1's
    excess_energy_kwh. Returns one row per date that has at least one
    dark-gated row -- dates with zero dark-period claims are simply
    absent (caller should treat missing as 0.0, see
    compute_daily_flag_severity_kwh below).
    """
    is_dark = ts_table["twin_clearsky_ghi"] < NIGHT_INJECTION_DARK_GHI_THRESHOLD
    night_energy_kwh = ts_table.loc[is_dark, "ac_power"] * dt_hours
    if len(night_energy_kwh) == 0:
        return pd.Series(dtype=float)
    return night_energy_kwh.groupby(ts_table.loc[is_dark].index.date).sum()


def compute_stuck_sensor_severity_kwh(
    ts_table: pd.DataFrame, dt_hours: float = 0.25, run_length_threshold: int = 8,
) -> pd.Series:
    """
    Day-level severity (kWh) for flagged_stuck_sensor days: |claimed -
    expected_q50| integrated over ONLY the rows that are part of a
    stuck/frozen run (flag_stuck_sensor's own mask) -- not the whole
    day. A day-wide computation would dilute a short frozen run across
    many ordinary timesteps, the same failure mode already found for
    3.4's autoencoder day-mean/day-max reconstruction error.
    """
    stuck_mask = flag_stuck_sensor(ts_table, run_length_threshold)
    if not stuck_mask.any():
        return pd.Series(dtype=float)
    gap_kwh = (
        (ts_table.loc[stuck_mask, "ac_power"] - ts_table.loc[stuck_mask, "expected_q50"]).abs()
        * dt_hours
    )
    return gap_kwh.groupby(ts_table.loc[stuck_mask].index.date).sum()


def compute_daily_flag_severity_kwh(
    daily: pd.DataFrame, labels: pd.Series, ts_table: pd.DataFrame, dt_hours: float = 0.25,
) -> pd.Series:
    """
    Returns severity_kwh, one value per date in `daily`'s index, using
    the flag-type-correct magnitude definition (see module docstring).
    `labels` must be classify_days()'s output (conformal.py), sharing
    `daily`'s date index. Days with no flag (fault_underclaim, normal)
    get 0.0.
    """
    night_sev = compute_night_injection_severity_kwh(ts_table, dt_hours)
    stuck_sev = compute_stuck_sensor_severity_kwh(ts_table, dt_hours)

    severity = pd.Series(0.0, index=daily.index)
    for date in daily.index:
        flag = labels.get(date, "normal")
        if flag == "flagged_overclaim":
            severity[date] = daily.loc[date, "excess_energy_kwh"]
        elif flag == "flagged_night_injection":
            severity[date] = night_sev.get(date, 0.0)
        elif flag == "flagged_stuck_sensor":
            severity[date] = stuck_sev.get(date, 0.0)
        # fault_underclaim, normal -> stays 0.0
    return severity
