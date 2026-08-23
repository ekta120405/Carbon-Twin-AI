"""
src/layer3_anomaly/residual.py

Layer 3.1: residual definition (Overview v2 SS6.1).

Computes the capacity-normalized residual between claimed and expected
generation, then aggregates to DAILY ENERGY -- the primary decision level
for Layer 3.2's conformal calibration. Timestamp-level residuals are kept
for localisation/explanation (Layer 4), not for the primary flag.

CRITICAL: uses the twin's expected_q50 (a monotone-constrained conditional
MEAN, not a true median -- see docs/LAYER2_HANDOFF.md SS2) as the point
estimate. q10/q90 are read separately here for context but the primary
residual is always computed against q50.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_timestamp_residual(
    claimed_kw: pd.Series, expected_kw: pd.Series, ac_nameplate_kw: float,
) -> pd.Series:
    """
    r(t) = (claimed(t) - expected(t)) / P_ac_nameplate

    Capacity-normalized, NOT the old (claimed-expected)/expected ratio from
    the original v1 design -- that divides by ~0 at dawn/dusk/night. This
    metric is well-defined everywhere, including where expected_kw is 0.
    """
    return (claimed_kw - expected_kw) / ac_nameplate_kw


def compute_daily_energy_residual(
    df: pd.DataFrame, claimed_col: str = "ac_power", expected_col: str = "expected_q50",
    daylight_col: str = "is_daylight", ac_nameplate_kw: float = None,
    timestep_hours: float = 0.25,
) -> pd.DataFrame:
    """
    Aggregates timestamp-level claimed/expected to DAILY ENERGY discrepancy
    -- the primary decision level (Overview v2 SS6.1). Only daylight
    timestamps contribute (night-time is definitionally zero for both
    claimed and expected under correct QC, and including it would dilute
    the signal with uninformative agreement).

    Returns one row per date:
        claimed_energy_kwh, expected_energy_kwh, excess_energy_kwh,
        daylight_hours, r_day (normalised residual), n_daylight_samples
    """
    assert ac_nameplate_kw is not None, "ac_nameplate_kw must be provided (frozen Layer 1 constant)"

    daylight = df[df[daylight_col].astype(bool)].copy()
    daylight["date"] = daylight.index.date

    grouped = daylight.groupby("date").agg(
        claimed_energy_kwh=(claimed_col, lambda s: (s * timestep_hours).sum()),
        expected_energy_kwh=(expected_col, lambda s: (s * timestep_hours).sum()),
        n_daylight_samples=(claimed_col, "size"),
    )
    grouped["daylight_hours"] = grouped["n_daylight_samples"] * timestep_hours
    grouped["excess_energy_kwh"] = grouped["claimed_energy_kwh"] - grouped["expected_energy_kwh"]
    grouped["r_day"] = grouped["excess_energy_kwh"] / (ac_nameplate_kw * grouped["daylight_hours"])

    return grouped


def build_residual_table(
    twin, physics_ac_kw: pd.Series, df: pd.DataFrame, ac_nameplate_kw: float,
    twin_feature_cols: list[str],
) -> pd.DataFrame:
    """
    Full pipeline: twin prediction -> timestamp residual -> daily aggregation.
    `df` must already contain twin_* feature columns, ac_power, is_daylight,
    twin_kt, matching the contract from data/processed/<plant>.parquet.

    CRITICAL: passes `df["twin_kt"]` as `clearsky_index` to twin.predict().
    Without this, the low-Kt (<0.4) physics-anchor fallback built into
    PIGBQRTwin.predict() (see src/layer2_twin/twin_model.py) never
    activates -- Layer 3 would silently receive the twin's raw, empirically
    unreliable overcast predictions instead of the validated fallback.
    `kt_switch` is left at its default (0.4), which is intentionally the
    same value as config.yaml's layer3.conformal.csi_bins[1] -- do not
    pass a different value here without also updating that config entry,
    or Layer 2's fallback and Layer 3's regime stratification will
    silently disagree on where "overcast" starts.

    Returns the timestamp-level table (with expected_q10/50/90 and the
    per-timestamp residual attached) -- callers needing the daily table
    should pass this result to compute_daily_energy_residual().
    """
    preds = twin.predict(df[twin_feature_cols], physics_ac_kw, clearsky_index=df["twin_kt"])
    out = df.copy()
    out = out.join(preds)
    out["residual"] = compute_timestamp_residual(out["ac_power"], out["expected_q50"], ac_nameplate_kw)
    return out
