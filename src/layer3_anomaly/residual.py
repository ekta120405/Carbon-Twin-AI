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


NIGHT_INJECTION_DARK_GHI_THRESHOLD = 1.0  # W/m2, effectively zero, float noise only
NIGHT_INJECTION_TOLERANCE_FRAC = 0.015    # fraction of ac_nameplate_kw


def flag_night_injection(df: pd.DataFrame, ac_nameplate_kw: float) -> pd.Series:
    """
    Deterministic guardrail for A3 (night injection) -- a categorically
    different case from 3.1-3.3's magnitude-based residual test, which is
    STRUCTURALLY BLIND to it: compute_daily_energy_residual() filters to
    is_daylight rows before aggregating, so any claim recorded when
    is_daylight=False contributes exactly zero to r_day, regardless of
    size. This function is a separate, additive flag path -- it does not
    change daylight-only residual logic above, which stays correctly
    scoped for continuous over-claiming during generation hours.

    TWO PRIOR APPROACHES WERE TRIED AND REJECTED, both verified against
    real injected replicate data, not assumed:
    - Using the existing precomputed `exceeds_ceiling_fraud` column:
      REJECTED. The injector (src/injection/run_injection.py) only
      modifies `ac_power` -- it never recomputes derived QC columns
      afterward, so this flag is stale (computed on pre-injection clean
      data) and never fires on injected rows. Confirmed empirically: 0/11
      A3-injected nights showed exceeds_ceiling_fraud=True.
    - Gating on `is_daylight==False` with a flat/recomputed-ceiling
      tolerance: REJECTED. `is_daylight` uses a min_solar_elevation_deg=10
      cutoff (config.yaml), so "night" includes a real dawn/dusk twilight
      zone with legitimate, sometimes substantial generation (up to
      ~2000 kW observed). A flat tolerance and a recomputed
      physics_ceiling_fraud_kw comparison both produced high false-
      positive rates (10/10 and 20/20 clean nights tested) concentrated
      exactly at IST sunrise/sunset.

    RESOLUTION: gate on `twin_clearsky_ghi < NIGHT_INJECTION_DARK_GHI_THRESHOLD`
    instead of `is_daylight`. Clearsky GHI is the actual input
    physics_anchor.py's PVWatts chain is computed from -- when it's zero,
    physics_ac_kw is zero BY CONSTRUCTION of the formula, not by an
    empirically-tuned threshold. This has no calibration gap to discover,
    unlike the two rejected approaches above.

    Tolerance: NIGHT_INJECTION_TOLERANCE_FRAC (1.5% of ac_nameplate_kw).
    Verified against real clean train/calib data, both plants:
    - plant_1: 0/1039 genuinely-dark clean rows exceed this tolerance
    - plant_2: 4/1097 do (~0.36%) -- a small, KNOWN, irreducible overlap
      between plant_2's night sensor noise and A3's own minimum injection
      magnitude (2% of capacity, config/anomalies.yaml) -- not fully
      eliminable by any threshold choice; report this honestly rather
      than picking a tolerance high enough to hide it (which would also
      start missing real low-magnitude A3 injections).
    Verified coverage against real A3-injected replicates: 11/11 plant_1
    and 5/5 plant_2 injected days had at least one row flagged (the unit
    that matters, since this feeds a DAY-level flag in 3.3 -- not every
    individual injected row needs to be caught, just enough per day).

    Returns a boolean Series, True where a night-injection-style
    fabrication is suspected.
    """
    is_dark = df["twin_clearsky_ghi"] < NIGHT_INJECTION_DARK_GHI_THRESHOLD
    tolerance_kw = NIGHT_INJECTION_TOLERANCE_FRAC * ac_nameplate_kw
    return is_dark & (df["ac_power"] > tolerance_kw)


def flag_stuck_sensor(df: pd.DataFrame, run_length_threshold: int = 8) -> pd.Series:
    """
    Deterministic guardrail for A8 (flatline / stuck-sensor replay).
    Analogous investigation to flag_night_injection above: Layer 1 already
    computes an `is_stuck` column (config.yaml's
    layer1.quality.stuck_sensor_run_length=8), but it suffers the IDENTICAL
    staleness bug -- the injector (apply_a8_flatline, taxonomy.py) only
    modifies `ac_power`, never recomputes `is_stuck` afterward. Confirmed
    empirically: is_stuck=False on all 29/29 rows of a real, verified
    frozen run. Using the stored column would silently catch nothing,
    exactly like the first exceeds_ceiling_fraud attempt for A3.

    This recomputes the same "N consecutive identical DAYLIGHT values"
    rule fresh, directly from the live ac_power column, using the same
    threshold (8) as Layer 1's original definition.

    Verified against real data, both plants:
    - A8-injected replicates: 8/8 (plant_1), 7/7 (plant_2) days caught
    - real clean train/calib data: 0/25 false-positive days on EITHER
      plant -- unlike the A3 guardrail, this one has no known noise/fraud
      overlap; a genuinely clean plant essentially never produces 8+
      bit-identical consecutive readings, since real sensor/inverter
      noise varies continuously.

    Also caught 3.4's autoencoder missing this entirely: reconstruction
    error on the exact labeled frozen window was well BELOW baseline
    (not elevated at all), despite features engineered specifically to
    target it (ac_power_diff1). Don't rely on the autoencoder for A8;
    this deterministic check is the actual mechanism that catches it.

    Returns a boolean Series, True where a stuck/frozen-value run of at
    least run_length_threshold consecutive daylight readings is detected.
    """
    daylight_idx = df.index[df["is_daylight"].astype(bool)]
    ac = df.loc[daylight_idx, "ac_power"]
    change = ac.ne(ac.shift()).cumsum()
    run_sizes = ac.groupby(change).transform("size")
    stuck_mask = run_sizes >= run_length_threshold

    result = pd.Series(False, index=df.index)
    result.loc[daylight_idx] = stuck_mask
    return result


def compute_daily_night_injection_flag(df: pd.DataFrame, ac_nameplate_kw: float) -> pd.Series:
    """
    Day-level aggregation of flag_night_injection(): True if ANY row that
    day trips the guardrail. Operates on the FULL table (not daylight-
    filtered) since flag_night_injection specifically targets non-daylight
    rows. Index is per-date, matching compute_daily_energy_residual()'s
    output index so the two can be joined directly.
    """
    row_flags = flag_night_injection(df, ac_nameplate_kw)
    return row_flags.groupby(df.index.date).any()


def compute_daily_stuck_sensor_flag(df: pd.DataFrame, run_length_threshold: int = 8) -> pd.Series:
    """Day-level aggregation of flag_stuck_sensor(): True if ANY row that day is part of a stuck/frozen run."""
    row_flags = flag_stuck_sensor(df, run_length_threshold)
    return row_flags.groupby(df.index.date).any()


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
