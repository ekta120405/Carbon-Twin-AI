"""
src/injection/taxonomy.py

OWNED BY: Ekta (per current team allocation — Injection Framework). Part of the frozen injector (git tag injector-v1.0).
Do not modify after freezing without re-tagging and re-running the full
evaluation -- see run_injection.py for the freeze protocol.

Implements the positive anomaly taxonomy A1-A8 (Overview v2 SS8.2, config in
config/anomalies.yaml). Each function:
  - takes ONE DAY's slice of the processed plant DataFrame (~96 rows, 15-min)
  - returns a dict: {"ac_power": modified Series, "evidence_onsite_irradiation":
    modified Series or None, "meta": {type, params...}}
  - never reads or depends on any Layer 2/3 code -- the injector must have
    zero knowledge of the detector, or the evaluation is circular.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _local_hour(index: pd.DatetimeIndex, tz: str = "Asia/Kolkata") -> pd.Series:
    local = index.tz_convert(tz)
    return pd.Series(local.hour + local.minute / 60, index=index)


def apply_a1_constant_overclaim(day: pd.DataFrame, rng: np.random.Generator, params: dict) -> dict:
    """Multiply claimed power by a constant factor across the whole day."""
    delta = rng.uniform(*params["delta_range"])
    modified = day["ac_power"] * (1 + delta)
    return {"ac_power": modified, "evidence_onsite_irradiation": None,
            "meta": {"type": "A1_constant_overclaim", "delta": delta}}


def apply_a2_peak_hour_shaving(day: pd.DataFrame, rng: np.random.Generator, params: dict) -> dict:
    """Inflate only during the specified local-time hour window."""
    delta = rng.uniform(*params["delta_range"])
    lo, hi = params["hours_local"]
    hour = _local_hour(day.index)
    mask = (hour >= lo) & (hour <= hi)
    modified = day["ac_power"].copy()
    modified[mask] = modified[mask] * (1 + delta)
    return {"ac_power": modified, "evidence_onsite_irradiation": None,
            "meta": {"type": "A2_peak_hour_shaving", "delta": delta,
                     "hours_local": [lo, hi], "n_affected": int(mask.sum())}}


def apply_a3_night_injection(day: pd.DataFrame, rng: np.random.Generator, params: dict,
                              capacity_kw: float) -> dict:
    """Add non-zero output at timestamps where solar elevation says there
    should be none -- the crudest, most physically obvious fabrication."""
    magnitude_frac = rng.uniform(*params["magnitude_frac_of_capacity"])
    night_mask = ~day["is_daylight"].astype(bool)
    modified = day["ac_power"].copy()
    modified[night_mask] = magnitude_frac * capacity_kw
    return {"ac_power": modified, "evidence_onsite_irradiation": None,
            "meta": {"type": "A3_night_injection", "magnitude_frac": magnitude_frac,
                     "n_affected": int(night_mask.sum())}}


def apply_a4_day_swap(target_day: pd.DataFrame, source_day: pd.DataFrame) -> dict:
    """
    Replace the target (cloudy) day's claimed generation SHAPE with the
    source (sunny) day's shape, aligned by time-of-day. Weather features for
    the target day are left untouched -- this is exactly the mismatch the
    twin should catch: correct (cloudy) weather, implausible (sunny) claim.
    """
    target_hour = target_day.index.hour * 4 + target_day.index.minute // 15  # slot-of-day
    source_hour = source_day.index.hour * 4 + source_day.index.minute // 15
    source_by_slot = pd.Series(source_day["ac_power"].values, index=source_hour)
    modified = pd.Series(
        source_by_slot.reindex(target_hour).values, index=target_day.index
    )
    return {"ac_power": modified, "evidence_onsite_irradiation": None,
            "meta": {"type": "A4_day_swap",
                     "source_date": str(source_day.index[0].date()),
                     "target_date": str(target_day.index[0].date())}}


def apply_a5_weather_spoofing(day: pd.DataFrame, rng: np.random.Generator, params: dict) -> dict:
    """
    Inflate BOTH the on-site sensor reading AND the claimed generation
    together, simulating a tampered on-site sensor used to 'justify' an
    inflated claim. The independent NASA-derived twin features are untouched
    -- this is what the flag-SHAP divergence feature (Layer 4.2) should catch.
    """
    inflation = rng.uniform(*params["onsite_irr_inflation"])
    modified_power = day["ac_power"] * (1 + inflation)
    modified_onsite_irr = (
        day["evidence_onsite_irradiation"] * (1 + inflation)
        if "evidence_onsite_irradiation" in day.columns else None
    )
    return {"ac_power": modified_power, "evidence_onsite_irradiation": modified_onsite_irr,
            "meta": {"type": "A5_weather_spoofing", "inflation": inflation}}


def apply_a6_slow_drift(days: list[pd.DataFrame], rng: np.random.Generator, params: dict) -> list[dict]:
    """
    Ramps overclaim from 0 to `final_delta` across a contiguous block of days
    sampled within `ramp_days` (now [3,9], fitting the 9-day eligible pool).
    Returns one result dict PER DAY in the block -- the hardest case, because
    no single day crosses a magnitude threshold; only CUSUM should catch it.
    """
    final_delta = rng.uniform(*params["final_delta"])
    n_days = len(days)
    results = []
    for i, day in enumerate(days):
        day_delta = final_delta * (i + 1) / n_days  # linear ramp
        modified = day["ac_power"] * (1 + day_delta)
        results.append({"ac_power": modified, "evidence_onsite_irradiation": None,
                         "meta": {"type": "A6_slow_drift", "day_index_in_ramp": i,
                                  "ramp_length_days": n_days, "day_delta": day_delta,
                                  "final_delta": final_delta}})
    return results


def apply_a7_capacity_violation(day: pd.DataFrame, rng: np.random.Generator, params: dict,
                                 capacity_kw: float) -> dict:
    """Scale the day so its peak exceeds nameplate capacity outright."""
    overshoot = rng.uniform(*params["overshoot"])
    current_peak = day["ac_power"].max()
    if current_peak <= 0:
        scale = 1.0
    else:
        target_peak = capacity_kw * overshoot
        scale = target_peak / current_peak
    modified = day["ac_power"] * scale
    return {"ac_power": modified, "evidence_onsite_irradiation": None,
            "meta": {"type": "A7_capacity_violation", "overshoot": overshoot, "scale": scale}}


def apply_a8_flatline(day: pd.DataFrame, rng: np.random.Generator, params: dict) -> dict:
    """Freeze claimed power at a plausible value for a run of consecutive
    daylight slots -- simulates a stuck/replayed meter reading."""
    run_length = int(rng.integers(*params["run_length_slots"]))
    daylight_idx = day.index[day["is_daylight"].astype(bool)]
    if len(daylight_idx) <= run_length:
        run_length = max(1, len(daylight_idx) - 1)
    start_pos = int(rng.integers(0, max(1, len(daylight_idx) - run_length)))
    start_ts = daylight_idx[start_pos]
    freeze_value = day.loc[start_ts, "ac_power"]

    modified = day["ac_power"].copy()
    run_slots = daylight_idx[start_pos:start_pos + run_length]
    modified.loc[run_slots] = freeze_value

    return {"ac_power": modified, "evidence_onsite_irradiation": None,
            "meta": {"type": "A8_flatline", "run_length_slots": run_length,
                     "start_ts": str(start_ts), "freeze_value": float(freeze_value)}}


POSITIVE_TYPES = ["A1_constant_overclaim", "A2_peak_hour_shaving", "A3_night_injection",
                  "A4_day_swap", "A5_weather_spoofing", "A6_slow_drift",
                  "A7_capacity_violation", "A8_flatline"]
