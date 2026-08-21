"""
src/injection/hard_negatives.py

OWNED BY: Ekta (per current team allocation — Injection Framework). Part of the frozen injector.
Hard negatives (Overview v2 SS8.3): legitimate operational events that the
detector must NOT flag as fraud. Two families:
  - SELECTED: real events already present in the processed data (N1, N5, N7)
    -- these are not synthesized, just identified and labeled.
  - SYNTHESIZED: applied the same way as the positive taxonomy (N2, N3, N4, N6).

N1 and N7 require scanning ALL days (not just the 9-day eligible pool) to
find a real qualifying event, then using that day's UNMODIFIED data as the
hard-negative test case -- if the real event happened inside the eligible
pool, use it in place; if it happened elsewhere (e.g. in TRAIN), it still
serves as a hard negative for evaluation purposes since we are not
retraining on it, only testing that the detector doesn't fire on its shape.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def select_n1_inverter_outage(full_df: pd.DataFrame, capacity_kw: float) -> pd.Timestamp | None:
    """
    Real partial-fleet outage: a day where claimed power is meaningfully
    BELOW what clear-sky conditions would suggest (high Kt, low output) --
    the opposite signature of fraud, so it's a natural hard negative.
    Returns the date of the best candidate, or None if none found.
    """
    daily = full_df.groupby(full_df.index.date).agg(
        mean_kt=("twin_kt", "mean"),
        mean_power=("ac_power", "mean"),
        is_clean_frac=("is_clean", "mean"),
    )
    if len(daily) < 3 or daily["mean_kt"].std() < 1e-9:
        return None
    # Regression-residual approach: fit expected power ~ Kt across all days,
    # flag the day with the largest NEGATIVE residual -- i.e. most
    # underperforming relative to what its own clear-sky index would predict.
    # This naturally separates genuine outages (low power despite good Kt)
    # from genuinely cloudy days (low power WITH low Kt, small residual),
    # without an arbitrary top-N-by-Kt threshold that breaks under ties.
    slope, intercept = np.polyfit(daily["mean_kt"], daily["mean_power"], deg=1)
    predicted = slope * daily["mean_kt"] + intercept
    residual = daily["mean_power"] - predicted
    candidate = residual.idxmin()
    return pd.Timestamp(candidate)


def select_n5_heavy_cloud_day(full_df: pd.DataFrame, eligible_dates: list) -> pd.Timestamp | None:
    """Real low-clear-sky-index day within the eligible pool -- genuinely
    cloudy, wide conformal interval expected, no flag warranted."""
    daily_kt = full_df.groupby(full_df.index.date)["twin_kt"].mean()
    daily_kt = daily_kt[daily_kt.index.isin(eligible_dates)]
    if daily_kt.empty:
        return None
    return pd.Timestamp(daily_kt.idxmin())


def select_n7_cloud_enhancement(full_df: pd.DataFrame, eligible_dates: list) -> pd.Timestamp | None:
    """
    Real day with high positive on-site-vs-satellite irradiance divergence --
    exactly the pattern identified during Task 5 (cloud-edge enhancement /
    satellite grid-averaging limitation). Must not be flagged as fraud.
    """
    if "evidence_irr_divergence_pct" not in full_df.columns:
        return None
    daily_div = full_df.groupby(full_df.index.date)["evidence_irr_divergence_pct"].mean()
    daily_div = daily_div[daily_div.index.isin(eligible_dates)]
    if daily_div.empty:
        return None
    return pd.Timestamp(daily_div.idxmax())


def synthesize_n2_soiling_loss(day: pd.DataFrame, day_position: int, params: dict) -> dict:
    """Gradual efficiency loss -- `day_position` is 1-indexed position
    within the eligible pool, per the corrected days:[1,9] range."""
    loss = params["loss_per_day"] * day_position
    modified = day["ac_power"] * (1 - loss)
    return {"ac_power": modified, "meta": {"type": "N2_soiling_loss",
                                            "day_position": day_position, "loss_frac": loss}}


def synthesize_n3_curtailment(day: pd.DataFrame, rng: np.random.Generator, params: dict,
                               capacity_kw: float) -> dict:
    """Grid-ordered output cap -- claimed power correctly LOWER than
    expected due to an external constraint, not fraud."""
    cap_frac = rng.uniform(*params["cap_frac_of_capacity"])
    cap_value = capacity_kw * cap_frac
    modified = day["ac_power"].clip(upper=cap_value)
    return {"ac_power": modified, "meta": {"type": "N3_curtailment", "cap_frac": cap_frac}}


def synthesize_n4_sensor_dropout(day: pd.DataFrame, rng: np.random.Generator, params: dict) -> dict:
    """Missing data run -- should be caught by QC (Layer 1.4), never reach
    Layer 3 as a value to test, so this checks the pipeline drops it cleanly."""
    run_length = int(rng.integers(*params["missing_run_slots"]))
    daylight_idx = day.index[day["is_daylight"].astype(bool)]
    if len(daylight_idx) <= run_length:
        run_length = max(1, len(daylight_idx) - 1)
    start_pos = int(rng.integers(0, max(1, len(daylight_idx) - run_length)))
    modified = day["ac_power"].copy()
    modified.loc[daylight_idx[start_pos:start_pos + run_length]] = np.nan
    return {"ac_power": modified, "meta": {"type": "N4_sensor_dropout", "run_length": run_length}}


def synthesize_n6_extreme_heat(day: pd.DataFrame, rng: np.random.Generator, params: dict) -> dict:
    """Heat-derated output -- physically LOWER power at high cell
    temperature, correctly explained by the twin's temperature feature."""
    offset = rng.uniform(*params["t2m_offset_c"])
    gamma = -0.004  # matches the twin's temperature coefficient assumption
    derate = 1 + gamma * offset
    modified = day["ac_power"] * derate
    return {"ac_power": modified, "meta": {"type": "N6_extreme_heat",
                                            "t2m_offset_c": offset, "derate_factor": derate}}


HARD_NEGATIVE_TYPES = ["N1_inverter_outage", "N2_soiling_loss", "N3_curtailment",
                       "N4_sensor_dropout", "N5_heavy_cloud_day", "N6_extreme_heat",
                       "N7_cloud_enhancement"]
