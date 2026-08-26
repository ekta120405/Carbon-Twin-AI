"""
src/layer4_xai/flag_shap.py

Layer 4.2: Flag-SHAP -- "why is this suspicious?" (Overview v2 Module 4.2;
docs/LAYER4_ROADMAP.md SS4.2).

TWO SEPARATE CODE PATHS, per roadmap SS2.5 -- do not conflate them:

Path A (statistical flags -- flagged_overclaim): deviation-based ranking
of evidence features against the clean reference pool (calib+test_clean,
n=10 real days), NOT a trained model. Same small-n discipline as every
other Layer 3 module this project built (LSTM, autoencoder both needed
real bug fixes at ~600-900 rows; n=10 clean reference days is far too
little to responsibly train a new explainer on).

Path B (guardrail flags -- flagged_night_injection, flagged_stuck_sensor):
template-based, reporting the exact deterministic rule and trigger values
that fired. There is nothing non-deterministic to attribute here -- do not
force statistical ranking onto an if-statement.

VERIFIED AGAINST THE REAL REPO BEFORE WRITING THIS MODULE (not assumed
from the roadmap's sketch, which flagged its own column names as
placeholders -- see roadmap SS3):

- Real evidence_* columns present in data/processed/<plant>.parquet:
  evidence_onsite_irradiation, evidence_onsite_ambient_temp,
  evidence_onsite_module_temp, evidence_irr_divergence,
  evidence_irr_divergence_pct, evidence_temp_divergence.
- evidence_yield_discontinuity, listed as a candidate feature in roadmap
  SS4.2, does NOT exist in the real data -- src/layer1_data/features.py's
  build_evidence_features() only adds it `if daily_yield is not None`, and
  `daily_yield` is not a column anywhere upstream of the processed
  parquet (confirmed: 'daily_yield' not in df.columns). Substituted below
  with `yield_jump_frac`, a feature computed directly from Layer 3's own
  `claimed_energy_kwh` (day-over-day diff, normalized by ac_nameplate_kw)
  -- same INTENT ("sudden jump vs. the plant's own recent trend") using
  data that actually exists, not the exact column name the roadmap assumed.
- Night-time claimed energy re-uses aggregate.compute_night_injection_
  severity_kwh() directly (same twin_clearsky_ghi dark-gate as
  flag_night_injection) rather than reimplementing a second "is it night"
  definition -- per this project's established rule (roadmap SS2.4,
  extended here to a second function) of never recomputing something
  already built and verified elsewhere in the codebase.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.layer3_anomaly.residual import (
    NIGHT_INJECTION_DARK_GHI_THRESHOLD,
    flag_stuck_sensor,
)
from src.layer3_anomaly.aggregate import compute_night_injection_severity_kwh

EVIDENCE_FEATURE_COLS = [
    "evidence_irr_divergence_pct",
    "evidence_temp_divergence",
    "evidence_onsite_irradiation",
]

# -----------------------------------------------------------------------
# Day-level evidence feature construction (Path A input)
# -----------------------------------------------------------------------


def compute_day_level_evidence_features(
    ts_table: pd.DataFrame, daily: pd.DataFrame, ac_nameplate_kw: float, dt_hours: float = 0.25,
) -> pd.DataFrame:
    """
    Aggregates timestamp-level evidence to ONE ROW PER DATE, at this
    project's established daily decision level (Overview v2 SS6.1), using
    the SAME daylight-only convention as every other evidence computation
    in this codebase (aggregate.py's divergence severity and divergence
    cap calibration) -- comparing on-site sensor readings to satellite
    data at night is not meaningful (on-site irradiance is ~0 by
    definition after dark).

    `daily` must be residual.compute_daily_energy_residual()'s output
    (for claimed_energy_kwh, used by yield_jump_frac).

    Returns columns: irr_divergence_pct_mean, temp_divergence_mean,
    onsite_irradiation_mean, night_claimed_kwh, yield_jump_frac.
    """
    daylight = ts_table[ts_table["is_daylight"].astype(bool)].copy()
    daylight["date"] = daylight.index.date

    agg = daylight.groupby("date").agg(
        irr_divergence_pct_mean=("evidence_irr_divergence_pct", "mean"),
        temp_divergence_mean=("evidence_temp_divergence", "mean"),
        onsite_irradiation_mean=("evidence_onsite_irradiation", "mean"),
    )

    night_kwh = compute_night_injection_severity_kwh(ts_table, dt_hours)
    agg["night_claimed_kwh"] = agg.index.map(lambda d: float(night_kwh.get(d, 0.0)))

    claimed = daily["claimed_energy_kwh"].sort_index()
    jump = claimed.diff().abs() / ac_nameplate_kw
    jump.iloc[0] = 0.0  # first day in the table has no prior day to jump from
    agg["yield_jump_frac"] = agg.index.map(
        lambda d: float(jump.get(d, 0.0)) if d in jump.index else 0.0
    )

    return agg


# -----------------------------------------------------------------------
# Path A -- deviation-based ranking
# -----------------------------------------------------------------------


def rank_suspicion_drivers(
    flagged_day_evidence: pd.Series,
    clean_reference_pool: pd.DataFrame,
    evidence_feature_cols: list[str] = None,
) -> list[tuple[str, float]]:
    """
    NOT Shapley values -- a simpler, small-n-appropriate deviation ranking
    (z-score against the clean reference pool) serving the same
    interpretive purpose: which evidence features are most unusual for
    this flagged day relative to genuinely clean data. See module
    docstring for why a trained explainer was rejected.

    `flagged_day_evidence` is one row (Series) from
    compute_day_level_evidence_features()'s output.
    `clean_reference_pool` is the SAME function's output restricted to
    calib+test_clean dates ONLY (never injected data -- same calibration
    discipline as conformal's threshold, CUSUM's h, and the divergence
    cap, all fit on calib-only/clean-only data per this project's rule).

    Returns [(feature, z_score), ...] sorted by |z_score| descending.
    """
    if evidence_feature_cols is None:
        evidence_feature_cols = list(clean_reference_pool.columns)

    z_scores = {}
    for col in evidence_feature_cols:
        mean = clean_reference_pool[col].mean()
        std = clean_reference_pool[col].std()
        z_scores[col] = (flagged_day_evidence[col] - mean) / std if std > 0 else 0.0
    return sorted(z_scores.items(), key=lambda kv: abs(kv[1]), reverse=True)


@dataclass
class StatisticalFlagExplanation:
    date: object
    flag_type: str
    top_drivers: list[tuple[str, float]]  # [(feature, z_score), ...]
    raw_values: dict  # feature -> this day's raw value, for report context
    reference_n: int


def explain_statistical_flag(
    date, evidence_table: pd.DataFrame, clean_reference_pool: pd.DataFrame,
    flag_type: str = "flagged_overclaim", evidence_feature_cols: list[str] = None,
) -> StatisticalFlagExplanation:
    row = evidence_table.loc[date]
    ranked = rank_suspicion_drivers(row, clean_reference_pool, evidence_feature_cols)
    cols = evidence_feature_cols or list(clean_reference_pool.columns)
    return StatisticalFlagExplanation(
        date=date, flag_type=flag_type, top_drivers=ranked,
        raw_values={c: float(row[c]) for c in cols}, reference_n=len(clean_reference_pool),
    )


# -----------------------------------------------------------------------
# Path B -- deterministic guardrail templates
# -----------------------------------------------------------------------

STUCK_SENSOR_CAVEAT = (
    "KNOWN, DOCUMENTED FALSE-POSITIVE RISK: this rule also fires on "
    "N3_curtailment (a legitimate grid-ordered output cap), when the cap "
    "binds continuously for >= the run-length threshold. A discriminator "
    "based on the sign of (claimed - expected_q50) within the frozen run "
    "was tried and reverted after it caused a severe coverage regression "
    "on real A8 examples (8/8->4/8, 7/7->1/7 across the two plants) -- see "
    "docs/LAYER3_HANDOFF.md SS3. This flag should be treated as suspected, "
    "not confirmed, fraud until manually reviewed."
)


@dataclass
class GuardrailFlagExplanation:
    date: object
    flag_type: str
    text: str
    trigger_values: dict
    caveat: str = ""


def explain_night_injection_flag(date, day_ts_table: pd.DataFrame, ac_nameplate_kw: float) -> GuardrailFlagExplanation:
    """
    Path B, A3. Reports the SPECIFIC worst-violating row for the day (the
    row with the largest claimed power during a genuinely-dark timestep),
    per this project's Section-0 standing rule of reporting exact affected
    rows rather than a day-wide blanket statistic -- extended here from
    severity computation to explanation reporting for consistency.
    """
    is_dark = day_ts_table["twin_clearsky_ghi"] < NIGHT_INJECTION_DARK_GHI_THRESHOLD
    dark_rows = day_ts_table[is_dark]
    if len(dark_rows) == 0 or dark_rows["ac_power"].max() <= 0:
        worst = day_ts_table.iloc[0]
        text = (
            f"Flagged as night injection, but no dark-gated row with positive "
            f"claimed power was found on re-check -- inspect manually, this "
            f"combination should not occur if the flag fired correctly."
        )
    else:
        worst = dark_rows.loc[dark_rows["ac_power"].idxmax()]
        tolerance_kw = 0.015 * ac_nameplate_kw
        text = (
            f"Flagged: {worst['ac_power']:.1f} kW claimed at "
            f"{worst.name} where twin_clearsky_ghi={worst['twin_clearsky_ghi']:.3f} W/m^2 "
            f"(below the {NIGHT_INJECTION_DARK_GHI_THRESHOLD} W/m^2 dark-sky gate, "
            f"exceeding the {tolerance_kw:.1f} kW tolerance) -- physically "
            f"implausible generation during deep night, since clearsky GHI=0 "
            f"means physics_ac_kw=0 by construction of the physics anchor."
        )
    return GuardrailFlagExplanation(
        date=date, flag_type="flagged_night_injection", text=text,
        trigger_values={
            "worst_ac_power_kw": float(worst["ac_power"]),
            "twin_clearsky_ghi": float(worst["twin_clearsky_ghi"]),
            "timestamp": str(worst.name),
        },
    )


def find_stuck_run_details(day_ts_table: pd.DataFrame, run_length_threshold: int = 8) -> dict | None:
    """
    Re-derives the SPECIFIC frozen run within this day (not just the
    boolean day-level flag) -- reuses flag_stuck_sensor()'s exact mask so
    the reported run is guaranteed consistent with what actually triggered
    the flag, then finds the run's start timestamp, length, and frozen
    value. Returns None if no run is found (should not happen for a day
    that was actually flagged_stuck_sensor -- surfacing None rather than
    silently guessing is intentional).
    """
    mask = flag_stuck_sensor(day_ts_table, run_length_threshold)
    if not mask.any():
        return None
    daylight_idx = day_ts_table.index[day_ts_table["is_daylight"].astype(bool)]
    ac = day_ts_table.loc[daylight_idx, "ac_power"]
    change = ac.ne(ac.shift()).cumsum()
    run_sizes = ac.groupby(change).transform("size")
    stuck_groups = change[run_sizes >= run_length_threshold]
    if len(stuck_groups) == 0:
        return None
    # longest run in this day, in case of multiple (rare but not impossible)
    run_id = run_sizes[stuck_groups.index].idxmax()
    run_group_id = change.loc[run_id]
    run_idx = change[change == run_group_id].index
    return {
        "start_ts": run_idx[0],
        "end_ts": run_idx[-1],
        "run_length": len(run_idx),
        "frozen_value_kw": float(ac.loc[run_idx[0]]),
    }


def explain_stuck_sensor_flag(date, day_ts_table: pd.DataFrame, run_length_threshold: int = 8) -> GuardrailFlagExplanation:
    """Path B, A8. Carries the N3-curtailment false-positive caveat
    directly in the returned object (roadmap SS2.6) -- not as a footnote
    someone building the evidence bundle or dashboard might drop."""
    details = find_stuck_run_details(day_ts_table, run_length_threshold)
    if details is None:
        text = "Flagged as stuck sensor, but the run could not be re-located on inspection -- check manually."
        trigger_values = {}
    else:
        text = (
            f"Flagged: {details['run_length']} consecutive identical readings "
            f"of {details['frozen_value_kw']:.1f} kW during daylight, from "
            f"{details['start_ts']} to {details['end_ts']} -- possible sensor "
            f"freeze or replay."
        )
        trigger_values = {
            "run_length": details["run_length"],
            "frozen_value_kw": details["frozen_value_kw"],
            "start_ts": str(details["start_ts"]),
            "end_ts": str(details["end_ts"]),
        }
    return GuardrailFlagExplanation(
        date=date, flag_type="flagged_stuck_sensor", text=text,
        trigger_values=trigger_values, caveat=STUCK_SENSOR_CAVEAT,
    )


# -----------------------------------------------------------------------
# Dispatcher
# -----------------------------------------------------------------------


def explain_flagged_day(
    date, flag_type: str, ts_table: pd.DataFrame, evidence_table: pd.DataFrame = None,
    clean_reference_pool: pd.DataFrame = None, ac_nameplate_kw: float = None,
):
    """
    Routes to Path A or Path B by flag_type. fault_underclaim/normal are
    not suspicion-explained here (nothing to explain -- normal is not a
    flag, fault_underclaim is an operational-fault channel, not fraud).
    """
    ts_dates = pd.Series(ts_table.index.date, index=ts_table.index)
    day_ts_table = ts_table[ts_dates == date]

    if flag_type == "flagged_overclaim":
        return explain_statistical_flag(date, evidence_table, clean_reference_pool, flag_type=flag_type)
    elif flag_type == "flagged_night_injection":
        return explain_night_injection_flag(date, day_ts_table, ac_nameplate_kw)
    elif flag_type == "flagged_stuck_sensor":
        return explain_stuck_sensor_flag(date, day_ts_table)
    else:
        raise ValueError(
            f"No suspicion explanation defined for flag_type={flag_type!r} "
            f"(expected one of flagged_overclaim / flagged_night_injection / "
            f"flagged_stuck_sensor -- fault_underclaim/normal are not fraud flags)."
        )
