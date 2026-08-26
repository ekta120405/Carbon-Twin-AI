"""
src/layer4_xai/severity_ranking.py

Layer 4.5: per-flag severity ranking (Overview v2 Module 4.5;
docs/LAYER4_ROADMAP.md SS4.5).

OVERLAP CHECK, DONE BEFORE WRITING ANY CODE (per roadmap SS4.5's explicit
instruction not to duplicate 3.6's work) -- read the entirety of
src/layer3_anomaly/aggregate.py first. Confirmed:
`compute_risk_score()` takes exactly ONE `RiskScoreInputs` (frac_flagged,
cum_severity_frac, cusum_max_s, cusum_h, calibration_ratio,
calibration_ratio_clean_baseline, divergence_flagged_only) and returns
exactly ONE `RiskScoreResult` -- a single project/period-level 0-100
score and band. There is no per-flag object, list, or loop anywhere in
that file; every one of its five inputs is already a project-level
aggregate before compute_risk_score even sees it. So this module is
built as a genuinely NEW, separate ranking mechanism, per the roadmap's
own stated fallback -- not an extension of 3.6, because there is nothing
at the per-flag granularity to extend.

DESIGN CHOICE, STATED PLAINLY (same discipline as 3.6's own weight
choices): this module does NOT collapse severity and confidence into one
more arbitrarily-weighted composite score on top of 3.6's already
five-input weighted score. Ranking flags for an auditor's attention is a
different job than producing one clean number for a dashboard headline --
an auditor benefits more from seeing severity ($) and confidence
(relative excess past the flag's own trigger) as two separate, honestly
-labeled columns than from a third invented weighting scheme layered on
top of 3.6's. Primary sort key is `value_at_risk_usd` (the single
directly-comparable quantity across all three flag mechanisms, since raw
severity_kwh means physically different things per mechanism -- see
aggregate.py's own module docstring); `confidence_ratio` is reported
alongside as a secondary signal, not multiplied in.

CONFIDENCE_RATIO IS MECHANISM-SPECIFIC AND NOT A UNIFIED STATISTIC --
STATED HONESTLY, NOT HIDDEN. Each flag type has a structurally different
trigger condition (a conformal bound vs. a fixed tolerance vs. a run-length
count), so "confidence" here means "how far past this flag's OWN trigger
condition, in relative terms" for each mechanism, not a common p-value or
z-score across all three:
    flagged_overclaim        -> (r_day - threshold) / threshold
    flagged_night_injection  -> worst_ac_power_kw / tolerance_kw - 1
    flagged_stuck_sensor     -> run_length / run_length_threshold - 1
All three are "fractional excess past trigger," which is comparable IN
SPIRIT (0 = barely triggered, larger = more decisively triggered) but not
a rigorously unified confidence measure -- do not present it as such in
the paper.
"""
from __future__ import annotations

import pandas as pd

from src.layer3_anomaly.aggregate import compute_daily_flag_severity_kwh
from src.layer3_anomaly.conformal import ConformalResult
from src.layer3_anomaly.residual import NIGHT_INJECTION_TOLERANCE_FRAC
from src.layer4_xai.carbon import compute_carbon_exposure
from src.layer4_xai.flag_shap import find_stuck_run_details


def compute_confidence_ratio(
    date,
    flag_type: str,
    daily: pd.DataFrame,
    ts_table: pd.DataFrame,
    conformal_result: ConformalResult,
    ac_nameplate_kw: float,
    run_length_threshold: int = 8,
) -> float | None:
    """
    Mechanism-specific "fractional excess past trigger" -- see module
    docstring for why this isn't a single unified statistic. Returns None
    if the ratio cannot be computed (should not normally happen for a
    correctly-flagged day; surfacing None rather than silently defaulting
    to 0.0 keeps a real computation failure visible).
    """
    if flag_type == "flagged_overclaim":
        r_day = daily.loc[date, "r_day"]
        threshold = conformal_result.threshold
        if threshold == 0:
            return None
        return float((r_day - threshold) / threshold)

    elif flag_type == "flagged_night_injection":
        from src.layer3_anomaly.residual import NIGHT_INJECTION_DARK_GHI_THRESHOLD
        ts_dates = pd.Series(ts_table.index.date, index=ts_table.index)
        day_table = ts_table[ts_dates == date]
        is_dark = day_table["twin_clearsky_ghi"] < NIGHT_INJECTION_DARK_GHI_THRESHOLD
        dark_rows = day_table[is_dark]
        if len(dark_rows) == 0:
            return None
        worst_ac_power_kw = float(dark_rows["ac_power"].max())
        tolerance_kw = NIGHT_INJECTION_TOLERANCE_FRAC * ac_nameplate_kw
        if tolerance_kw == 0:
            return None
        return float(worst_ac_power_kw / tolerance_kw - 1)

    elif flag_type == "flagged_stuck_sensor":
        ts_dates = pd.Series(ts_table.index.date, index=ts_table.index)
        day_table = ts_table[ts_dates == date]
        details = find_stuck_run_details(day_table, run_length_threshold)
        if details is None:
            return None
        return float(details["run_length"] / run_length_threshold - 1)

    else:
        return None


def rank_flags(
    daily: pd.DataFrame,
    labels: pd.Series,
    ts_table: pd.DataFrame,
    conformal_result: ConformalResult,
    ac_nameplate_kw: float,
    emission_factor_tco2_per_mwh: float,
    carbon_price_usd_per_tco2: float,
    dt_hours: float = 0.25,
    run_length_threshold: int = 8,
) -> pd.DataFrame:
    """
    One row per flagged day, sorted by value_at_risk_usd descending (the
    single directly-comparable-across-mechanisms quantity -- see module
    docstring). Columns: flag_type, severity_kwh, excess_tco2e,
    value_at_risk_usd, confidence_ratio.
    """
    severity_kwh_series = compute_daily_flag_severity_kwh(daily, labels, ts_table, dt_hours)
    flagged_dates = labels[labels.str.startswith("flagged")].index

    rows = []
    for date in flagged_dates:
        flag_type = labels.loc[date]
        severity_kwh = float(severity_kwh_series.get(date, 0.0))
        carbon = compute_carbon_exposure(severity_kwh, emission_factor_tco2_per_mwh, carbon_price_usd_per_tco2)
        confidence = compute_confidence_ratio(
            date, flag_type, daily, ts_table, conformal_result, ac_nameplate_kw, run_length_threshold,
        )
        rows.append(
            {
                "date": date,
                "flag_type": flag_type,
                "severity_kwh": severity_kwh,
                "excess_tco2e": carbon.excess_tco2e,
                "value_at_risk_usd": carbon.value_at_risk_usd,
                "confidence_ratio": confidence,
            }
        )
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out.set_index("date") if "date" in out.columns else out
    out = out.set_index("date").sort_values("value_at_risk_usd", ascending=False)
    return out
