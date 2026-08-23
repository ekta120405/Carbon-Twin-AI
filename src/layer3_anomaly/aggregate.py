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

from dataclasses import dataclass

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


def compute_divergence_severity_pct(ts_table: pd.DataFrame, labels: pd.Series):
    """
    Mean evidence_irr_divergence_pct restricted to FLAGGED days only (any
    flag type) -- NOT the whole evaluation window. This is the THIRD
    instance of the same aggregation mistake in this session (see 3.4's
    autoencoder day-mean/day-max diluting A8, and this module's original
    excess-energy severity conflating unrelated daylight variation with
    night-injection fraud): a localized signal averaged over a mostly-
    unaffected background gets diluted toward the background's value,
    regardless of how strong the signal actually is on the days that
    matter. Confirmed empirically before this fix existed: the naive
    whole-table mean showed IDENTICAL values (0.259234) across 6
    different injected replicates with different anomaly mixes -- a
    clear sign of dilution, not a genuinely flat/uninformative signal.

    Restricted to daylight rows within flagged days, since on-site
    irradiation readings aren't meaningfully comparable at night.

    Returns None (NOT 0.0) if there are no flagged days -- "no evidence
    to compute this from" is not the same claim as "zero divergence
    measured", and callers must not silently conflate the two when this
    feeds into a combined score.
    """
    flagged_dates = labels[labels.str.startswith("flagged")].index
    if len(flagged_dates) == 0:
        return None
    daylight = ts_table[ts_table["is_daylight"]]
    flagged_rows = daylight[pd.Series(daylight.index.date, index=daylight.index).isin(flagged_dates)]
    if len(flagged_rows) == 0:
        return None
    return float(flagged_rows["evidence_irr_divergence_pct"].mean())


RISK_BANDS = [
    (0, 25, "Low"),
    (25, 50, "Elevated"),
    (50, 75, "High"),
    (75, 101, "Critical"),  # 101 so score=100 lands in Critical, not off the end
]


@dataclass
class RiskScoreInputs:
    frac_flagged: float
    cum_severity_frac: float          # cumulative severity_kwh / total claimed_kwh
    cusum_max_s: float
    cusum_h: float                    # the calibrated h used, for normalization
    calibration_ratio: float
    calibration_ratio_clean_baseline: float  # THIS PLANT's own clean-data CR, not 1.0
    divergence_flagged_only: float | None    # None if no flagged days


@dataclass
class RiskScoreResult:
    score: float
    band: str
    components: dict  # each normalized [0,1] component, for transparency/debugging


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def compute_risk_score(
    inputs: RiskScoreInputs,
    weights: dict = None,
    severity_frac_cap: float = 0.20,
    calibration_ratio_deviation_cap: float = 0.5,
    divergence_cap: float = 0.3,
) -> RiskScoreResult:
    """
    Combines 3.1-3.5's five outputs into a 0-100 project-level risk score.

    Each input is normalized to [0,1] BEFORE weighting -- the five raw
    signals are on wildly different scales (a fraction, kWh-fraction,
    a CUSUM statistic relative to its own threshold, a ratio near 1.0,
    a percentage), the same scaling problem already handled for 3.4's
    autoencoder inputs.

    NORMALIZATION CHOICES, EACH DOCUMENTED:
    - frac_flagged: already naturally in [0,1], used directly.
    - severity: cum_severity_frac (severity_kwh / total claimed_kwh) is
      linearly scaled against `severity_frac_cap` (default 20% of total
      claimed energy) -- chosen as a round, conservative "even a fifth of
      claimed energy being unexplained is already severe" ceiling; not
      empirically derived, should be revisited once more replicates are
      scored (n=6/plant sampled this session is not enough to fit a
      principled cap).
    - CUSUM: cusum_max_s / cusum_h, clipped to [0,1] -- this is a natural
      normalization already grounded in a real calibrated quantity (h),
      not an arbitrary cap: 1.0 means CUSUM's OWN calibrated threshold
      was reached, not an externally chosen number.
    - calibration_ratio: measured as deviation from THIS PLANT'S OWN
      clean-data calibration ratio, NOT from a universal 1.0. Verified
      necessary: plant_2's real clean CR is 0.897 (10.3% below 1.0)
      purely from its own physics-anchor fit characteristics (see
      docs/LAYER2_HANDOFF.md Section 4) -- scoring against absolute 1.0
      would make plant_2 look moderately risky even when genuinely
      clean. |CR - clean_baseline| is linearly scaled against
      `calibration_ratio_deviation_cap` (default 0.5, i.e. a 50%
      relative swing from the plant's own clean baseline is treated as
      maximally severe).
    - divergence: |divergence_flagged_only|, scaled against
      `divergence_cap`. Uses absolute value since the flagged-only
      values observed this session were both positive and negative with
      no single consistent fraud direction (see LAYER3_HANDOFF.md
      Section 7 -- this signal is primarily an A5-specific corroborator,
      not a universal discriminator, and is weighted low accordingly).
      If no flagged days exist, contributes 0.0 (no evidence, not
      penalized).

    WEIGHTS (default, documented judgment call -- not empirically
    optimized, state this plainly in the paper): frac_flagged and
    severity get the highest weight as the most direct fraud evidence;
    CUSUM moderate (independent corroborating mechanism); calibration
    ratio lower (slower-moving, plant-baseline-relative signal);
    divergence lowest (narrowest-scope, A5-specific corroborator).
    """
    if weights is None:
        weights = {
            "frac_flagged": 0.30,
            "severity": 0.30,
            "cusum": 0.20,
            "calibration_ratio": 0.12,
            "divergence": 0.08,
        }

    frac_flagged_norm = _clip01(inputs.frac_flagged)
    severity_norm = _clip01(inputs.cum_severity_frac / severity_frac_cap)
    cusum_norm = _clip01(inputs.cusum_max_s / inputs.cusum_h) if inputs.cusum_h > 0 else 0.0
    cr_deviation = abs(inputs.calibration_ratio - inputs.calibration_ratio_clean_baseline)
    calibration_norm = _clip01(cr_deviation / calibration_ratio_deviation_cap)
    divergence_norm = (
        _clip01(abs(inputs.divergence_flagged_only) / divergence_cap)
        if inputs.divergence_flagged_only is not None else 0.0
    )

    components = {
        "frac_flagged": frac_flagged_norm,
        "severity": severity_norm,
        "cusum": cusum_norm,
        "calibration_ratio": calibration_norm,
        "divergence": divergence_norm,
    }

    score = 100.0 * sum(weights[k] * components[k] for k in components)
    score = max(0.0, min(100.0, score))

    band = next(name for lo, hi, name in RISK_BANDS if lo <= score < hi)

    return RiskScoreResult(score=score, band=band, components=components)


def calibrate_divergence_cap(
    ts_table: pd.DataFrame, reference_dates, percentile: float = 0.95,
) -> float:
    """
    Derives the divergence normalization cap from the CLEAN reference
    pool (calib+test_clean) ALONE -- never from injected/labeled data.
    This is calibration, not cosmetic scaling: the cap controls how much
    a given divergence level contributes to the final risk score, the
    same functional role as conformal's threshold or CUSUM's h. Deriving
    it by looking at injected values too would risk the exact thing this
    project has been careful to avoid everywhere else -- shaping
    detection logic around known answers.

    Uses the target percentile of |evidence_irr_divergence_pct| across
    daylight rows in the reference window (population-level, not
    flagged-only -- there's no "flagged day" concept within a purely
    clean reference window; this establishes the natural clean-data
    spread of the signal itself, which is a different, valid use of the
    whole window, not the same dilution mistake as averaging a localized
    signal across an unrelated background).
    """
    daylight = ts_table[ts_table["is_daylight"]]
    ref_rows = daylight[pd.Series(daylight.index.date, index=daylight.index).isin(reference_dates)]
    return float(ref_rows["evidence_irr_divergence_pct"].abs().quantile(percentile))


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
