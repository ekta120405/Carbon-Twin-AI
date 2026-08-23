"""
src/layer3_anomaly/conformal.py

Layer 3.2: split conformal calibration (one-sided, over-claims only).

Fits ONLY on the `calib` split (days 21-25, real clean data) -- per this
project's established rule: never calibrate against train, test_clean, or
injected data. That's what gives the false-positive-rate guarantee its
validity.

DESIGN DECISION (documented, not silent): with only 5 calib days per
plant, the standard split-conformal requirement for a VALID threshold at
the nominal alpha is n >= ceil(1/alpha) - 1 = 19 (the threshold is the
k-th order statistic, k = ceil((n+1)(1-alpha)), and this requires
k <= n). n=5 is far below this floor -- before even considering
per-regime (Kt-stratified) thresholds, which would push some bins to
n=0-3 (plant_1 has ZERO overcast calib days: nothing to calibrate an
overcast-specific threshold from at all).

Alternatives considered and rejected (do not silently revisit these
without a real reason -- they were ruled out on the actual math, not
skipped):
- Re-splitting days to give calib >= 19 days: would sacrifice over a
  third of the entire 34-day dataset, gutting train/test_clean. The
  injector is already frozen (injector-v1.0) against the CURRENT split
  boundaries -- re-splitting would require re-tagging and re-verifying
  it, out of scope for fixing one split's sample size.
- CQR (conformalizing on the twin's own q10/q90 instead of raw r_day):
  uses the IDENTICAL ceil((n+1)(1-alpha)) <= n order-statistic formula,
  just on a different nonconformity score. Same n=5 floor, same problem.
  Does not help.

Resolution (this module's actual behavior):
- ONE pooled, unstratified threshold per plant, from all calib days'
  r_day values (NOT stratified by Kt regime) -- stratifying would only
  make an already-inadequate n worse, not better.
- Threshold = max(calib r_day values) when n_calib is below the nominal-
  alpha floor -- the standard conservative split-conformal convention.
  (If n_calib is ever large enough to clear the floor, this correctly
  falls back to the proper order-statistic threshold instead of always
  taking the max -- this function isn't a max-only hack.)
- The ACTUAL achievable false-positive rate is reported explicitly:
  alpha_achievable = 1 / (n_calib + 1) when below the floor. At n_calib=5
  this is ~0.167 (~16.7%), NOT the nominal config alpha (0.05). Report
  this honestly in the paper -- do not silently present it as "5% FPR."
- Each day is still tagged with its Kt regime for interpretability /
  paper reporting (e.g. "flags concentrated in clear-sky days"), but the
  threshold itself carries no regime-specific statistical guarantee --
  don't conflate diagnostic labeling with a second calibration.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ConformalResult:
    threshold: float
    n_calib: int
    alpha_nominal: float
    alpha_achievable: float
    calib_r_day_values: np.ndarray


def min_n_for_nominal_alpha(alpha_nominal: float) -> int:
    """
    Minimum n_calib such that a valid order-statistic threshold exists at
    the nominal alpha: k = ceil((n+1)(1-alpha)) <= n.
    Solving the inequality: n >= (1-alpha)/alpha = 1/alpha - 1.
    """
    return math.ceil(1.0 / alpha_nominal) - 1


def compute_achievable_alpha(n_calib: int) -> float:
    """
    The actual false-positive-rate guarantee achievable by a threshold
    set as the max of n_calib calibration scores: alpha_achievable =
    1 / (n_calib + 1). Only meaningful when n_calib is below
    min_n_for_nominal_alpha -- otherwise the nominal alpha is achievable
    directly (see fit_conformal_threshold).
    """
    if n_calib < 1:
        raise ValueError("Need at least 1 calibration sample to compute a threshold.")
    return 1.0 / (n_calib + 1)


def fit_conformal_threshold(
    calib_r_day: pd.Series, alpha_nominal: float = 0.05,
) -> ConformalResult:
    """
    One-sided, pooled (unstratified) split-conformal threshold.

    calib_r_day: r_day values from ONLY the calib split (real clean
    data -- never train/test_clean/injected).

    If n_calib >= min_n_for_nominal_alpha(alpha_nominal): uses the proper
    ceil((n+1)(1-alpha))-th order statistic, achieving the nominal alpha
    exactly.

    If n_calib is below that floor (true for this project, n=5): uses
    the conservative max-order-statistic convention instead, and reports
    the REAL achievable alpha (1/(n_calib+1)), not the nominal one.
    """
    values = calib_r_day.dropna().values
    n = len(values)
    if n < 1:
        raise ValueError("No calibration samples provided.")

    sorted_values = np.sort(values)
    min_n_needed = min_n_for_nominal_alpha(alpha_nominal)

    if n >= min_n_needed:
        k = math.ceil((n + 1) * (1 - alpha_nominal))
        k = min(k, n)  # safety clamp, should not trigger given the n check above
        threshold = sorted_values[k - 1]  # k-th smallest, 1-indexed
        alpha_achievable = alpha_nominal
    else:
        threshold = sorted_values[-1]
        alpha_achievable = compute_achievable_alpha(n)

    return ConformalResult(
        threshold=float(threshold),
        n_calib=n,
        alpha_nominal=alpha_nominal,
        alpha_achievable=alpha_achievable,
        calib_r_day_values=sorted_values,
    )


def flag_days(daily_r_day: pd.Series, result: ConformalResult) -> pd.Series:
    """
    One-sided flagging: r_day > threshold -> flagged (candidate over-claim).
    r_day meaningfully negative is NEVER flagged here -- that's a fault-
    channel signal (under-generation: inverter outage, soiling,
    curtailment), not a fraud flag. Keeping this strictly one-sided is
    what keeps the false-positive rate honest against the project's
    known under-generation hard negatives.
    """
    return daily_r_day > result.threshold


def classify_day(
    r_day: float, result: ConformalResult,
    night_injection_flagged: bool = False, stuck_sensor_flagged: bool = False,
) -> str:
    """
    Layer 3.3: one-sided day classification.
    - "flagged_night_injection": A3-style fabrication (flag_night_injection)
    - "flagged_stuck_sensor": A8-style frozen replay (flag_stuck_sensor)
      Both above are PARALLEL, deterministic, guaranteed paths -- checked
      FIRST, independent of r_day/threshold. r_day's daylight-only
      aggregation is structurally blind to A3 (residual.py docstring),
      and 3.4's autoencoder was empirically verified NOT to catch A8 even
      with features engineered for it (see flag_stuck_sensor docstring)
      -- these two deterministic checks are the actual detection
      mechanism for their respective anomaly types, not a backup.
    - "flagged_overclaim": r_day > conformal threshold -> candidate fraud
    - "fault_underclaim":  r_day < 0 -> under-generation, operational
      fault channel (inverter outage, soiling, curtailment) -- NEVER a
      fraud flag
    - "normal": 0 <= r_day <= threshold

    NOTE: the r_day < 0 boundary is a simple sign rule, NOT a separately
    calibrated statistical threshold -- config.yaml defines no such
    threshold, and calibrating one from the same 5-day calib split would
    face the identical n=5 problem as the positive side, compounded by
    the twin's own documented positive bias (Ekta's recorded MBE is
    negative -- expected_q50 tends to UNDERSHOOT actual claims -- so
    r_day skews positive even on clean days; negative r_day is
    comparatively rare and plausibly more informative when it occurs,
    not just noise). If a formally calibrated fault threshold is needed
    later (e.g. for the N1-N3/N5-N6 hard-negative evaluation), that
    should draw on Layer 1's own fault flags (is_stuck, is_negative)
    and/or a dedicated fault-side calibration -- not be smuggled into
    this simple rule silently.
    """
    if night_injection_flagged:
        return "flagged_night_injection"
    if stuck_sensor_flagged:
        return "flagged_stuck_sensor"
    if r_day > result.threshold:
        return "flagged_overclaim"
    elif r_day < 0:
        return "fault_underclaim"
    return "normal"


def classify_days(
    daily_r_day: pd.Series, result: ConformalResult,
    night_injection_flagged: pd.Series | None = None,
    stuck_sensor_flagged: pd.Series | None = None,
) -> pd.Series:
    """
    Vectorized wrapper around classify_day. Both flag series, if given,
    must share daily_r_day's index (e.g. from residual.py's
    compute_daily_night_injection_flag() / compute_daily_stuck_sensor_flag())
    -- default to all-False if omitted, for backward compatibility.
    """
    if night_injection_flagged is None:
        night_injection_flagged = pd.Series(False, index=daily_r_day.index)
    if stuck_sensor_flagged is None:
        stuck_sensor_flagged = pd.Series(False, index=daily_r_day.index)
    return pd.Series(
        [
            classify_day(
                r, result,
                bool(night_injection_flagged.get(idx, False)),
                bool(stuck_sensor_flagged.get(idx, False)),
            )
            for idx, r in daily_r_day.items()
        ],
        index=daily_r_day.index,
    )


def tag_kt_regime(daily_kt: pd.Series, csi_bins: list[float]) -> pd.Series:
    """
    Tags each day with its Kt regime (clear / partly_cloudy / overcast),
    using the same csi_bins boundary as Layer 2's fallback and this
    project's other regime-stratified reporting. Diagnostic labeling
    ONLY -- the conformal threshold computed above is NOT regime-specific
    (see module docstring); do not present a per-regime breakdown of
    flag rates as if it carries a per-regime statistical guarantee.
    """
    labels = ["overcast", "partly_cloudy", "clear"]
    return pd.cut(daily_kt, bins=csi_bins, labels=labels, right=False)
