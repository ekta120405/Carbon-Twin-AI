"""
src/layer3_anomaly/escalation.py

Secondary-evidence escalation -- built after this project's own
evaluation work (src/evaluation/_experiment_ae_runlength.py,
_experiment_cusum_smart.py, _experiment_split_validate.py) found that the
autoencoder (3.4) and CUSUM (3.5) each carry real detection signal that
classify_days() (3.3) doesn't use, but that BLINDLY trusting either one
(a flat OR against the whole day pool) costs more in new false positives
than the extra recall is worth -- see docs/TASK7_HANDOFF.md.

WHAT THIS ACTUALLY DOES: upgrades a day already classified "normal" by
classify_days() to "flagged_overclaim" ONLY if BOTH of the following
hold:
  1. Its r_day sits close below the main conformal threshold (within
     `band_width` of it, e.g. the top 30% of the gap between 0 and the
     threshold) -- i.e. it was already a near-miss on the primary test,
     not an arbitrary clean day.
  2. At least one secondary signal corroborates: the autoencoder's
     calibrated run-length flag (src/layer3_anomaly/autoencoder.py's
     compute_daily_ae_flag), or CUSUM's own running statistic exceeding
     `cusum_frac * h`.

Never touches flagged_night_injection, flagged_stuck_sensor, or
fault_underclaim -- this only ever upgrades "normal" to
"flagged_overclaim", the one flag type this escalation logic actually
has evidence for.

CALIBRATION HONESTY NOTE (the reason this module exists as a small,
separate function rather than a hardcoded threshold change): `band_width`
and `cusum_frac` were chosen via honest split validation (tune on
replicates 0-19, verify on the untouched 20-29), NOT by testing directly
against the full labeled set used for final scoring -- an earlier version
of this experiment did exactly that and reported a "free" (zero
specificity cost) result that did not survive when properly validated.
The real, disclosed result: F1 improves substantially on held-out data,
but at a real, non-zero specificity cost. See config.yaml's
layer3.escalation section and docs/TASK7_HANDOFF.md for the full story --
do not re-tune these two numbers by testing directly against
data/ground_truth/ without repeating that same tune/holdout discipline.
"""
from __future__ import annotations

import pandas as pd


def apply_secondary_evidence_escalation(
    daily: pd.DataFrame,
    labels: pd.Series,
    conformal_threshold: float,
    cusum_s_t: pd.Series,
    cusum_h: float,
    ae_day_flag: pd.Series,
    band_width: float,
    cusum_frac: float,
) -> pd.Series:
    """
    `daily` and `labels` share the date index produced by
    residual.compute_daily_energy_residual() / conformal.classify_days().
    `cusum_s_t` is cusum.run_cusum()'s per-day statistic (e.g. harness.py's
    cusum_frame['cusum_s_t']); `ae_day_flag` is
    autoencoder.compute_daily_ae_flag()'s output. Both are reindexed to
    `daily`'s index defensively (missing -> not corroborating), since a
    caller might legitimately not have CUSUM/AE results for every day
    (e.g. a window too short for CUSUM to run meaningfully).

    Returns a NEW labels Series -- does not mutate the input.
    """
    # Strict "<" on the upper bound, not "<=": a day sitting EXACTLY at the
    # threshold is, for at least one real plant, the very calib day whose
    # r_day *defined* that threshold (threshold = max(calib r_day) --
    # conformal.py's fit_conformal_threshold). That day cannot legitimately
    # be "suspicious" -- it's the most extreme example of NORMAL behavior
    # the threshold was calibrated from, not evidence against it. Found via
    # a real regression: plant_1's real_clean block flagged its own
    # threshold-defining calib day before this was tightened to "<".
    near_threshold = (daily["r_day"] > band_width * conformal_threshold) & (daily["r_day"] < conformal_threshold)

    cusum_elevated = cusum_s_t.reindex(daily.index).fillna(0.0) > (cusum_frac * cusum_h)
    ae_flagged = ae_day_flag.reindex(daily.index, fill_value=False).astype(bool)

    corroborated = (cusum_elevated | ae_flagged) & near_threshold

    upgraded = labels.copy()
    upgrade_mask = (labels == "normal") & corroborated
    upgraded[upgrade_mask] = "flagged_overclaim"
    return upgraded
