"""
src/layer3_anomaly/cusum.py

Layer 3.5: one-sided CUSUM sequential drift detection.

Catches the hardest realistic fraud case (Overview v2): a persistent
small inflation (A6_slow_drift, ~5-15% ramped in over 3-9 days per
config/anomalies.yaml) too small to trip any SINGLE-day conformal
threshold (3.2), but detectable via accumulated evidence over time.

REFERENCE STATISTICS -- DECISION, VERIFIED NOT ASSUMED:
The roadmap's initial suggestion was to draw CUSUM's reference mean/std
from the `train` split (20 days), reasoning that CUSUM's sequential
evidence-accumulation doesn't need the same one-shot order-statistic
guarantee that made calib's small n=5 a hard floor for 3.2. That
reasoning about the STATISTICAL MECHANISM is correct, but checking it
against real data (this session) found a DIFFERENT, disqualifying
problem: `train`'s r_day is systematically ~2x lower than pooled
calib+test_clean's, on BOTH plants:
    plant_1: train mean=0.047 vs pooled(calib+test_clean) mean=0.086
    plant_2: train mean=0.037 vs pooled(calib+test_clean) mean=0.103
This is the same in-sample-optimism pattern already found and corrected
for 3.4's autoencoder (the twin fits its own training data better than
held-out data -- see docs/LAYER2_HANDOFF.md Section 4's train/test R^2
gaps for the same root cause). Using train stats as the CUSUM reference
would set the "normal" baseline roughly half of what genuinely clean
held-out data actually looks like, causing severe over-triggering.

RESOLUTION: reference mean/std computed from pooled calib+test_clean
(n=10 days) -- smaller sample than train, but not subject to the same
optimism bias. n=10 is still modest; report the resulting reference
statistics' uncertainty honestly in the paper (same standard as 3.2's
achievable-alpha reporting), don't silently treat them as exact.

ONE-SIDED (Overview v2 SS2.6): only accumulates evidence for persistent
POSITIVE bias (over-claiming). A sustained negative shift (persistent
under-generation) is a fault-channel signal, not fraud -- symmetric with
3.2/3.3's one-sided design, for the same reason (N1/N2/N3/N5/N6 hard
negatives are under-generation patterns that a two-sided test would
wrongly flag).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class CusumReference:
    mu_ref: float
    sigma_ref: float
    n_ref: int


def calibrate_h_from_clean_data(
    reference_r_day: pd.Series, k: float, window_len: int,
    target_fpr: float = 0.05, h_candidates: list[float] = None,
    n_bootstrap: int = 2000, seed: int = 42,
) -> dict:
    """
    Calibrates h using ONLY clean reference data -- NEVER injected/labeled
    data. This is the calib-only discipline already applied to 3.2
    (conformal never fits on test_injected); the same principle applies
    here: picking h because it makes known A6 cases flag would be fitting
    the detector to the answer key, making any later detection-rate
    number reported in the paper meaningless (reverse-engineered from
    ground truth rather than validated against it).

    Method: bootstrap-resample reference_r_day (the same pooled
    calib+test_clean pool used for mu_ref/sigma_ref) into many synthetic
    `window_len`-day clean sequences, run CUSUM on each, and measure how
    often S_t crosses each h_candidate purely by chance. Pick the
    smallest h whose empirical false-positive rate is <= target_fpr --
    the most SENSITIVE (lowest) h that still respects the target FPR,
    since a stricter comparison than necessary would just make CUSUM
    right back to whatever(sluggish) behavior motivated this recalibration.

    Returns a dict: {h_candidate: empirical_fpr for each candidate},
    plus 'recommended_h': the chosen value.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    values = reference_r_day.dropna().values
    reference = fit_cusum_reference(reference_r_day)

    if h_candidates is None:
        h_candidates = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

    fpr_by_h = {}
    for h_candidate in h_candidates:
        false_positive_count = 0
        for _ in range(n_bootstrap):
            synthetic_seq = rng.choice(values, size=window_len, replace=True)
            s = 0.0
            triggered = False
            for r in synthetic_seq:
                z = (r - reference.mu_ref) / reference.sigma_ref
                s = max(0.0, s + z - k)
                if s > h_candidate:
                    triggered = True
                    break
            false_positive_count += triggered
        fpr_by_h[h_candidate] = false_positive_count / n_bootstrap

    eligible = [h for h, fpr in fpr_by_h.items() if fpr <= target_fpr]
    recommended_h = min(eligible) if eligible else max(h_candidates)

    return {"fpr_by_h": fpr_by_h, "recommended_h": recommended_h, "target_fpr": target_fpr}


def fit_cusum_reference(reference_r_day: pd.Series) -> CusumReference:
    """
    reference_r_day: r_day values from POOLED calib+test_clean (NOT
    train -- see module docstring). Caller is responsible for pooling;
    this function just computes mean/std of whatever it's given.
    """
    values = reference_r_day.dropna()
    n = len(values)
    if n < 2:
        raise ValueError("Need at least 2 reference samples to compute std.")
    return CusumReference(mu_ref=float(values.mean()), sigma_ref=float(values.std(ddof=1)), n_ref=n)


@dataclass
class CusumResult:
    dates: list
    r_day_values: list
    s_values: list          # cumulative statistic S_t, one per date, in order
    flagged: list            # bool, True where S_t > h
    k: float
    h: float
    reference: CusumReference


def run_cusum(
    r_day_sequence: pd.Series, reference: CusumReference, k: float, h: float,
) -> CusumResult:
    """
    One-sided upper CUSUM:
        z_t = (r_day(t) - mu_ref) / sigma_ref          -- standardized
        S_0 = 0
        S_t = max(0, S_{t-1} + z_t - k)                -- k = slack, sigma units
        flag day t if S_t > h

    r_day_sequence MUST be in chronological order (CUSUM is sequential --
    passing an unordered or shuffled series silently produces a
    meaningless result, since S_t depends on accumulated history, not
    just the current value). This function does NOT sort for you --
    sort by date before calling, and verify visually before trusting output.

    k (config.yaml's layer3.cusum.k, default 0.25): the process is
    allowed to run `k` sigma above the reference mean indefinitely
    without triggering -- this is slack for ordinary day-to-day noise,
    not itself a detection threshold.

    h (config.yaml's layer3.cusum.h, default 5.0): decision interval --
    once accumulated evidence S_t exceeds this, the day is flagged.
    """
    dates = list(r_day_sequence.index)
    values = list(r_day_sequence.values)

    s_values = []
    flagged = []
    s_prev = 0.0
    for r in values:
        z = (r - reference.mu_ref) / reference.sigma_ref
        s_t = max(0.0, s_prev + z - k)
        s_values.append(s_t)
        flagged.append(s_t > h)
        s_prev = s_t

    return CusumResult(
        dates=dates, r_day_values=values, s_values=s_values, flagged=flagged,
        k=k, h=h, reference=reference,
    )


def cusum_result_to_frame(result: CusumResult) -> pd.DataFrame:
    """Convenience: CusumResult -> DataFrame, one row per date, for easy inspection/joining."""
    return pd.DataFrame(
        {"r_day": result.r_day_values, "s_t": result.s_values, "flagged": result.flagged},
        index=result.dates,
    )
