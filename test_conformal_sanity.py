"""
test_conformal_sanity.py (throwaway diagnostic) -- verifies
fit_conformal_threshold() against hand-computed values in BOTH regimes:
n below the nominal-alpha floor (this project's real n=5 case) and n
above it (synthetic, to prove the proper order-statistic branch is
correct even though real calib data won't exercise it).
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from src.layer3_anomaly.conformal import (
    fit_conformal_threshold, min_n_for_nominal_alpha, compute_achievable_alpha, flag_days,
)

TOL = 1e-9
all_ok = True


def check(label, actual, expected, tol=TOL):
    global all_ok
    ok = abs(actual - expected) < tol
    all_ok &= ok
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: expected {expected}, got {actual}")


# --- floor formula itself ---
# k = ceil((n+1)(1-alpha)) <= n  =>  n >= 1/alpha - 1
# at alpha=0.05: n >= 19
check("min_n_for_nominal_alpha(0.05)", min_n_for_nominal_alpha(0.05), 19)
check("min_n_for_nominal_alpha(0.10)", min_n_for_nominal_alpha(0.10), 9)

# --- small-n branch (this project's real case: n=5) ---
calib_small = pd.Series([0.01, 0.08, 0.02, 0.05, 0.03])  # deliberately unsorted
result_small = fit_conformal_threshold(calib_small, alpha_nominal=0.05)
check("small-n threshold (should be max = 0.08)", result_small.threshold, 0.08)
check("small-n alpha_achievable (1/(5+1))", result_small.alpha_achievable, 1 / 6)
print(f"  n_calib={result_small.n_calib}, alpha_nominal={result_small.alpha_nominal}, "
      f"alpha_achievable={result_small.alpha_achievable:.4f}")
print()

# flagging check: a day with r_day > 0.08 should be flagged, one below should not
test_days = pd.Series([0.09, 0.08, 0.07, -0.5], index=["above", "equal", "below", "very_negative"])
flags = flag_days(test_days, result_small)
check("flag 'above' (0.09 > 0.08)", float(flags["above"]), 1.0)
check("flag 'equal' (0.08 > 0.08, strict) -> not flagged", float(flags["equal"]), 0.0)
check("flag 'below' (0.07 > 0.08) -> not flagged", float(flags["below"]), 0.0)
check("flag 'very_negative' -> not flagged (fault channel, not fraud)", float(flags["very_negative"]), 0.0)
print()

# --- large-n branch (synthetic, n=20 clears the floor at alpha=0.05) ---
np.random.seed(42)
calib_large = pd.Series(np.random.normal(loc=0.0, scale=0.05, size=20))
result_large = fit_conformal_threshold(calib_large, alpha_nominal=0.05)
# hand-verify: k = ceil(21 * 0.95) = ceil(19.95) = 20 -> the max again at n=20
# (n=20 is exactly at the floor, so it still lands on the max -- use n=25 to get
# a threshold that ISN'T just the max, proving the general formula, not a fluke)
calib_large25 = pd.Series(np.random.normal(loc=0.0, scale=0.05, size=25))
result_large25 = fit_conformal_threshold(calib_large25, alpha_nominal=0.05)
sorted_vals = np.sort(calib_large25.values)
import math
expected_k = math.ceil((25 + 1) * (1 - 0.05))  # = ceil(24.7) = 25 -> still max at n=25!
print(f"  n=25: expected_k={expected_k} (out of 25) -- still the max, need larger n to see interior order stat")

# use n=100 to definitively land on an INTERIOR order statistic, not the max
calib_100 = pd.Series(np.random.normal(loc=0.0, scale=0.05, size=100))
result_100 = fit_conformal_threshold(calib_100, alpha_nominal=0.05)
sorted_100 = np.sort(calib_100.values)
expected_k_100 = math.ceil(101 * 0.95)  # = ceil(95.95) = 96
expected_threshold_100 = sorted_100[expected_k_100 - 1]
check(f"n=100 threshold matches hand-computed {expected_k_100}th order statistic",
      result_100.threshold, expected_threshold_100)
check("n=100 alpha_achievable == nominal (0.05), floor cleared", result_100.alpha_achievable, 0.05)
print(f"  n=100: threshold={result_100.threshold:.5f} (NOT the max={sorted_100[-1]:.5f} -- "
      f"confirms interior order statistic, not a max-only hack)")

print()
print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
sys.exit(0 if all_ok else 1)
