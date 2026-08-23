"""
test_cusum_sanity.py (throwaway diagnostic) -- verifies run_cusum()'s
trajectory against a hand-computed sequence before trusting it on real
data. See inline comments for the manual S_t derivation.
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

import pandas as pd

from src.layer3_anomaly.cusum import CusumReference, run_cusum

TOL = 1e-9


def run():
    ref = CusumReference(mu_ref=0.0, sigma_ref=1.0, n_ref=100)
    seq = pd.Series(
        [0.2, 0.6, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
        index=pd.date_range("2026-01-01", periods=8),
    )
    result = run_cusum(seq, ref, k=0.5, h=3.0)

    # hand computation (k=0.5, mu_ref=0, sigma_ref=1, so z_t == r_day(t)):
    # S0=0
    # t1: S=max(0, 0+0.2-0.5)=0.0
    # t2: S=max(0, 0.0+0.6-0.5)=0.1
    # t3: S=max(0, 0.1+0.9-0.5)=0.5
    # t4: S=max(0, 0.5+0.9-0.5)=0.9
    # t5: S=max(0, 0.9+0.9-0.5)=1.3
    # t6: S=1.7
    # t7: S=2.1
    # t8: S=2.5  (never crosses h=3.0)
    expected_s = [0.0, 0.1, 0.5, 0.9, 1.3, 1.7, 2.1, 2.5]

    all_ok = True
    for i, (got, exp) in enumerate(zip(result.s_values, expected_s)):
        ok = abs(got - exp) < TOL
        all_ok &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] t{i+1}: expected S={exp}, got {got:.4f}")

    no_flag_ok = not any(result.flagged)
    all_ok &= no_flag_ok
    print(f"[{'PASS' if no_flag_ok else 'FAIL'}] none flagged (max S=2.5 < h=3.0)")

    print()
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return all_ok


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
