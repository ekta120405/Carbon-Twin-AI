"""
test_residual_sanity_v2.py (throwaway diagnostic -- verifies the daily
aggregation math in the REAL residual.py against hand-computed values,
independent of the twin/model, before trusting it further).
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

import pandas as pd

from src.layer3_anomaly.residual import compute_daily_energy_residual

CAPACITY_KW = 1000.0
TOL = 1e-9


def _rows(date: str, ac_power: float, expected_q50: float, n: int = 4, is_daylight: bool = True):
    idx = pd.date_range(f"{date} 06:00", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {"ac_power": [ac_power] * n, "expected_q50": [expected_q50] * n, "is_daylight": [is_daylight] * n},
        index=idx,
    )


def build_synthetic_df() -> pd.DataFrame:
    # day 1: claimed=300, expected=100 -> excess=200kW/row * 4 rows * 0.25h = 200 kWh
    #        daylight_hours=1h -> r_day = 200 / (1000 * 1) = 0.2
    day1 = _rows("2026-01-01", ac_power=300.0, expected_q50=100.0)
    # absurd night-time claim on day 1 -- must NOT affect day 1's total
    day1_night = pd.DataFrame(
        {"ac_power": [5000.0], "expected_q50": [0.0], "is_daylight": [False]},
        index=pd.DatetimeIndex(["2026-01-01 23:00+00:00"]),
    )
    # day 2: claimed=250, expected=250 -> excess=0 -> r_day = 0.0 (perfect agreement)
    day2 = _rows("2026-01-02", ac_power=250.0, expected_q50=250.0)
    # day 3: claimed=80, expected=100 -> excess=-20kW/row*4*0.25=-20kWh -> r_day = -20/(1000*1) = -0.02
    day3 = _rows("2026-01-03", ac_power=80.0, expected_q50=100.0)

    return pd.concat([day1, day1_night, day2, day3]).sort_index()


def run():
    df = build_synthetic_df()
    daily = compute_daily_energy_residual(df, ac_nameplate_kw=CAPACITY_KW)

    print(daily)
    print()

    expected_r_day = {
        pd.Timestamp("2026-01-01").date(): 0.2,
        pd.Timestamp("2026-01-02").date(): 0.0,
        pd.Timestamp("2026-01-03").date(): -0.02,
    }

    all_ok = True
    for day, expected_val in expected_r_day.items():
        actual = daily.loc[day, "r_day"]
        ok = abs(actual - expected_val) < TOL
        all_ok &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {day}: expected r_day={expected_val}, got {actual}")

    # explicit check: day 1's excess_energy_kwh must be exactly 200, not inflated
    # by the 5000kW night spike (which would add 5000*0.25=1250 kWh if leaked in)
    day1_excess = daily.loc[pd.Timestamp("2026-01-01").date(), "excess_energy_kwh"]
    night_excluded_ok = abs(day1_excess - 200.0) < TOL
    all_ok &= night_excluded_ok
    print(f"[{'PASS' if night_excluded_ok else 'FAIL'}] night-time spike excluded (excess_energy_kwh == 200.0, not 1450.0)")

    print()
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return all_ok


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
