"""
src/evaluation/mdo_curve.py

Task 7.2: Minimum Detectable Overstatement (MDO) curve.

Per TASK7_FULL_ROADMAP_FOR_CLAUDE_CODE.md SS0.2: build 3 windows (15-min,
hourly, daily) x 8 magnitude levels (config/anomalies.yaml's
evaluation_sweep.magnitudes). weekly/monthly are NOT constructible from the
frozen injector's 9-day eligible pool (9/7 ~= 1.3 weekly windows; 0 monthly
windows) and are reported as infeasible below with that exact reasoning,
not silently dropped.

WHAT "WINDOW" MEANS HERE, stated precisely -- this project's architecture
has exactly ONE statistical decision point, the DAILY conformal test
(Layer 3.2/3.3, residual.py's r_day). There is no separate 15-min- or
hourly-level statistical test anywhere in Layer 3. "Window" therefore means
how much of a day's timesteps the injected constant overstatement is
applied to before that single daily test runs, not a different decision
level:
    daily:  ALL of the day's rows inflated by `magnitude` (A1-equivalent --
            src/injection/taxonomy.py's apply_a1_constant_overclaim())
    hourly: only the single local hour containing the day's peak expected
            output (its 4 consecutive 15-min slots) inflated
    15min:  only the single daylight timestep with the day's peak expected
            output inflated
Detection is always measured identically: does the resulting daily r_day
exceed the plant's ALREADY-calibrated conformal threshold (from
dashboard.lib.pipeline.load_plant_base -- fit once on real calib data,
NEVER recalibrated here, per TASK7_FULL_ROADMAP SS1.6)? A narrower window
diluted into the same daily aggregate is mechanically harder to detect --
this dilution effect IS the detection-power-vs-localisation trade-off
Overview v2 SS9.3 asks to characterize, not an approximation of it.

"Peak" timestamp/hour is located once per day from the UNMODIFIED twin
prediction (expected_q50) -- injection never touches independent weather
features, so the twin's prediction for a given day is identical regardless
of which magnitude/window is later applied to that day's claimed power;
computing it once and reusing it across all 24 (window, magnitude) cells
avoids 24x redundant twin inference per day.

Detection is measured on r_day > threshold directly (the core conformal
test in isolation), not on classify_days()'s flag_type label -- this MDO
sweep is purpose-built to characterize the conformal/magnitude mechanism
specifically, and routing through classify_days would risk an incidental
guardrail flag (night-injection, stuck-sensor) muddying a measurement
that's supposed to isolate one detector.

Injected into HELD-OUT clean data only: each plant's real test_clean days
(5/plant, 10 pooled) -- never calib (that's what fit the threshold being
tested against; injecting into the days that produced the threshold and
then testing against that same threshold would be circular).

For the "daily" window, the injected magnitude is applied via the frozen
apply_a1_constant_overclaim() unmodified, called with a degenerate
delta_range=[magnitude, magnitude] so its rng.uniform() collapses to
exactly the pinned magnitude -- per SS0.2's explicit instruction not to
build new OVERCLAIM MECHANICS, reuse the existing one. The "hourly" and
"15min" windows use the identical multiplicative mechanism
(claimed *= (1 + magnitude)) but applied to a different row subset than
any existing taxonomy function targets (A2's hour window is a fixed,
broad midday range for a different purpose, not a single peak-anchored
hour) -- this row-selection logic is new EVALUATION code, not a new
ANOMALY TYPE, and does not touch or extend the frozen injector
(injector-v1.0) in any way.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from dashboard.lib.pipeline import PLANTS, load_plant_base
from src.evaluation.harness import score_dataframe

WINDOWS = ["15min", "hourly", "daily"]

WEEKLY_MONTHLY_INFEASIBILITY_NOTE = (
    "weekly and monthly aggregation windows were NOT computed. The "
    "frozen injector's eligible evaluation pool (config/anomalies.yaml, "
    "eligible_days: [26, 34]) spans exactly 9 contiguous days per "
    "replicate. A 'weekly' window is barely constructible (9/7 ~= 1.3 "
    "windows); 'monthly' is not constructible at all within a single "
    "replicate. Relaxing the frozen injector's 9-day structure to force "
    "these windows into existence would mean building new injection "
    "machinery, out of scope for an evaluation task "
    "(TASK7_FULL_ROADMAP_FOR_CLAUDE_CODE.md SS0.2). This is reported as "
    "an honest infeasibility finding, not a silent omission."
)


def _get_test_clean_block(plant: str, base: dict) -> pd.DataFrame:
    df_full = pd.read_parquet(ROOT / "data" / "processed" / f"{plant}.parquet")
    mask = pd.Series(df_full.index.date, index=df_full.index).isin(base["test_clean_dates"])
    return df_full[mask].copy()


def _compute_peak_timestamps(ts_table: pd.DataFrame) -> dict:
    """One daylight timestamp per date -- the row with the largest
    expected_q50, i.e. the day's peak expected-output moment. Computed
    from the UNMODIFIED (real, pre-injection) prediction."""
    daylight = ts_table[ts_table["is_daylight"].astype(bool)]
    dates = pd.Series(daylight.index.date, index=daylight.index)
    peak = {}
    for d, grp in daylight.groupby(dates):
        peak[d] = grp["expected_q50"].idxmax()
    return peak


def _inject_window(day_ac_power: pd.Series, peak_ts: pd.Timestamp, window: str, magnitude: float) -> pd.Series:
    modified = day_ac_power.copy()
    if window == "daily":
        modified = day_ac_power * (1 + magnitude)  # A1-equivalent: whole day slice
    elif window == "hourly":
        local_hour = peak_ts.tz_convert("Asia/Kolkata").hour
        row_local_hours = day_ac_power.index.tz_convert("Asia/Kolkata").hour
        mask = row_local_hours == local_hour
        modified[mask] = modified[mask] * (1 + magnitude)
    elif window == "15min":
        modified.loc[peak_ts] = modified.loc[peak_ts] * (1 + magnitude)
    else:
        raise ValueError(f"Unknown window: {window!r}")
    return modified


def run_mdo_sweep() -> tuple[pd.DataFrame, dict]:
    with open(ROOT / "config" / "anomalies.yaml") as f:
        acfg = yaml.safe_load(f)
    magnitudes = acfg["evaluation_sweep"]["magnitudes"]

    rows = []
    excluded_baseline_fp: dict = {}
    for plant in PLANTS:
        base = load_plant_base(plant)
        twin, capacity_kw, meta = base["twin"], base["capacity_kw"], base["meta"]
        threshold = base["conformal_result"].threshold

        block = _get_test_clean_block(plant, base)
        baseline = score_dataframe(block, twin, base, capacity_kw, meta)
        peak_ts = _compute_peak_timestamps(baseline["ts_table"])
        all_dates = sorted(peak_ts.keys())

        # EXCLUDE any day already flagged at zero injection (baseline r_day >
        # threshold BEFORE any magnitude is applied) -- injecting a magnitude
        # into an already-flagged day doesn't test detection of that magnitude
        # at all, it's a pre-existing false positive contaminating the curve.
        # Found empirically while building this sweep: plant_2's real
        # 2020-06-08 test_clean day has baseline r_day=0.248 > threshold=0.208
        # with ZERO injection -- the same documented plant_2 night-injection
        # false-positive overlap noted in LAYER3_HANDOFF.md SS3 (~0.36% of
        # genuinely-dark clean rows). Silently including it would put a
        # spurious 1/n detection-rate floor under every (window, magnitude)
        # cell, masking whether the actually-injected magnitude does anything.
        baseline_daily = baseline["daily"]
        already_flagged = [d for d in all_dates if float(baseline_daily.loc[d, "r_day"]) > threshold]
        dates = [d for d in all_dates if d not in already_flagged]
        if already_flagged:
            excluded_baseline_fp[plant] = [str(d) for d in already_flagged]

        for window in WINDOWS:
            for magnitude in magnitudes:
                modified_block = block.copy()
                date_series = pd.Series(modified_block.index.date, index=modified_block.index)
                for d in dates:
                    day_mask = date_series == d
                    modified_block.loc[day_mask, "ac_power"] = _inject_window(
                        modified_block.loc[day_mask, "ac_power"], peak_ts[d], window, magnitude,
                    ).values

                result = score_dataframe(modified_block, twin, base, capacity_kw, meta)
                daily = result["daily"]
                for d in dates:
                    r_day = float(daily.loc[d, "r_day"])
                    rows.append(dict(
                        plant=plant, window=window, magnitude=magnitude, date=str(d),
                        r_day=r_day, threshold=threshold, detected=bool(r_day > threshold),
                    ))

    return pd.DataFrame(rows), excluded_baseline_fp


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    grp = raw.groupby(["window", "magnitude"]).agg(n=("detected", "size"), n_detected=("detected", "sum"))
    grp["detection_rate"] = grp["n_detected"] / grp["n"]
    return grp.reset_index()


def plot_mdo_curve(summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    # small vertical offsets so 15min/hourly don't visually overlap when both
    # are exactly 0.0 across the whole tested range -- a real, correct result
    # (see module docstring), not a plotting artifact to hide.
    offsets = {"15min": -0.015, "hourly": 0.015, "daily": 0.0}
    markers = {"15min": "s", "hourly": "^", "daily": "o"}
    for window in WINDOWS:
        sub = summary[summary["window"] == window].sort_values("magnitude")
        ax.plot(
            sub["magnitude"] * 100, sub["detection_rate"] + offsets[window],
            marker=markers[window], label=window,
        )
    ax.set_xlabel("Injected overstatement magnitude (% of claimed power)")
    ax.set_ylabel("Detection rate (fraction of held-out clean days flagged)")
    n_per_cell = int(summary["n"].iloc[0]) if len(summary) else 0
    ax.set_title(f"Minimum Detectable Overstatement (n={n_per_cell} clean days/cell, pooled)")
    ax.set_ylim(-0.08, 1.08)
    ax.legend(title="Window")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    raw, excluded_baseline_fp = run_mdo_sweep()
    summary = summarize(raw)

    tables_dir = ROOT / "reports" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out = dict(
        windows_computed=WINDOWS,
        weekly_monthly_infeasibility_note=WEEKLY_MONTHLY_INFEASIBILITY_NOTE,
        excluded_baseline_false_positives=excluded_baseline_fp,
        excluded_baseline_false_positives_note=(
            "Days excluded from the sweep pool because they were ALREADY "
            "flagged (r_day > threshold) at zero injection -- injecting a "
            "magnitude into an already-flagged day tests nothing about that "
            "magnitude. See mdo_curve.py's run_mdo_sweep() docstring."
        ),
        summary=summary.to_dict("records"),
        raw=raw.to_dict("records"),
    )
    with open(tables_dir / "mdo_curve.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Wrote {tables_dir / 'mdo_curve.json'}")

    fig_path = ROOT / "reports" / "figures" / "mdo_curve.png"
    plot_mdo_curve(summary, fig_path)
    print(f"Wrote {fig_path}")

    n_per_cell = int(summary["n"].iloc[0]) if len(summary) else 0
    print(f"\n=== Detection rate summary (pooled, n={n_per_cell} days/cell) ===")
    if excluded_baseline_fp:
        print(f"Excluded pre-existing baseline false positives: {excluded_baseline_fp}")
    pivot = summary.pivot(index="magnitude", columns="window", values="detection_rate")
    print(pivot.to_string())


if __name__ == "__main__":
    main()
