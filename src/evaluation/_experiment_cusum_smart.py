"""
EXPERIMENTAL, not wired into production. Applies the SAME "smart
escalation" pattern that worked for the autoencoder (src/evaluation/
_experiment_ae_runlength.py) to CUSUM instead: CUSUM alone as a blanket OR
was already tested and found to be a net loss (more false positives than
the recall gain was worth). This tests whether only trusting CUSUM's
opinion on days that are ALREADY borderline on the main conformal test
(instead of a blanket OR) avoids that cost while still recovering some of
CUSUM's real signal for A6_slow_drift specifically.

Also tests combining this with the already-validated, clean-calibrated
autoencoder fix, to see whether the two gains stack additively (they
should, since they're independent corroborating signals feeding the same
kind of "only escalate a borderline day" rule).

CUSUM elevation threshold (what counts as "CUSUM also thinks this is
elevated") is swept as a fraction of CUSUM's own calibrated h, and picked
by checking its effect on CLEAN/hard-negative specificity first (the same
calibration discipline as everything else in this project), not by
directly maximizing against fraud labels.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

import pandas as pd

from dashboard.lib.pipeline import PLANTS, load_plant_base
from src.evaluation.harness import score_all_replicates, N_REPLICATES, _h_for_window
from src.evaluation.benchmark import merge_with_ground_truth, _metrics_block
from src.evaluation.ablations import train_plant_autoencoder, _full_ts_table
from src.evaluation._experiment_ae_runlength import ae_day_flag_with_runlength
from src.layer2_twin.physics_anchor import compute_physics_anchor
from src.layer3_anomaly.residual import build_residual_table
from src.layer3_anomaly.autoencoder import (
    build_autoencoder_features, compute_reconstruction_error,
    AUTOENCODER_META_COLS, AUTOENCODER_DIFF_COLS,
)


def calibrate_ae_run_length(plant: str, base: dict, ae_result, ae_threshold) -> int:
    clean_dates = base["calib_dates"].union(base["test_clean_dates"])
    ts_full = _full_ts_table(plant, base)
    tfc = [c for c in ae_result.input_cols if c not in AUTOENCODER_META_COLS + AUTOENCODER_DIFF_COLS]
    feats_full = build_autoencoder_features(ts_full, tfc)
    err_full = compute_reconstruction_error(feats_full, ae_result)
    daylight_idx = ts_full.index[ts_full["is_daylight"].astype(bool)]
    err_dl = err_full[err_full.index.isin(daylight_idx)].sort_index()
    dates_s = pd.Series(err_dl.index.date, index=err_dl.index)
    err_clean = err_dl[dates_s.isin(clean_dates)]
    over_clean = err_clean > ae_threshold
    rid = (~over_clean).cumsum()
    rsz = over_clean.groupby(rid).transform("sum") * over_clean
    return int(rsz.max()) + 1


def main():
    plant_dfs = {}
    for plant in PLANTS:
        base = load_plant_base(plant)
        twin, capacity_kw, meta = base["twin"], base["capacity_kw"], base["meta"]
        threshold = base["conformal_result"].threshold

        ae_result, ae_threshold = train_plant_autoencoder(plant, base)
        calibrated_rl = calibrate_ae_run_length(plant, base, ae_result, ae_threshold)
        print(f"[{plant}] AE calibrated run_length={calibrated_rl}")

        scored = score_all_replicates(plant, base=base).reset_index()
        df = merge_with_ground_truth(scored, plant)
        df["near_threshold"] = (df["r_day"] > 0.5 * threshold) & (df["r_day"] <= threshold)
        df["near_threshold_tight"] = (df["r_day"] > 0.8 * threshold) & (df["r_day"] <= threshold)

        ae_flags, cusum_s = {}, {}
        for rep_id in range(N_REPLICATES):
            rep_df = pd.read_parquet(ROOT / "data" / "processed" / "injected" / plant / f"replicate_{rep_id:03d}.parquet")
            anchor = compute_physics_anchor(rep_df, meta["latitude"], meta["longitude"], capacity_kw)
            dfj = rep_df.join(anchor)
            ts = build_residual_table(twin, dfj["physics_ac_kw"], dfj, capacity_kw, twin.feature_cols)
            day_flag = ae_day_flag_with_runlength(ts, ae_result, ae_threshold, calibrated_rl)
            for d, flagged in day_flag.items():
                ae_flags[(rep_id, d)] = bool(flagged)

        df["ae_cal"] = df.apply(lambda r: ae_flags.get((r["replicate"], r["date"]), False), axis=1)
        df["ae_smart"] = df["detected"] | (df["ae_cal"] & df["near_threshold"])

        # cusum_s_t is already computed by score_all_replicates -> merged via scored/df.
        # cusum_h is a CONSTANT for this plant (every replicate is the same 9-day
        # window length) -- harness.py recalibrates h fresh per distinct window
        # length, not per replicate, so this one call covers all 30 replicates.
        assert "cusum_s_t" in df.columns, sorted(df.columns)
        window_len = 9  # every injected replicate is a 9-day window
        cusum_h = _h_for_window(base, window_len)
        print(f"[{plant}] cusum_h (window_len={window_len}) = {cusum_h}")

        for frac in [0.3, 0.5, 0.7, 0.9]:
            elevated = df["cusum_s_t"] > (frac * cusum_h)
            col = f"cusum_smart_f{frac}"
            df[col] = df["detected"] | (elevated & df["near_threshold"])
            combined_col = f"combined_f{frac}"
            df[combined_col] = df["ae_smart"] | (elevated & df["near_threshold"])
            # tight-band variants: only escalate on the narrower 80-100% band
            col_t = f"cusum_smart_tight_f{frac}"
            df[col_t] = df["detected"] | (elevated & df["near_threshold_tight"])
            combined_col_t = f"combined_tight_f{frac}"
            df[combined_col_t] = df["ae_smart"] | (elevated & df["near_threshold_tight"])

        plant_dfs[plant] = df

    pooled = pd.concat(plant_dfs.values())
    baseline = _metrics_block(pooled, "detected")
    print("\n=== POOLED ===")
    print(f"baseline: recall={baseline['overall']['recall']:.3f} precision={baseline['overall']['precision']:.3f} "
          f"f1={baseline['overall']['f1']:.3f} specificity={baseline['hard_negative_overall']['specificity']:.3f}")

    ae_only = _metrics_block(pooled, "ae_smart")
    print(f"ae_smart only: recall={ae_only['overall']['recall']:.3f} precision={ae_only['overall']['precision']:.3f} "
          f"f1={ae_only['overall']['f1']:.3f} specificity={ae_only['hard_negative_overall']['specificity']:.3f}")

    results = {}
    for frac in [0.3, 0.5, 0.7, 0.9]:
        for variant in [f"cusum_smart_f{frac}", f"combined_f{frac}",
                        f"cusum_smart_tight_f{frac}", f"combined_tight_f{frac}"]:
            m = _metrics_block(pooled, variant)
            results[variant] = dict(overall=m["overall"], specificity=m["hard_negative_overall"]["specificity"])
            a6 = m["per_type"].get("A6_slow_drift", {})
            print(f"{variant}: recall={m['overall']['recall']:.3f} precision={m['overall']['precision']:.3f} "
                  f"f1={m['overall']['f1']:.3f} specificity={m['hard_negative_overall']['specificity']:.3f} "
                  f"| A6 recall={a6.get('recall')} (n={a6.get('n')})")

    with open(ROOT / "reports" / "tables" / "_experiment_cusum_smart.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
