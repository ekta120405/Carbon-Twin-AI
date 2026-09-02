"""
EXPERIMENTAL, not wired into production. Tests whether requiring a RUN of
N consecutive elevated timesteps (instead of ANY single one) for the
autoencoder's day-level flag improves things, per the untried idea noted
in docs/HOW_THE_SYSTEM_WORKS.md. Also tests a "smart ensemble" variant:
only let the autoencoder's opinion escalate a day that's already
borderline on the main conformal test, instead of a blanket OR.
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
from src.evaluation.harness import score_all_replicates, N_REPLICATES
from src.evaluation.benchmark import merge_with_ground_truth, _metrics_block
from src.evaluation.ablations import (
    train_plant_autoencoder, _full_ts_table,
)
from src.layer2_twin.physics_anchor import compute_physics_anchor
from src.layer3_anomaly.residual import build_residual_table
from src.layer3_anomaly.autoencoder import build_autoencoder_features, compute_reconstruction_error


def ae_day_flag_with_runlength(ts_table, ae_result, ae_threshold, run_length: int):
    from src.layer3_anomaly.autoencoder import AUTOENCODER_META_COLS, AUTOENCODER_DIFF_COLS
    twin_feature_cols = [c for c in ae_result.input_cols if c not in AUTOENCODER_META_COLS + AUTOENCODER_DIFF_COLS]
    feats = build_autoencoder_features(ts_table, twin_feature_cols)
    err = compute_reconstruction_error(feats, ae_result)

    daylight_idx = ts_table.index[ts_table["is_daylight"].astype(bool)]
    err_daylight = err[err.index.isin(daylight_idx)].sort_index()
    over = err_daylight > ae_threshold

    # find runs of consecutive True values (consecutive in this sorted, still time-ordered, index)
    run_id = (~over).cumsum()
    run_sizes = over.groupby(run_id).transform("sum") * over  # size of the True-run this row belongs to (0 if not over)
    qualifies = run_sizes >= run_length

    dates = pd.Series(err_daylight.index.date, index=err_daylight.index)
    day_flag = qualifies.groupby(dates).any()
    return day_flag


def main():
    results = {}
    plant_dfs = {}
    for plant in PLANTS:
        base = load_plant_base(plant)
        twin, capacity_kw, meta = base["twin"], base["capacity_kw"], base["meta"]
        ae_result, ae_threshold = train_plant_autoencoder(plant, base)

        scored = score_all_replicates(plant, base=base).reset_index()
        df = merge_with_ground_truth(scored, plant)

        # precompute conformal "borderline" flag: within 50% of threshold magnitude below it
        threshold = base["conformal_result"].threshold
        df["near_threshold"] = (df["r_day"] > 0.5 * threshold) & (df["r_day"] <= threshold)

        # Calibrated from CLEAN data only (calib+test_clean), matching this
        # project's established discipline for every other threshold: never
        # derive a cutoff from injected/labeled data. See investigation
        # notes -- plant_1's own clean data produces a run as long as 28,
        # so a naive "28" pick (eyeballed from an injected replicate) would
        # NOT actually be above the clean noise floor. max_clean_run+1 is
        # the calibrated value; a few honest neighbors are included too.
        clean_dates = base["calib_dates"].union(base["test_clean_dates"])
        ts_full = _full_ts_table(plant, base)
        from src.layer3_anomaly.autoencoder import AUTOENCODER_META_COLS as _M, AUTOENCODER_DIFF_COLS as _D
        _tfc = [c for c in ae_result.input_cols if c not in _M + _D]
        _feats_full = build_autoencoder_features(ts_full, _tfc)
        _err_full = compute_reconstruction_error(_feats_full, ae_result)
        _daylight_idx = ts_full.index[ts_full["is_daylight"].astype(bool)]
        _err_daylight = _err_full[_err_full.index.isin(_daylight_idx)].sort_index()
        _dates_series = pd.Series(_err_daylight.index.date, index=_err_daylight.index)
        _err_clean = _err_daylight[_dates_series.isin(clean_dates)]
        _over_clean = _err_clean > ae_threshold
        _run_id_clean = (~_over_clean).cumsum()
        _run_sizes_clean = _over_clean.groupby(_run_id_clean).transform("sum") * _over_clean
        max_clean_run = int(_run_sizes_clean.max())
        calibrated_rl = max_clean_run + 1
        print(f"[{plant}] max over-threshold run on CLEAN data: {max_clean_run} -> calibrated run_length={calibrated_rl}")

        for run_length in sorted(set([1, calibrated_rl])):
            ae_flags = {}
            for rep_id in range(N_REPLICATES):
                rep_path = ROOT / "data" / "processed" / "injected" / plant / f"replicate_{rep_id:03d}.parquet"
                rep_df = pd.read_parquet(rep_path)
                anchor = compute_physics_anchor(rep_df, meta["latitude"], meta["longitude"], capacity_kw)
                dfj = rep_df.join(anchor)
                ts = build_residual_table(twin, dfj["physics_ac_kw"], dfj, capacity_kw, twin.feature_cols)
                day_flag = ae_day_flag_with_runlength(ts, ae_result, ae_threshold, run_length)
                for d, flagged in day_flag.items():
                    ae_flags[(rep_id, d)] = bool(flagged)

            col = f"ae_rl{run_length}"
            df[col] = df.apply(lambda r: ae_flags.get((r["replicate"], r["date"]), False), axis=1)
            ensemble_col = f"ensemble_rl{run_length}"
            df[ensemble_col] = df["detected"] | df[col]
            smart_col = f"smart_ensemble_rl{run_length}"
            df[smart_col] = df["detected"] | (df[col] & df["near_threshold"])
            if run_length == calibrated_rl:
                df["ae_calibrated"] = df[col]
                df["ensemble_calibrated"] = df[ensemble_col]
                df["smart_ensemble_calibrated"] = df[smart_col]

            key = f"{plant}_rl{run_length}"
            results[key] = dict(
                ae_alone=_metrics_block(df, col)["overall"],
                ae_alone_specificity=_metrics_block(df, col)["hard_negative_overall"]["specificity"],
                ensemble=_metrics_block(df, ensemble_col)["overall"],
                ensemble_specificity=_metrics_block(df, ensemble_col)["hard_negative_overall"]["specificity"],
                smart_ensemble=_metrics_block(df, smart_col)["overall"],
                smart_ensemble_specificity=_metrics_block(df, smart_col)["hard_negative_overall"]["specificity"],
            )
            print(f"{key}: ae_alone recall={results[key]['ae_alone']['recall']:.3f} spec={results[key]['ae_alone_specificity']:.3f} | "
                  f"ensemble recall={results[key]['ensemble']['recall']:.3f} spec={results[key]['ensemble_specificity']:.3f} f1={results[key]['ensemble']['f1']:.3f} | "
                  f"smart recall={results[key]['smart_ensemble']['recall']:.3f} spec={results[key]['smart_ensemble_specificity']:.3f} f1={results[key]['smart_ensemble']['f1']:.3f}")

        plant_dfs[plant] = df

    print("\n=== POOLED (both plants) ===")
    pooled = pd.concat(plant_dfs.values())
    baseline = _metrics_block(pooled, "detected")
    print(f"baseline (production only): recall={baseline['overall']['recall']:.3f} "
          f"precision={baseline['overall']['precision']:.3f} f1={baseline['overall']['f1']:.3f} "
          f"specificity={baseline['hard_negative_overall']['specificity']:.3f}")
    pooled_results = {"baseline": dict(overall=baseline["overall"], specificity=baseline["hard_negative_overall"]["specificity"])}
    print("--- rl=1 (naive 'any single timestep', current committed behavior) ---")
    for variant in ["ensemble_rl1", "smart_ensemble_rl1"]:
        m = _metrics_block(pooled, variant)
        pooled_results[variant] = dict(overall=m["overall"], specificity=m["hard_negative_overall"]["specificity"])
        print(f"{variant}: recall={m['overall']['recall']:.3f} precision={m['overall']['precision']:.3f} "
              f"f1={m['overall']['f1']:.3f} specificity={m['hard_negative_overall']['specificity']:.3f}")

    print("--- calibrated run-length (derived from CLEAN data only, per plant) ---")
    for variant in ["ae_calibrated", "ensemble_calibrated", "smart_ensemble_calibrated"]:
        m = _metrics_block(pooled, variant)
        pooled_results[variant] = dict(overall=m["overall"], specificity=m["hard_negative_overall"]["specificity"])
        print(f"{variant}: recall={m['overall']['recall']:.3f} precision={m['overall']['precision']:.3f} "
              f"f1={m['overall']['f1']:.3f} specificity={m['hard_negative_overall']['specificity']:.3f}")

    results["pooled"] = pooled_results
    with open(ROOT / "reports" / "tables" / "_experiment_ae_runlength.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
