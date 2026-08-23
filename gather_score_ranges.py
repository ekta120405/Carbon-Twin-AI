import sys
sys.path.insert(0, ".")

import pandas as pd
import yaml

from src.layer2_twin.twin_model import PIGBQRTwin
from src.layer2_twin.physics_anchor import compute_physics_anchor, compute_calibration_ratio
from src.layer3_anomaly.residual import (
    build_residual_table,
    compute_daily_energy_residual,
    compute_daily_night_injection_flag,
    compute_daily_stuck_sensor_flag,
)
from src.layer3_anomaly.conformal import fit_conformal_threshold, classify_days
from src.layer3_anomaly.cusum import fit_cusum_reference, run_cusum, calibrate_h_from_clean_data
from src.layer3_anomaly.aggregate import compute_daily_flag_severity_kwh, compute_divergence_severity_pct

with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)
alpha_nominal = cfg["layer3"]["conformal"]["alpha"]
cusum_cfg = cfg["layer3"]["cusum"]


def score_inputs_for_table(df, twin, capacity_kw, meta, ref, h_calibrated, conformal_result):
    anchor = compute_physics_anchor(df, meta["latitude"], meta["longitude"], capacity_kw)
    df = df.join(anchor)
    ts_table = build_residual_table(twin, df["physics_ac_kw"], df, capacity_kw, twin.feature_cols)
    daily = compute_daily_energy_residual(ts_table, ac_nameplate_kw=capacity_kw)
    night_flags = compute_daily_night_injection_flag(ts_table, capacity_kw)
    stuck_flags = compute_daily_stuck_sensor_flag(ts_table)

    # NOTE: conformal_result, ref, h_calibrated are all fit ONCE from real
    # clean main-table data (train/calib/test_clean) OUTSIDE this function,
    # then APPLIED here to whatever table is passed in (real or a replicate).
    # Never re-derive calibration from the table being scored -- a replicate
    # only spans ~9 days and may not even contain the calib split's actual
    # calendar dates, and re-fitting per-table would also just be wrong in
    # principle (calibrate once on trusted data, evaluate anything against
    # that fixed threshold).
    labels = classify_days(daily["r_day"], conformal_result, night_flags, stuck_flags)

    severity = compute_daily_flag_severity_kwh(daily, labels, ts_table)
    flagged_mask = labels.str.startswith("flagged")
    frac_flagged = flagged_mask.mean()
    cum_severity_kwh = severity[flagged_mask].sum()
    total_claimed_kwh = daily["claimed_energy_kwh"].sum()
    cum_severity_frac = cum_severity_kwh / total_claimed_kwh if total_claimed_kwh > 0 else 0.0

    all_r_sorted = daily.sort_index()["r_day"]
    cusum_res = run_cusum(all_r_sorted, ref, k=cusum_cfg["k"], h=h_calibrated)
    max_s = max(cusum_res.s_values)

    cr = compute_calibration_ratio(df["ac_power"], df["physics_ac_kw"], df["twin_kt"])

    divergence_mean_global = df["evidence_irr_divergence_pct"].mean()
    divergence_flagged_only = compute_divergence_severity_pct(ts_table, labels)

    return {
        "frac_flagged": frac_flagged,
        "cum_severity_kwh": cum_severity_kwh,
        "cum_severity_frac": cum_severity_frac,
        "cusum_max_s": max_s,
        "cusum_h": h_calibrated,
        "calibration_ratio": cr,
        "divergence_mean_global": divergence_mean_global,
        "divergence_flagged_only": divergence_flagged_only,
    }


results = []
for plant in ("plant_1", "plant_2"):
    twin = PIGBQRTwin.load(f"reports/models/{plant}")
    with open(f"data/processed/{plant}_meta.yaml") as f:
        capacity_kw = yaml.safe_load(f)["ac_nameplate_kw"]
    with open("config/plants.yaml") as f:
        meta = yaml.safe_load(f)[plant]

    df_main = pd.read_parquet(f"data/processed/{plant}.parquet")
    anchor_main = compute_physics_anchor(df_main, meta["latitude"], meta["longitude"], capacity_kw)
    df_main_joined = df_main.join(anchor_main)
    ts_main = build_residual_table(twin, df_main_joined["physics_ac_kw"], df_main_joined, capacity_kw, twin.feature_cols)
    daily_main = compute_daily_energy_residual(ts_main, ac_nameplate_kw=capacity_kw)
    daylight_main = ts_main[ts_main["is_daylight"]]
    split_main = daylight_main.groupby(daylight_main.index.date)["split"].agg(lambda s: s.mode().iloc[0])
    calib_dates = split_main[split_main == "calib"].index
    calib_r_day = daily_main.loc[daily_main.index.isin(calib_dates), "r_day"]
    ref = fit_cusum_reference(pd.concat([
        daily_main.loc[daily_main.index.isin(calib_dates), "r_day"],
        daily_main.loc[daily_main.index.isin(split_main[split_main == "test_clean"].index), "r_day"],
    ]))
    reference_pool = pd.concat([
        daily_main.loc[daily_main.index.isin(calib_dates), "r_day"],
        daily_main.loc[daily_main.index.isin(split_main[split_main == "test_clean"].index), "r_day"],
    ])
    h_calibrated = calibrate_h_from_clean_data(reference_pool, k=cusum_cfg["k"], window_len=9, target_fpr=0.05)["recommended_h"]
    conformal_result = fit_conformal_threshold(calib_r_day, alpha_nominal=alpha_nominal)

    # real clean data
    r = score_inputs_for_table(pd.read_parquet(f"data/processed/{plant}.parquet"), twin, capacity_kw, meta, ref, h_calibrated, conformal_result)
    r.update({"plant": plant, "source": "REAL_CLEAN", "replicate": None})
    results.append(r)

    # sample of injected replicates (every 5th, to keep runtime reasonable, ~6 per plant)
    for rep_id in range(0, 30, 5):
        rep_df = pd.read_parquet(f"data/processed/injected/{plant}/replicate_{rep_id:03d}.parquet")
        r = score_inputs_for_table(rep_df, twin, capacity_kw, meta, ref, h_calibrated, conformal_result)
        r.update({"plant": plant, "source": "INJECTED", "replicate": rep_id})
        results.append(r)

res_df = pd.DataFrame(results)
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)
print(res_df.to_string(index=False))
