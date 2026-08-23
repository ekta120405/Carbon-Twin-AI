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
from src.layer3_anomaly.aggregate import (
    compute_daily_flag_severity_kwh,
    compute_divergence_severity_pct,
    calibrate_divergence_cap,
    compute_risk_score,
    RiskScoreInputs,
)

with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)
alpha_nominal = cfg["layer3"]["conformal"]["alpha"]
cusum_cfg = cfg["layer3"]["cusum"]


def score_table(df, twin, capacity_kw, meta, ref, h_for_this_window, conformal_result, clean_cr_baseline, divergence_cap):
    anchor = compute_physics_anchor(df, meta["latitude"], meta["longitude"], capacity_kw)
    df = df.join(anchor)
    ts_table = build_residual_table(twin, df["physics_ac_kw"], df, capacity_kw, twin.feature_cols)
    daily = compute_daily_energy_residual(ts_table, ac_nameplate_kw=capacity_kw)
    night_flags = compute_daily_night_injection_flag(ts_table, capacity_kw)
    stuck_flags = compute_daily_stuck_sensor_flag(ts_table)
    labels = classify_days(daily["r_day"], conformal_result, night_flags, stuck_flags)
    severity = compute_daily_flag_severity_kwh(daily, labels, ts_table)
    flagged_mask = labels.str.startswith("flagged")
    frac_flagged = flagged_mask.mean()
    total_claimed = daily["claimed_energy_kwh"].sum()
    cum_severity_frac = severity[flagged_mask].sum() / total_claimed if total_claimed > 0 else 0.0
    cusum_res = run_cusum(daily.sort_index()["r_day"], ref, k=cusum_cfg["k"], h=h_for_this_window)
    max_s = max(cusum_res.s_values)
    cr = compute_calibration_ratio(df["ac_power"], df["physics_ac_kw"], df["twin_kt"])
    div = compute_divergence_severity_pct(ts_table, labels)

    inputs = RiskScoreInputs(
        frac_flagged=frac_flagged, cum_severity_frac=cum_severity_frac,
        cusum_max_s=max_s, cusum_h=h_for_this_window,
        calibration_ratio=cr, calibration_ratio_clean_baseline=clean_cr_baseline,
        divergence_flagged_only=div,
    )
    return compute_risk_score(inputs, divergence_cap=divergence_cap)


results = []
for plant in ("plant_1", "plant_2"):
    twin = PIGBQRTwin.load(f"reports/models/{plant}")
    with open(f"data/processed/{plant}_meta.yaml") as f:
        capacity_kw = yaml.safe_load(f)["ac_nameplate_kw"]
    with open("config/plants.yaml") as f:
        meta = yaml.safe_load(f)[plant]

    df_main = pd.read_parquet(f"data/processed/{plant}.parquet")
    anchor_main = compute_physics_anchor(df_main, meta["latitude"], meta["longitude"], capacity_kw)
    df_main_j = df_main.join(anchor_main)
    ts_main = build_residual_table(twin, df_main_j["physics_ac_kw"], df_main_j, capacity_kw, twin.feature_cols)
    daily_main = compute_daily_energy_residual(ts_main, ac_nameplate_kw=capacity_kw)
    daylight_main = ts_main[ts_main["is_daylight"]]
    split_main = daylight_main.groupby(daylight_main.index.date)["split"].agg(lambda s: s.mode().iloc[0])
    calib_dates = split_main[split_main == "calib"].index
    test_clean_dates = split_main[split_main == "test_clean"].index
    calib_test_clean_dates = calib_dates.union(test_clean_dates)

    reference_pool = pd.concat([
        daily_main.loc[daily_main.index.isin(calib_dates), "r_day"],
        daily_main.loc[daily_main.index.isin(test_clean_dates), "r_day"],
    ])
    ref = fit_cusum_reference(reference_pool)

    # h calibrated separately per window length -- 10 days for the real
    # clean calib+test_clean check, 9 days for replicate evaluation
    h_10 = calibrate_h_from_clean_data(reference_pool, k=cusum_cfg["k"], window_len=10, target_fpr=0.05)["recommended_h"]
    h_9 = calibrate_h_from_clean_data(reference_pool, k=cusum_cfg["k"], window_len=9, target_fpr=0.05)["recommended_h"]

    conformal_result = fit_conformal_threshold(daily_main.loc[daily_main.index.isin(calib_dates), "r_day"], alpha_nominal=alpha_nominal)
    clean_cr_baseline = compute_calibration_ratio(df_main_j["ac_power"], df_main_j["physics_ac_kw"], df_main_j["twin_kt"])

    # divergence cap calibrated from calib+test_clean ONLY
    div_cap = calibrate_divergence_cap(ts_main, calib_test_clean_dates, percentile=0.95)
    print(f"{plant}: h_10={h_10}, h_9={h_9}, divergence_cap={div_cap:.4f}")

    # "real clean" check: restrict to the calib+test_clean 10-day block specifically
    clean_block_df = df_main[pd.Series(df_main.index.date, index=df_main.index).isin(calib_test_clean_dates)]
    r = score_table(clean_block_df, twin, capacity_kw, meta, ref, h_10, conformal_result, clean_cr_baseline, div_cap)
    results.append({"plant": plant, "source": "REAL_CLEAN(calib+test_clean block)", "replicate": None, "score": r.score, "band": r.band, **r.components})

    for rep_id in range(0, 30, 5):
        rep_df = pd.read_parquet(f"data/processed/injected/{plant}/replicate_{rep_id:03d}.parquet")
        r = score_table(rep_df, twin, capacity_kw, meta, ref, h_9, conformal_result, clean_cr_baseline, div_cap)
        results.append({"plant": plant, "source": "INJECTED", "replicate": rep_id, "score": r.score, "band": r.band, **r.components})

res_df = pd.DataFrame(results)
pd.set_option("display.width", 160)
print(res_df.to_string(index=False))
