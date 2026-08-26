import sys
sys.path.insert(0, ".")

import pandas as pd
import yaml

from src.layer2_twin.twin_model import PIGBQRTwin
from src.layer2_twin.physics_anchor import compute_physics_anchor
from src.layer3_anomaly.residual import (
    build_residual_table,
    compute_daily_energy_residual,
    compute_daily_night_injection_flag,
    compute_daily_stuck_sensor_flag,
)
from src.layer3_anomaly.conformal import fit_conformal_threshold, classify_days
from src.layer4_xai.carbon import build_carbon_table, aggregate_carbon_exposure, REQUIRED_CAVEATS

with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)
alpha_nominal = cfg["layer3"]["conformal"]["alpha"]
ef = cfg["layer4"]["carbon"]["grid_emission_factor_tco2_per_mwh"]
price = cfg["layer4"]["carbon"]["carbon_price_usd_per_tco2"]
print(f"Using config values: EF={ef} tCO2e/MWh, price=${price}/tCO2e\n")


def build_labels_and_carbon(df, twin, capacity_kw, meta, conformal_result):
    anchor = compute_physics_anchor(df, meta["latitude"], meta["longitude"], capacity_kw)
    df = df.join(anchor)
    ts_table = build_residual_table(twin, df["physics_ac_kw"], df, capacity_kw, twin.feature_cols)
    daily = compute_daily_energy_residual(ts_table, ac_nameplate_kw=capacity_kw)
    night_flags = compute_daily_night_injection_flag(ts_table, capacity_kw)
    stuck_flags = compute_daily_stuck_sensor_flag(ts_table)
    labels = classify_days(daily["r_day"], conformal_result, night_flags, stuck_flags)
    carbon_table = build_carbon_table(daily, labels, ts_table, ef, price)
    return carbon_table


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

    conformal_result = fit_conformal_threshold(
        daily_main.loc[daily_main.index.isin(calib_dates), "r_day"], alpha_nominal=alpha_nominal
    )

    # --- Check 1: real clean calib+test_clean block -> should be ~zero exposure ---
    calib_test_clean_dates = calib_dates.union(test_clean_dates)
    clean_block_df = df_main[pd.Series(df_main.index.date, index=df_main.index).isin(calib_test_clean_dates)]
    clean_carbon = build_labels_and_carbon(clean_block_df, twin, capacity_kw, meta, conformal_result)
    clean_agg = aggregate_carbon_exposure(clean_carbon)
    print(f"[{plant}] REAL CLEAN (calib+test_clean, n={len(clean_carbon)} days):")
    print(f"  flag_type counts: {clean_carbon['flag_type'].value_counts().to_dict()}")
    print(f"  {clean_agg}")

    # --- Check 2: a handful of injected replicates -> sane, small-magnitude numbers ---
    for rep_id in (0, 10, 20):
        rep_df = pd.read_parquet(f"data/processed/injected/{plant}/replicate_{rep_id:03d}.parquet")
        rep_carbon = build_labels_and_carbon(rep_df, twin, capacity_kw, meta, conformal_result)
        rep_agg = aggregate_carbon_exposure(rep_carbon)
        flagged_rows = rep_carbon[rep_carbon["flag_type"].str.startswith("flagged")]
        print(f"[{plant}] replicate_{rep_id:03d}: flags={rep_carbon['flag_type'].value_counts().to_dict()}")
        if len(flagged_rows):
            print(flagged_rows.to_string())
        print(f"  totals: {rep_agg}")
    print()

print("Required caveats (must accompany any report using these numbers):")
for c in REQUIRED_CAVEATS:
    print(" -", c)
