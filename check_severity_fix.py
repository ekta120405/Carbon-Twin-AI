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
from src.layer3_anomaly.aggregate import compute_daily_flag_severity_kwh

with open("config/config.yaml") as f:
    alpha_nominal = yaml.safe_load(f)["layer3"]["conformal"]["alpha"]

for plant in ("plant_1", "plant_2"):
    twin = PIGBQRTwin.load(f"reports/models/{plant}")
    with open(f"data/processed/{plant}_meta.yaml") as f:
        capacity_kw = yaml.safe_load(f)["ac_nameplate_kw"]
    with open("config/plants.yaml") as f:
        meta = yaml.safe_load(f)[plant]

    df = pd.read_parquet(f"data/processed/{plant}.parquet")
    anchor = compute_physics_anchor(df, meta["latitude"], meta["longitude"], capacity_kw)
    df = df.join(anchor)
    ts_table = build_residual_table(twin, df["physics_ac_kw"], df, capacity_kw, twin.feature_cols)
    daily = compute_daily_energy_residual(ts_table, ac_nameplate_kw=capacity_kw)
    night_flags = compute_daily_night_injection_flag(ts_table, capacity_kw)
    stuck_flags = compute_daily_stuck_sensor_flag(ts_table)

    daylight_only = ts_table[ts_table["is_daylight"]]
    daily_split = daylight_only.groupby(daylight_only.index.date)["split"].agg(lambda s: s.mode().iloc[0])
    calib_r_day = daily.loc[daily_split[daily_split == "calib"].index, "r_day"]
    result = fit_conformal_threshold(calib_r_day, alpha_nominal=alpha_nominal)
    labels = classify_days(daily["r_day"], result, night_flags, stuck_flags)

    severity = compute_daily_flag_severity_kwh(daily, labels, ts_table)
    flagged_mask = labels.str.startswith("flagged")

    old_total = daily.loc[flagged_mask, "excess_energy_kwh"].sum()
    new_total = severity[flagged_mask].sum()

    print(f"=== {plant} ===")
    print(f"OLD: {old_total:.1f} kWh, NEW: {new_total:.1f} kWh")
    print()
