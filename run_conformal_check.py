import sys
sys.path.insert(0, ".")

import pandas as pd
import yaml

from src.layer2_twin.twin_model import PIGBQRTwin
from src.layer2_twin.physics_anchor import compute_physics_anchor
from src.layer3_anomaly.residual import build_residual_table, compute_daily_energy_residual
from src.layer3_anomaly.conformal import fit_conformal_threshold, flag_days, tag_kt_regime

with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)
alpha_nominal = cfg["layer3"]["conformal"]["alpha"]
csi_bins = cfg["layer3"]["conformal"]["csi_bins"]

for plant in ("plant_1", "plant_2"):
    twin = PIGBQRTwin.load(f"reports/models/{plant}")
    df = pd.read_parquet(f"data/processed/{plant}.parquet")
    with open(f"data/processed/{plant}_meta.yaml") as f:
        capacity_kw = yaml.safe_load(f)["ac_nameplate_kw"]
    with open("config/plants.yaml") as f:
        meta = yaml.safe_load(f)[plant]
    anchor = compute_physics_anchor(df, meta["latitude"], meta["longitude"], capacity_kw)
    df = df.join(anchor)
    ts_table = build_residual_table(twin, df["physics_ac_kw"], df, capacity_kw, twin.feature_cols)

    daylight = ts_table[ts_table["is_daylight"]].copy()
    daylight["date"] = daylight.index.date
    date_to_split = daylight.groupby("date")["split"].agg(lambda s: s.mode().iloc[0])
    daily_kt = daylight.groupby("date")["twin_kt"].mean()

    daily = compute_daily_energy_residual(ts_table, ac_nameplate_kw=capacity_kw)
    daily["split"] = date_to_split
    daily["regime"] = tag_kt_regime(daily_kt, csi_bins)

    calib_r_day = daily.loc[daily["split"] == "calib", "r_day"]
    result = fit_conformal_threshold(calib_r_day, alpha_nominal=alpha_nominal)

    print(f"=== {plant} ===")
    print(f"n_calib={result.n_calib}, threshold={result.threshold:.4f}, "
          f"alpha_nominal={result.alpha_nominal}, alpha_achievable={result.alpha_achievable:.4f}")
    print()
    print("calib r_day values (sorted):", result.calib_r_day_values)
    print()

    test_clean = daily[daily["split"] == "test_clean"].copy()
    test_clean["flagged"] = flag_days(test_clean["r_day"], result)
    n_flagged = int(test_clean["flagged"].sum())
    n_total = len(test_clean)

    print("test_clean days vs threshold:")
    print(test_clean[["r_day", "regime", "flagged"]])
    print(f"n flagged out of {n_total}: {n_flagged}")
    print()
