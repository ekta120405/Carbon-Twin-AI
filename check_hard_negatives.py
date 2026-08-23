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

with open("config/config.yaml") as f:
    alpha_nominal = yaml.safe_load(f)["layer3"]["conformal"]["alpha"]

results = []

for plant in ("plant_1", "plant_2"):
    twin = PIGBQRTwin.load(f"reports/models/{plant}")
    with open(f"data/processed/{plant}_meta.yaml") as f:
        capacity_kw = yaml.safe_load(f)["ac_nameplate_kw"]
    with open("config/plants.yaml") as f:
        meta = yaml.safe_load(f)[plant]

    # fit conformal threshold ONCE from real clean main-table data
    df_main = pd.read_parquet(f"data/processed/{plant}.parquet")
    anchor_main = compute_physics_anchor(df_main, meta["latitude"], meta["longitude"], capacity_kw)
    df_main_j = df_main.join(anchor_main)
    ts_main = build_residual_table(twin, df_main_j["physics_ac_kw"], df_main_j, capacity_kw, twin.feature_cols)
    daily_main = compute_daily_energy_residual(ts_main, ac_nameplate_kw=capacity_kw)
    daylight_main = ts_main[ts_main["is_daylight"]]
    split_main = daylight_main.groupby(daylight_main.index.date)["split"].agg(lambda s: s.mode().iloc[0])
    calib_dates = split_main[split_main == "calib"].index
    conformal_result = fit_conformal_threshold(daily_main.loc[daily_main.index.isin(calib_dates), "r_day"], alpha_nominal=alpha_nominal)

    labels_gt = pd.read_parquet(f"data/ground_truth/{plant}_labels.parquet")
    hard_negatives = labels_gt[labels_gt["kind"] == "hard_negative"]

    for hn_type in sorted(hard_negatives["type"].unique()):
        examples = hard_negatives[hard_negatives["type"] == hn_type].head(2)
        for _, ex in examples.iterrows():
            rep_id = int(ex["replicate"])
            date = ex["date"]
            rep_df = pd.read_parquet(f"data/processed/injected/{plant}/replicate_{rep_id:03d}.parquet")
            rep_anchor = compute_physics_anchor(rep_df, meta["latitude"], meta["longitude"], capacity_kw)
            rep_df = rep_df.join(rep_anchor)
            rep_ts = build_residual_table(twin, rep_df["physics_ac_kw"], rep_df, capacity_kw, twin.feature_cols)
            rep_daily = compute_daily_energy_residual(rep_ts, ac_nameplate_kw=capacity_kw)
            night_flags = compute_daily_night_injection_flag(rep_ts, capacity_kw)
            stuck_flags = compute_daily_stuck_sensor_flag(rep_ts)
            rep_labels = classify_days(rep_daily["r_day"], conformal_result, night_flags, stuck_flags)

            target_date = pd.Timestamp(date).date()
            label = rep_labels.get(target_date, "MISSING")
            is_falsely_flagged = str(label).startswith("flagged")

            results.append({
                "plant": plant, "type": hn_type, "replicate": rep_id, "date": date,
                "label": label, "FALSE_POSITIVE": is_falsely_flagged,
            })

res_df = pd.DataFrame(results)
pd.set_option("display.width", 140)
pd.set_option("display.max_rows", 100)
print(res_df.to_string(index=False))
print()
print("=== SPECIFICITY SUMMARY (fraction NOT falsely flagged) ===")
print(res_df.groupby(["plant", "type"])["FALSE_POSITIVE"].agg(["sum", "count"]))
print()
overall_fp = res_df["FALSE_POSITIVE"].sum()
overall_n = len(res_df)
print(f"Overall: {overall_fp} false positives out of {overall_n} hard-negative examples checked")
print(f"Specificity: {1 - overall_fp/overall_n:.3f}")
