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
from src.layer4_xai.flag_shap import (
    compute_day_level_evidence_features,
    explain_flagged_day,
    EVIDENCE_FEATURE_COLS,
)

with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)
alpha_nominal = cfg["layer3"]["conformal"]["alpha"]


def build_pipeline(df, twin, capacity_kw, meta):
    anchor = compute_physics_anchor(df, meta["latitude"], meta["longitude"], capacity_kw)
    df = df.join(anchor)
    ts_table = build_residual_table(twin, df["physics_ac_kw"], df, capacity_kw, twin.feature_cols)
    daily = compute_daily_energy_residual(ts_table, ac_nameplate_kw=capacity_kw)
    return ts_table, daily


for plant in ("plant_1", "plant_2"):
    print(f"\n=== {plant} ===")
    twin = PIGBQRTwin.load(f"reports/models/{plant}")
    with open(f"data/processed/{plant}_meta.yaml") as f:
        capacity_kw = yaml.safe_load(f)["ac_nameplate_kw"]
    with open("config/plants.yaml") as f:
        meta = yaml.safe_load(f)[plant]

    df_main = pd.read_parquet(f"data/processed/{plant}.parquet")
    ts_main, daily_main = build_pipeline(df_main, twin, capacity_kw, meta)
    daylight_main = ts_main[ts_main["is_daylight"]]
    split_main = daylight_main.groupby(daylight_main.index.date)["split"].agg(lambda s: s.mode().iloc[0])
    calib_dates = split_main[split_main == "calib"].index
    test_clean_dates = split_main[split_main == "test_clean"].index
    calib_test_clean_dates = calib_dates.union(test_clean_dates)

    conformal_result = fit_conformal_threshold(
        daily_main.loc[daily_main.index.isin(calib_dates), "r_day"], alpha_nominal=alpha_nominal
    )

    # Clean reference pool for Path A (n=10, calib+test_clean, NEVER injected)
    evidence_main = compute_day_level_evidence_features(ts_main, daily_main, capacity_kw)
    clean_reference_pool = evidence_main[evidence_main.index.isin(calib_test_clean_dates)]
    print(f"Clean reference pool (n={len(clean_reference_pool)}):")
    print(clean_reference_pool)

    # --- Path B sanity check: real N3-curtailment-vs-A8 ambiguity is documented ---
    for rep_id in (10,):  # plant_1 rep_010 had a flagged_stuck_sensor day (from 4.1's run)
        if plant != "plant_1":
            continue
        rep_df = pd.read_parquet(f"data/processed/injected/{plant}/replicate_{rep_id:03d}.parquet")
        ts_rep, daily_rep = build_pipeline(rep_df, twin, capacity_kw, meta)
        night_flags = compute_daily_night_injection_flag(ts_rep, capacity_kw)
        stuck_flags = compute_daily_stuck_sensor_flag(ts_rep)
        labels_rep = classify_days(daily_rep["r_day"], conformal_result, night_flags, stuck_flags)
        print(f"\nreplicate_{rep_id:03d} flags: {labels_rep.value_counts().to_dict()}")

        evidence_rep = compute_day_level_evidence_features(ts_rep, daily_rep, capacity_kw)
        for date, flag_type in labels_rep.items():
            if not flag_type.startswith("flagged"):
                continue
            expl = explain_flagged_day(
                date, flag_type, ts_rep, evidence_table=evidence_rep,
                clean_reference_pool=clean_reference_pool, ac_nameplate_kw=capacity_kw,
            )
            print(f"\n  --- {date} [{flag_type}] ---")
            if flag_type == "flagged_overclaim":
                print(f"  top_drivers (z-scores): {expl.top_drivers}")
                print(f"  raw_values: {expl.raw_values}")
            else:
                print(f"  text: {expl.text}")
                print(f"  trigger_values: {expl.trigger_values}")
                if expl.caveat:
                    print(f"  CAVEAT: {expl.caveat}")

    # --- Path A on a plant with several flagged_overclaim days, check for dilution ---
    print(f"\nPath A ranking check across multiple replicates ({plant}):")
    for rep_id in (0, 5, 15):
        rep_df = pd.read_parquet(f"data/processed/injected/{plant}/replicate_{rep_id:03d}.parquet")
        ts_rep, daily_rep = build_pipeline(rep_df, twin, capacity_kw, meta)
        night_flags = compute_daily_night_injection_flag(ts_rep, capacity_kw)
        stuck_flags = compute_daily_stuck_sensor_flag(ts_rep)
        labels_rep = classify_days(daily_rep["r_day"], conformal_result, night_flags, stuck_flags)
        evidence_rep = compute_day_level_evidence_features(ts_rep, daily_rep, capacity_kw)
        overclaim_dates = labels_rep[labels_rep == "flagged_overclaim"].index
        for date in overclaim_dates:
            expl = explain_flagged_day(
                date, "flagged_overclaim", ts_rep, evidence_table=evidence_rep,
                clean_reference_pool=clean_reference_pool, ac_nameplate_kw=capacity_kw,
            )
            top = expl.top_drivers[0]
            print(f"  rep{rep_id:03d} {date}: top driver = {top[0]} (z={top[1]:.2f})")

    # --- Path B, explicit night-injection check (the other guardrail path --
    #     NOT exercised anywhere above, since plant_1's sampled replicates
    #     happened to only surface flagged_stuck_sensor as the Path B example) ---
    print(f"\nPath B night-injection check ({plant}):")
    found_any = False
    for rep_id in range(3):
        rep_df = pd.read_parquet(f"data/processed/injected/{plant}/replicate_{rep_id:03d}.parquet")
        ts_rep, daily_rep = build_pipeline(rep_df, twin, capacity_kw, meta)
        night_flags = compute_daily_night_injection_flag(ts_rep, capacity_kw)
        stuck_flags = compute_daily_stuck_sensor_flag(ts_rep)
        labels_rep = classify_days(daily_rep["r_day"], conformal_result, night_flags, stuck_flags)
        night_dates = labels_rep[labels_rep == "flagged_night_injection"].index
        for date in night_dates:
            found_any = True
            expl = explain_flagged_day(date, "flagged_night_injection", ts_rep, ac_nameplate_kw=capacity_kw)
            print(f"  rep{rep_id:03d} {date}: {expl.text}")
            print(f"    trigger_values: {expl.trigger_values}")
    if not found_any:
        print("  (no flagged_night_injection days in replicates 0-2 for this plant -- "
              "not a failure, just means this plant/sample didn't surface one; "
              "check other replicate indices if you want to see this path fire here)")
