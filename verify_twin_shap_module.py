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
from src.layer4_xai.twin_shap import TwinShapExplainer, explain_flagged_days, DEFAULT_KT_SWITCH

with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)
alpha_nominal = cfg["layer3"]["conformal"]["alpha"]
max_flags = cfg["layer4"]["shap"]["max_flags_explained"]


def print_explanation(date, expl):
    print(f"  {date} [{expl.mode}] twin_kt={expl.twin_kt:.3f}")
    print(f"    physics_ac_kw={expl.physics_ac_kw:.1f}  twin_correction_kw={expl.twin_correction_kw}  expected_q50_kw={expl.expected_q50_kw:.1f}")
    if expl.top_features_kw:
        for feat, val in expl.top_features_kw:
            print(f"      {feat:22s} {val:+9.2f} kW")
    print(f"    note: {expl.note}")


for plant in ("plant_1", "plant_2"):
    print(f"\n=== {plant} ===")
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
    conformal_result = fit_conformal_threshold(
        daily_main.loc[daily_main.index.isin(calib_dates), "r_day"], alpha_nominal=alpha_nominal
    )

    # --- Unit test: explicit fallback-mode row (low Kt) ---
    explainer = TwinShapExplainer(twin, kt_switch=DEFAULT_KT_SWITCH)
    low_kt_row = ts_main[(ts_main["twin_kt"] < DEFAULT_KT_SWITCH) & (ts_main["is_daylight"])].iloc[10]
    expl_fallback = explainer.explain_timestamp(
        low_kt_row, physics_ac_kw=low_kt_row["physics_ac_kw"], twin_kt=low_kt_row["twin_kt"]
    )
    assert expl_fallback.mode == "physics_anchor_fallback"
    assert expl_fallback.twin_correction_kw is None
    assert expl_fallback.top_features_kw == []
    assert abs(expl_fallback.expected_q50_kw - expl_fallback.physics_ac_kw) < 1e-9
    print("Fallback-mode row (low Kt) -- correctly bypassed SHAP:")
    print_explanation(low_kt_row.name.date(), expl_fallback)

    # --- Unit test: explicit SHAP-mode row (clear sky) ---
    clear_row = ts_main[(ts_main["twin_kt"] > 0.85) & (ts_main["is_daylight"])].iloc[10]
    expl_shap = explainer.explain_timestamp(
        clear_row, physics_ac_kw=clear_row["physics_ac_kw"], twin_kt=clear_row["twin_kt"]
    )
    assert expl_shap.mode == "shap"
    assert len(expl_shap.top_features_kw) == 5
    # Reconstruction check: physics_ac_kw + twin_correction_kw must equal
    # the actual expected_q50 the twin/Layer 3 pipeline produced for this row
    actual_expected_q50 = ts_main.loc[clear_row.name, "expected_q50"]
    print(f"\nSHAP-mode row (clear sky) -- reconstruction check:")
    print(f"  explainer's expected_q50_kw = {expl_shap.expected_q50_kw:.3f}")
    print(f"  pipeline's actual expected_q50 = {actual_expected_q50:.3f}")
    assert abs(expl_shap.expected_q50_kw - actual_expected_q50) < 1.0, "SHAP reconstruction doesn't match pipeline output!"
    print_explanation(clear_row.name.date(), expl_shap)

    # --- Full flagged-days run against a real injected replicate ---
    for rep_id in (0,):
        rep_df = pd.read_parquet(f"data/processed/injected/{plant}/replicate_{rep_id:03d}.parquet")
        anchor = compute_physics_anchor(rep_df, meta["latitude"], meta["longitude"], capacity_kw)
        rep_df = rep_df.join(anchor)
        ts_rep = build_residual_table(twin, rep_df["physics_ac_kw"], rep_df, capacity_kw, twin.feature_cols)
        daily_rep = compute_daily_energy_residual(ts_rep, ac_nameplate_kw=capacity_kw)
        night_flags = compute_daily_night_injection_flag(ts_rep, capacity_kw)
        stuck_flags = compute_daily_stuck_sensor_flag(ts_rep)
        labels_rep = classify_days(daily_rep["r_day"], conformal_result, night_flags, stuck_flags)
        print(f"\nreplicate_{rep_id:03d} flags: {labels_rep.value_counts().to_dict()}")

        explanations = explain_flagged_days(twin, ts_rep, labels_rep, max_flags_explained=max_flags)
        for date, expl in explanations.items():
            print_explanation(date, expl)
