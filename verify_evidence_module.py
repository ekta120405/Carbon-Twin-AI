import sys
sys.path.insert(0, ".")

import json
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
from src.layer4_xai.flag_shap import compute_day_level_evidence_features
from src.layer4_xai.evidence import build_evidence_bundles_for_plant, save_evidence_bundles

with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)
alpha_nominal = cfg["layer3"]["conformal"]["alpha"]
ef = cfg["layer4"]["carbon"]["grid_emission_factor_tco2_per_mwh"]
price = cfg["layer4"]["carbon"]["carbon_price_usd_per_tco2"]
max_flags = cfg["layer4"]["shap"]["max_flags_explained"]


def build_pipeline(df, twin, capacity_kw, meta):
    anchor = compute_physics_anchor(df, meta["latitude"], meta["longitude"], capacity_kw)
    df = df.join(anchor)
    ts_table = build_residual_table(twin, df["physics_ac_kw"], df, capacity_kw, twin.feature_cols)
    daily = compute_daily_energy_residual(ts_table, ac_nameplate_kw=capacity_kw)
    return ts_table, daily


# plant_1/replicate_010 -> flagged_overclaim x3, flagged_stuck_sensor x1 (from 4.2's verified run)
# plant_2/replicate_000 -> flagged_night_injection x2, flagged_overclaim x1 (from 4.2's verified run)
# together these two runs cover all three flag_type explanation paths.
PLANT_REPLICATES = {"plant_1": 10, "plant_2": 0}

all_flag_types_seen = set()

for plant, rep_id in PLANT_REPLICATES.items():
    print(f"\n=== {plant} / replicate_{rep_id:03d} ===")
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
    evidence_main = compute_day_level_evidence_features(ts_main, daily_main, capacity_kw)
    clean_reference_pool = evidence_main[evidence_main.index.isin(calib_test_clean_dates)]

    rep_df = pd.read_parquet(f"data/processed/injected/{plant}/replicate_{rep_id:03d}.parquet")
    ts_rep, daily_rep = build_pipeline(rep_df, twin, capacity_kw, meta)
    night_flags = compute_daily_night_injection_flag(ts_rep, capacity_kw)
    stuck_flags = compute_daily_stuck_sensor_flag(ts_rep)
    labels_rep = classify_days(daily_rep["r_day"], conformal_result, night_flags, stuck_flags)
    print(f"flags: {labels_rep.value_counts().to_dict()}")

    bundles = build_evidence_bundles_for_plant(
        plant, twin, ts_rep, daily_rep, labels_rep, clean_reference_pool,
        capacity_kw, ef, price, max_flags_explained=max_flags,
    )

    # Cross-check: severity_kwh and carbon numbers in each bundle must match
    # 4.3's independently-computed severity for the SAME day exactly --
    # not just internally consistent with itself.
    independent_severity = compute_daily_flag_severity_kwh(daily_rep, labels_rep, ts_rep)

    for b in bundles:
        all_flag_types_seen.add(b["flag_type"])
        date = pd.Timestamp(b["date"]).date()
        expected_sev = float(independent_severity.get(date, 0.0))
        assert abs(b["severity_kwh"] - expected_sev) < 1e-6, (
            f"Bundle severity_kwh mismatch for {date}: bundle={b['severity_kwh']}, "
            f"independent compute_daily_flag_severity_kwh={expected_sev}"
        )
        print(f"\n--- {b['date']} [{b['flag_type']}] ---")
        print(f"  severity_kwh={b['severity_kwh']:.2f} (cross-check OK, matches independent 4.3 computation)")
        print(f"  carbon_exposure: {b['carbon_exposure']}")
        print(f"  physical_explanation: mode={b['physical_explanation']['mode']}, "
              f"expected_q50_kw={b['physical_explanation']['expected_q50_kw']:.1f}")
        if b['physical_explanation']['mode'] == 'shap':
            print(f"    top_features_kw: {b['physical_explanation']['top_features_kw']}")
        print(f"  suspicion_explanation: mode={b['suspicion_explanation']['mode']}")
        if b['suspicion_explanation']['mode'] == 'statistical':
            print(f"    top_drivers: {b['suspicion_explanation']['top_drivers']}")
        else:
            print(f"    text: {b['suspicion_explanation']['text']}")
        print(f"  known_caveats ({len(b['known_caveats'])}): "
              f"{[c[:60] + '...' for c in b['known_caveats']]}")

    out_path = f"reports/evidence_bundles/{plant}_replicate_{rep_id:03d}_evidence_bundles.json"
    save_evidence_bundles(bundles, out_path)
    # Round-trip check: JSON must be re-loadable and match
    with open(out_path) as f:
        reloaded = json.load(f)
    assert reloaded == bundles or json.dumps(reloaded, sort_keys=True) == json.dumps(
        json.loads(json.dumps(bundles, default=str)), sort_keys=True
    ), "Round-trip JSON mismatch!"
    print(f"\nSaved and round-trip-verified: {out_path}")

print(f"\nAll flag_type paths exercised: {sorted(all_flag_types_seen)}")
assert all_flag_types_seen == {"flagged_overclaim", "flagged_night_injection", "flagged_stuck_sensor"}, (
    f"Missing flag types! Expected all three, saw: {all_flag_types_seen}"
)
print("Confirmed: all three flag-type paths (Path A statistical, Path B night-injection, Path B stuck-sensor) produced bundles.")
