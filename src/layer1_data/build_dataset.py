"""
src/layer1_data/build_dataset.py

Layer 1 orchestrator (Task 5). Runs the full pipeline for one plant in the
exact dependency order and writes data/processed/<plant>.parquet.

Usage:
    python -m src.layer1_data.build_dataset --plant plant_1
    python -m src.layer1_data.build_dataset --plant plant_2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

from src.utils.timezone import parse_kaggle_datetime, to_utc, assert_solar_noon_sane
from src.layer1_data.downscale import downscale_ghi_csi_preserving
from src.layer1_data.quality import build_clean_mask
from src.layer1_data.features import build_twin_features, build_evidence_features
from src.layer1_data.capacity import estimate_ac_nameplate
from src.layer1_data.splits import assign_day_blocks, assert_splits_disjoint


def load_generation(plant_num: int) -> pd.Series:
    """5.1 + 5.2: parse timestamps, localise to UTC, aggregate 22 inverters."""
    path = Path(f"data/raw/kaggle/Plant_{plant_num}_Generation_Data.csv")
    df = pd.read_csv(path)
    ts = parse_kaggle_datetime(df["DATE_TIME"])
    df = df.assign(timestamp_utc=to_utc(ts))
    return df.groupby("timestamp_utc")["AC_POWER"].sum().sort_index()


def load_onsite_weather(plant_num: int) -> pd.DataFrame:
    path = Path(f"data/raw/kaggle/Plant_{plant_num}_Weather_Sensor_Data.csv")
    df = pd.read_csv(path)
    ts = parse_kaggle_datetime(df["DATE_TIME"])
    df = df.assign(timestamp_utc=to_utc(ts))
    df = df.set_index("timestamp_utc").sort_index()
    return df.rename(columns={
        "AMBIENT_TEMPERATURE": "onsite_ambient_temp",
        "MODULE_TEMPERATURE": "onsite_module_temp",
        "IRRADIATION": "onsite_irradiation",
    })[["onsite_ambient_temp", "onsite_module_temp", "onsite_irradiation"]]


def load_nasa_weather(site: str, lat: float, lon: float) -> pd.DataFrame:
    """5.3: load the resolved site's CENTER grid cell from Task 4's extraction."""
    tag = f"{site}_{lat:+.2f}_{lon:+.2f}".replace("+", "p").replace("-", "m").replace(".", "_")
    candidates = list(Path("data/raw/nasa_power").glob(f"{site}*"))
    center = [c for c in candidates if "center" in c.stem.lower()]
    path = center[0] if center else candidates[0]
    return pd.read_parquet(path).drop(columns=["site", "lat", "lon", "dlat", "dlon"], errors="ignore")


def run(plant_key: str, config: dict, plants_config: dict) -> Path:
    plant_num = int(plant_key.split("_")[1])
    meta = plants_config[plant_key]
    lat, lon = meta["latitude"], meta["longitude"]
    assert lat is not None and lon is not None, f"{plant_key} coordinates not set in plants.yaml"

    print(f"[{plant_key}] 5.1-5.2: loading + aggregating Kaggle generation data")
    claimed_power = load_generation(plant_num)
    onsite = load_onsite_weather(plant_num)

    print(f"[{plant_key}] validating timezone contract (solar-noon sanity check)")
    assert_solar_noon_sane(claimed_power, lon)
    print(f"[{plant_key}]   PASSED — timezone localisation confirmed correct")

    print(f"[{plant_key}] 5.3: loading NASA POWER independent weather")
    nasa_hourly = load_nasa_weather(meta["coordinate_source_site"], lat, lon) \
        if "coordinate_source_site" in meta else load_nasa_weather(plant_key, lat, lon)

    print(f"[{plant_key}] 5.4: clear-sky-index-preserving downscaling to 15-min")
    ghi_downscaled = downscale_ghi_csi_preserving(nasa_hourly["ALLSKY_SFC_SW_DWN"], lat, lon)
    other_weather = nasa_hourly[["T2M", "RH2M", "WS2M", "PS", "PRECTOTCORR"]].rename(
        columns={"T2M": "t2m", "RH2M": "rh2m", "WS2M": "ws2m", "PS": "ps", "PRECTOTCORR": "prectotcorr"}
    ).reindex(ghi_downscaled.index, method="nearest")  # hourly -> reindex onto the fine grid

    weather_fine = pd.concat([ghi_downscaled, other_weather], axis=1)

    # Align everything to the claimed-power index (15-min, ground truth for timing)
    combined = pd.DataFrame(index=claimed_power.index)
    combined["ac_power"] = claimed_power
    combined = combined.join(weather_fine, how="left")
    combined = combined.join(onsite, how="left")

    print(f"[{plant_key}] 5.6: estimating nameplate capacity (frozen, single-pass)")
    final_capacity = estimate_ac_nameplate(combined["ac_power"], combined["kt_15min"])
    print(f"[{plant_key}]   FROZEN AC nameplate: {final_capacity:.1f} kW")

    print(f"[{plant_key}] 5.5: physics ceiling QC")
    qc_cfg = config["layer1"]["quality"]
    qc = build_clean_mask(
        combined, final_capacity, lat, lon,
        ceiling_tolerance_training=qc_cfg["ceiling_tolerance_training"],
        ceiling_tolerance_fraud=qc_cfg["ceiling_tolerance_fraud"],
        stuck_run_length=qc_cfg["stuck_sensor_run_length"],
        min_elevation_deg=qc_cfg["min_solar_elevation_deg"],
    )
    combined = combined.join(qc)
    n_ceiling_violations_training = int(qc["exceeds_ceiling_training"].sum())
    n_ceiling_violations_fraud = int(qc["exceeds_ceiling_fraud"].sum())
    n_clean = int(qc["is_clean"].sum())
    print(f"[{plant_key}]   {n_ceiling_violations_training} training-exclusion ceiling "
          f"violations (generous threshold, excluded from Layer 2 training only)")
    print(f"[{plant_key}]   {n_ceiling_violations_fraud} fraud-severity ceiling "
          f"violations (strict threshold, genuine max-severity Layer 3 flags)")
    print(f"[{plant_key}]   {n_clean}/{len(combined)} rows clean")

    print(f"[{plant_key}] 5.5: feature engineering (twin_* / evidence_* separation)")
    twin_feats = build_twin_features(combined, lat, lon)
    evidence_feats = build_evidence_features(combined, ac_nameplate_kw=final_capacity)

    print(f"[{plant_key}] 5.7: day-block splitting")
    splits_cfg = config["layer1"]["splits"]
    assert_splits_disjoint(*[tuple(v) for v in splits_cfg.values()])
    split_labels = assign_day_blocks(
        combined.index,
        train_days=tuple(splits_cfg["train_days"]),
        calib_days=tuple(splits_cfg["calib_days"]),
        test_clean_days=tuple(splits_cfg["test_clean_days"]),
        test_injected_days=tuple(splits_cfg["test_injected_days"]),
    )

    final = pd.concat([
        combined[["ac_power"]],
        qc,
        twin_feats,
        evidence_feats,
        split_labels.to_frame(),
    ], axis=1)
    final.attrs["ac_nameplate_kw"] = final_capacity
    final.attrs["latitude"] = lat
    final.attrs["longitude"] = lon

    out_path = Path(f"data/processed/{plant_key}.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(out_path)

    # capacity is a frozen constant Layer 2/3 depend on — persist it alongside
    meta_path = Path(f"data/processed/{plant_key}_meta.yaml")
    with open(meta_path, "w") as fh:
        yaml.safe_dump({
            "ac_nameplate_kw": final_capacity,
            "latitude": lat,
            "longitude": lon,
            "n_rows": len(final),
            "n_clean": n_clean,
            "n_ceiling_violations_training": n_ceiling_violations_training,
            "n_ceiling_violations_fraud": n_ceiling_violations_fraud,
        }, fh)

    print(f"[{plant_key}] DONE -> {out_path} ({len(final)} rows, {len(final.columns)} columns)")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plant", required=True, choices=["plant_1", "plant_2"])
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--plants-config", default="config/plants.yaml")
    args = ap.parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)
    with open(args.plants_config) as fh:
        plants_config = yaml.safe_load(fh)

    run(args.plant, config, plants_config)
    return 0


if __name__ == "__main__":
    sys.exit(main())