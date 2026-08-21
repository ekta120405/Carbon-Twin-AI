"""
src/layer2_twin/train_twin.py

Orchestrates Layer 2.1-2.3 for one plant: physics anchor -> target -> PI-GBQR
training -> regime-stratified evaluation on TEST-CLEAN. Trains ONLY on the
'train' split block from data/processed/<plant>.parquet (real, un-injected
data) -- Layer 2 never sees the injector's output; that's Layer 3's job.

Usage:
    python -m src.layer2_twin.train_twin --plant plant_1
    python -m src.layer2_twin.train_twin --plant plant_2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.layer2_twin.physics_anchor import compute_physics_anchor, compute_calibration_ratio
from src.layer2_twin.twin_model import PIGBQRTwin, QUANTILES


def regime_stratified_metrics(y_true: pd.Series, y_pred: pd.Series, clearsky_index: pd.Series,
                               ac_nameplate_kw: float) -> dict:
    """MAE, RMSE, nRMSE (% of capacity), MBE, R2 -- overall AND split by
    cloud regime (clear / partly cloudy / overcast), per Overview v2 SS9.1.
    A single aggregate number hides exactly the failure mode (cloudy-day
    accuracy) that matters most for this project."""
    df = pd.DataFrame({"y": y_true, "yhat": y_pred, "kt": clearsky_index}).dropna()

    def _metrics(sub: pd.DataFrame) -> dict:
        if len(sub) < 5:
            return {"n": len(sub), "mae": None, "rmse": None, "nrmse_pct": None,
                    "mbe": None, "r2": None}
        err = sub["yhat"] - sub["y"]
        mae = float(err.abs().mean())
        rmse = float(np.sqrt((err**2).mean()))
        mbe = float(err.mean())
        ss_res = float((err**2).sum())
        ss_tot = float(((sub["y"] - sub["y"].mean())**2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-9 else None
        return {"n": len(sub), "mae": mae, "rmse": rmse,
                "nrmse_pct": 100 * rmse / ac_nameplate_kw, "mbe": mbe, "r2": r2}

    regimes = {
        "overall": df,
        "clear (Kt>=0.75)": df[df["kt"] >= 0.75],
        "partly_cloudy (0.4<=Kt<0.75)": df[(df["kt"] >= 0.4) & (df["kt"] < 0.75)],
        "overcast (Kt<0.4)": df[df["kt"] < 0.4],
    }
    return {name: _metrics(sub) for name, sub in regimes.items()}


def run(plant_key: str, config_path: str = "config/config.yaml",
        plants_path: str = "config/plants.yaml") -> None:
    with open(config_path) as fh:
        config = yaml.safe_load(fh)
    with open(plants_path) as fh:
        plants = yaml.safe_load(fh)

    meta = plants[plant_key]
    lat, lon = meta["latitude"], meta["longitude"]

    df = pd.read_parquet(f"data/processed/{plant_key}.parquet")
    with open(f"data/processed/{plant_key}_meta.yaml") as fh:
        pmeta = yaml.safe_load(fh)
    capacity_kw = pmeta["ac_nameplate_kw"]

    print(f"[{plant_key}] computing physics anchor (pvlib PVWatts chain)...")
    anchor = compute_physics_anchor(
        df, latitude=lat, longitude=lon, ac_nameplate_kw=capacity_kw,
        tilt_deg=meta.get("tilt_deg"), azimuth_deg=meta.get("azimuth_deg", 180.0),
        gamma_pdc=config["layer2"]["physics_anchor"]["gamma_pdc"],
        dc_ac_ratio=config["layer2"]["physics_anchor"]["inverter_ac_dc_ratio"],
        system_losses=config["layer2"]["physics_anchor"]["system_losses"],
    )
    df = df.join(anchor)

    cr = compute_calibration_ratio(df["ac_power"], df["physics_ac_kw"], df["twin_kt"])
    print(f"[{plant_key}] calibration ratio (claimed/physics on clear-sky days): {cr:.4f}")
    if cr > 1.15:
        print(f"[{plant_key}]   NOTE: CR > 1.15 -- global indicator of possible systematic "
              f"over-claiming, independent of any per-timestamp flag. Report in Layer 4 output.")

    twin_feature_cols = [c for c in df.columns if c.startswith("twin_")]
    train_mask = (df["split"] == "train") & df["is_clean"]
    test_mask = df["split"] == "test_clean"

    X_train = df.loc[train_mask, twin_feature_cols]
    print(f"[{plant_key}] training PI-GBQR on {len(X_train)} clean daylight rows "
          f"({len(twin_feature_cols)} twin_* features)...")

    twin = PIGBQRTwin(
        lgb_params=config["layer2"]["model"],
        monotone_map={
            "twin_ghi": config["layer2"]["monotone"]["poa_global"],
            "twin_clearsky_ghi": config["layer2"]["monotone"]["clearsky_index"],
            "twin_kt": config["layer2"]["monotone"]["clearsky_index"],
            "twin_cell_temp": config["layer2"]["monotone"]["temp_cell"],
        },
    )
    twin.fit(X_train, df.loc[train_mask, "physics_ac_kw"], df.loc[train_mask, "ac_power"], capacity_kw)

    X_test = df.loc[test_mask, twin_feature_cols]
    preds = twin.predict(X_test, df.loc[test_mask, "physics_ac_kw"], df.loc[test_mask, "twin_kt"])
    
    metrics = regime_stratified_metrics(
        df.loc[test_mask, "ac_power"], preds["expected_q50"],
        df.loc[test_mask, "twin_kt"], capacity_kw,
    )
    print(f"\n[{plant_key}] TEST-CLEAN metrics (q50 vs claimed), regime-stratified:")
    for regime, m in metrics.items():
        if m["mae"] is not None:
            print(f"  {regime:32s}  n={m['n']:4d}  MAE={m['mae']:7.2f}  "
                  f"RMSE={m['rmse']:7.2f}  nRMSE={m['nrmse_pct']:5.2f}%  "
                  f"MBE={m['mbe']:7.2f}  R2={m['r2']:.3f}")
        else:
            print(f"  {regime:32s}  n={m['n']:4d}  (too few rows)")

    physics_only_metrics = regime_stratified_metrics(
        df.loc[test_mask, "ac_power"], df.loc[test_mask, "physics_ac_kw"],
        df.loc[test_mask, "twin_kt"], capacity_kw,
    )
    print(f"\n[{plant_key}] Physics-anchor-ONLY baseline (no ML correction), for comparison:")
    for regime, m in physics_only_metrics.items():
        if m["mae"] is not None:
            print(f"  {regime:32s}  nRMSE={m['nrmse_pct']:5.2f}%  R2={m['r2']:.3f}")

    out_dir = Path(f"reports/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{plant_key}_twin_metrics.json", "w") as fh:
        json.dump({"calibration_ratio": cr, "pigbqr": metrics, "physics_only": physics_only_metrics},
                   fh, indent=2, default=str)

    model_dir = Path(f"reports/models/{plant_key}")
    twin.save(str(model_dir))
    print(f"\n[{plant_key}] twin saved to {model_dir}/ (LightGBM text format, version-stable)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plant", required=True, choices=["plant_1", "plant_2"])
    args = ap.parse_args()
    run(args.plant)
    return 0


if __name__ == "__main__":
    sys.exit(main())
