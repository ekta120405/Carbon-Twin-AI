"""
src/layer2_twin/compare_baselines.py

Runs Ridge and LSTM baselines against the SAME twin_* feature set the
PI-GBQR twin uses, and reports the same regime-stratified metrics format
used in train_twin.py -- direct, apples-to-apples comparison against the
twin and the physics-anchor-only baseline already on record.

Both baselines predict ac_power DIRECTLY, not the physics-residual target
(Task 2 SS2.5, deliberate design -- see baselines.py docstring). This is
the ablation that argues for (or against) the physics-informed approach.

Usage:
    python -m src.layer2_twin.compare_baselines --plant plant_1
    python -m src.layer2_twin.compare_baselines --plant plant_2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.layer2_twin.physics_anchor import compute_physics_anchor
from src.layer2_twin.baselines import fit_ridge_baseline, build_sequences, LSTMBaseline
from src.layer2_twin.train_twin import regime_stratified_metrics


def run(plant_key: str, config_path: str = "config/config.yaml",
        plants_path: str = "config/plants.yaml") -> None:
    with open(config_path) as fh:
        config = yaml.safe_load(fh)
    with open(plants_path) as fh:
        plants = yaml.safe_load(fh)

    meta = plants[plant_key]
    lat, lon = meta["latitude"], meta["longitude"]
    bcfg = config["layer2"]["baselines"]

    df = pd.read_parquet(f"data/processed/{plant_key}.parquet")
    with open(f"data/processed/{plant_key}_meta.yaml") as fh:
        pmeta = yaml.safe_load(fh)
    capacity_kw = pmeta["ac_nameplate_kw"]

    print(f"[{plant_key}] computing physics anchor (for regime stratification only)...")
    anchor = compute_physics_anchor(
        df, latitude=lat, longitude=lon, ac_nameplate_kw=capacity_kw,
        gamma_pdc=config["layer2"]["physics_anchor"]["gamma_pdc"],
        dc_ac_ratio=config["layer2"]["physics_anchor"]["inverter_ac_dc_ratio"],
        system_losses=config["layer2"]["physics_anchor"]["system_losses"],
    )
    df = df.join(anchor)

    twin_feature_cols = [c for c in df.columns if c.startswith("twin_")]
    train_mask = (df["split"] == "train") & df["is_clean"]
    test_mask = df["split"] == "test_clean"

    X_train = df.loc[train_mask, twin_feature_cols]
    y_train = df.loc[train_mask, "ac_power"]
    X_test = df.loc[test_mask, twin_feature_cols]
    y_test = df.loc[test_mask, "ac_power"]
    kt_test = df.loc[test_mask, "twin_kt"]

    print(f"[{plant_key}] training Ridge on {len(X_train)} rows...")
    ridge = fit_ridge_baseline(X_train, y_train, alpha=bcfg["ridge"]["alpha"])
    ridge_pred = pd.Series(ridge.predict(X_test), index=X_test.index).clip(lower=0)
    ridge_metrics = regime_stratified_metrics(y_test, ridge_pred, kt_test, capacity_kw)
    ridge_train_pred = pd.Series(ridge.predict(X_train), index=X_train.index).clip(lower=0)
    ridge_train_metrics = regime_stratified_metrics(y_train, ridge_train_pred,
                                                     df.loc[train_mask, "twin_kt"], capacity_kw)

    print(f"\n[{plant_key}] Ridge baseline (test_clean):")
    for regime, m in ridge_metrics.items():
        if m["mae"] is not None:
            print(f"  {regime:32s} n={m['n']:4d}  nRMSE={m['nrmse_pct']:5.2f}%  R2={m['r2']:.3f}")

    print(f"\n[{plant_key}] building LSTM sequences (seq_len={bcfg['lstm']['seq_len']})...")
    np.random.seed(bcfg["seed"])
    try:
        import torch
        torch.manual_seed(bcfg["seed"])
    except ImportError:
        torch = None

    X_train_seq, y_train_seq, train_end_idx = build_sequences(
        X_train, y_train, seq_len=bcfg["lstm"]["seq_len"]
    )
    X_test_seq, y_test_seq, test_end_idx = build_sequences(
        X_test, y_test, seq_len=bcfg["lstm"]["seq_len"]
    )
    print(f"[{plant_key}] LSTM sequences: {len(X_train_seq)} train, {len(X_test_seq)} test "
          f"(raw rows: {len(X_train)} train, {len(X_test)} test -- gap fragmentation check)")

    if len(X_train_seq) < 20 or len(X_test_seq) < 5:
        print(f"[{plant_key}] WARNING: too few valid sequences after gap-filtering to train/evaluate "
              f"LSTM meaningfully. Skipping LSTM for this plant.")
        lstm_metrics = None
        lstm_train_metrics = None
    else:
        lstm = LSTMBaseline(n_features=len(twin_feature_cols),
                             hidden_size=bcfg["lstm"]["hidden_size"],
                             num_layers=bcfg["lstm"]["num_layers"])
        lstm.fit(X_train_seq, y_train_seq, epochs=bcfg["lstm"]["epochs"],
                 lr=bcfg["lstm"]["learning_rate"], batch_size=bcfg["lstm"]["batch_size"])

        lstm_test_pred = pd.Series(lstm.predict(X_test_seq), index=test_end_idx).clip(lower=0)
        kt_test_seq = df.loc[test_end_idx, "twin_kt"]
        lstm_metrics = regime_stratified_metrics(
            df.loc[test_end_idx, "ac_power"], lstm_test_pred, kt_test_seq, capacity_kw
        )

        lstm_train_pred = pd.Series(lstm.predict(X_train_seq), index=train_end_idx).clip(lower=0)
        kt_train_seq = df.loc[train_end_idx, "twin_kt"]
        lstm_train_metrics = regime_stratified_metrics(
            df.loc[train_end_idx, "ac_power"], lstm_train_pred, kt_train_seq, capacity_kw
        )

        print(f"\n[{plant_key}] LSTM baseline (test_clean):")
        for regime, m in lstm_metrics.items():
            if m["mae"] is not None:
                print(f"  {regime:32s} n={m['n']:4d}  nRMSE={m['nrmse_pct']:5.2f}%  R2={m['r2']:.3f}")

        print(f"\n[{plant_key}] LSTM train-vs-test R2 (overfitting check):")
        for regime in lstm_train_metrics:
            tr = lstm_train_metrics[regime]["r2"]
            te = lstm_metrics[regime]["r2"]
            if tr is not None and te is not None:
                print(f"  {regime:32s} train_R2={tr:.3f}  test_R2={te:.3f}  gap={tr-te:.3f}")

    out_dir = Path("reports/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{plant_key}_baseline_metrics.json", "w") as fh:
        json.dump({
            "ridge": ridge_metrics,
            "ridge_train": ridge_train_metrics,
            "lstm": lstm_metrics,
            "lstm_train": lstm_train_metrics,
        }, fh, indent=2, default=str)
    print(f"\n[{plant_key}] baseline metrics saved to {out_dir}/{plant_key}_baseline_metrics.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plant", required=True, choices=["plant_1", "plant_2"])
    args = ap.parse_args()
    run(args.plant)
    return 0


if __name__ == "__main__":
    sys.exit(main())
