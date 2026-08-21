"""
deep_benchmark.py

Runs in .venv-deep ONLY (see requirements-deep.txt, Task 3). This script
NEVER imports from src/ -- it communicates with the main pipeline purely
through files, per the isolation boundary established in Task 3:

    reads:  data/processed/<plant>.parquet   (twin_* features, ac_power, split)
    writes: reports/benchmarks/<plant>_<model>_predictions.parquet
            reports/benchmarks/<plant>_<model>_metrics.json

Uses `darts` (Task 2 SS2.5 items 5-6): TFTModel and TSMixerModel, both with
native quantile output via a QuantileRegression likelihood, matching the
PI-GBQR twin's q10/q50/q90 output contract so Layer 3 can consume either
model's predictions identically.

TRAINING SPLIT: trains on 'train' ONLY, not 'train'+'calib' -- 'calib'
stays reserved for Layer 3's future conformal calibration, matching the
twin/Ridge/LSTM's train mask exactly.

EVALUATION METHOD: uses historical_forecasts() with forecast_horizon=1,
retrain=False -- independent one-step-ahead predictions at each test
timestamp using REAL known covariates, matching how the twin/Ridge/LSTM
are evaluated (not a single long recursive multi-step rollout). The
series passed in is explicitly truncated with drop_after() to end right
after test_clean -- without this, historical_forecasts continues
predicting all the way to the end of the full series (into
test_injected, reserved for Layer 3), confirmed empirically: an
unbounded run produced n=906 instead of the correct ~480 matching
test_clean's actual row count.

TIMEZONE HANDLING: darts silently strips tz info when building TimeSeries
objects. Any timestamp compared against a TimeSeries' internal index
(train_end, test_start, test_end) must be tz-naive first.

MISSING VALUES (confirmed empirically): is_clean-filtered rows reindexed
to a strict 15-min frequency produce ~67% NaN rows for plant_1
(2146/3210). Filled via darts' MissingValuesFiller (linear interpolation)
before training -- unfilled, this produced NaN training loss.

SCALING: target and covariates are scaled via darts' Scaler before
training and inverse-transformed on output.

If this environment fights you for more than 30 minutes to install (Task 3
SS3.1), stop and run this exact script on Google Colab's free T4 instead --
export data/processed/<plant>.parquet, upload, run, download the two output
files, and place them back under reports/benchmarks/. Nothing else in the
pipeline needs to know which machine produced them.

Usage (from within .venv-deep):
    python deep_benchmark.py --plant plant_1 --model tft
    python deep_benchmark.py --plant plant_1 --model tsmixer
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEQ_LEN_IN = 32
SEQ_LEN_OUT = 1
QUANTILES = [0.1, 0.5, 0.9]


def load_plant(plant_key: str) -> pd.DataFrame:
    df = pd.read_parquet(f"data/processed/{plant_key}.parquet")
    twin_cols = [c for c in df.columns if c.startswith("twin_")]
    return df[twin_cols + ["ac_power", "split", "is_clean"]].copy()


def build_darts_series(df: pd.DataFrame):
    from darts import TimeSeries
    from darts.dataprocessing.transformers import MissingValuesFiller

    clean = df[df["is_clean"]].copy()
    clean = clean.asfreq("15min")

    twin_cols = [c for c in df.columns if c.startswith("twin_")]
    target_raw = TimeSeries.from_series(clean["ac_power"], fill_missing_dates=True, freq="15min")
    covariates_raw = TimeSeries.from_dataframe(
        clean[twin_cols], fill_missing_dates=True, freq="15min"
    )

    filler = MissingValuesFiller()
    target = filler.transform(target_raw)
    covariates = filler.transform(covariates_raw)

    return target, covariates, clean


def train_tft(target, covariates, train_end: pd.Timestamp):
    from darts.models import TFTModel
    from darts.utils.likelihood_models import QuantileRegression

    model = TFTModel(
        input_chunk_length=SEQ_LEN_IN,
        output_chunk_length=SEQ_LEN_OUT,
        hidden_size=16,
        lstm_layers=1,
        num_attention_heads=2,
        dropout=0.1,
        batch_size=32,
        n_epochs=30,
        likelihood=QuantileRegression(quantiles=QUANTILES),
        add_relative_index=True,
        random_state=42,
        pl_trainer_kwargs={"accelerator": "cpu"},
    )
    model.fit(target.split_before(train_end)[0], past_covariates=covariates)
    return model


def train_tsmixer(target, covariates, train_end: pd.Timestamp):
    from darts.models import TSMixerModel
    from darts.utils.likelihood_models import QuantileRegression

    model = TSMixerModel(
        input_chunk_length=SEQ_LEN_IN,
        output_chunk_length=SEQ_LEN_OUT,
        hidden_size=32,
        num_blocks=2,
        dropout=0.1,
        batch_size=32,
        n_epochs=30,
        likelihood=QuantileRegression(quantiles=QUANTILES),
        random_state=42,
        pl_trainer_kwargs={"accelerator": "cpu"},
    )
    model.fit(target.split_before(train_end)[0], past_covariates=covariates)
    return model


def evaluate_and_save(model, target, covariates, target_scaler, clean_df: pd.DataFrame,
                       plant_key: str, model_name: str) -> None:
    test_mask = clean_df["split"] == "test_clean"
    n_test_points = int(test_mask.sum())
    if n_test_points == 0:
        print(f"WARNING: no test_clean points in cleaned+resampled data for {plant_key}")
        return

    test_start = clean_df.index[test_mask][0].tz_localize(None)
    test_end = clean_df.index[test_mask][-1].tz_localize(None)

    print(f"[{plant_key}/{model_name}] running historical_forecasts "
          f"(forecast_horizon=1, retrain=False) from {test_start} to {test_end}...")

    bounded_target = target.drop_after(test_end + pd.Timedelta("15min"))

    preds_scaled = model.historical_forecasts(
        series=bounded_target,
        past_covariates=covariates,
        start=test_start,
        forecast_horizon=1,
        stride=1,
        retrain=False,
        num_samples=200,
        last_points_only=True,
        verbose=True,
    )
    preds = target_scaler.inverse_transform(preds_scaled)

    q_preds = {q: preds.quantile(q).to_series() for q in QUANTILES}

    out_df = pd.DataFrame({
        "expected_q10": q_preds[0.1], "expected_q50": q_preds[0.5], "expected_q90": q_preds[0.9],
    }).clip(lower=0)

    out_dir = Path("reports/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_dir / f"{plant_key}_{model_name}_predictions.parquet")

    actual = clean_df["ac_power"].reset_index(drop=True).iloc[-len(out_df):]
    actual.index = out_df.index
    err = out_df["expected_q50"] - actual
    metrics = {
        "n": len(actual),
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "mbe": float(err.mean()),
    }
    with open(out_dir / f"{plant_key}_{model_name}_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"[{plant_key}/{model_name}] MAE={metrics['mae']:.2f}  RMSE={metrics['rmse']:.2f}  "
          f"MBE={metrics['mbe']:.2f}  (n={metrics['n']})")
    print(f"[{plant_key}/{model_name}] -> {out_dir}/{plant_key}_{model_name}_predictions.parquet")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plant", required=True, choices=["plant_1", "plant_2"])
    ap.add_argument("--model", required=True, choices=["tft", "tsmixer"])
    args = ap.parse_args()

    from darts.dataprocessing.transformers import Scaler

    df = load_plant(args.plant)
    target_raw, covariates_raw, clean_df = build_darts_series(df)

    target_scaler = Scaler()
    covariates_scaler = Scaler()
    target = target_scaler.fit_transform(target_raw)
    covariates = covariates_scaler.fit_transform(covariates_raw)

    train_end_mask = clean_df["split"] == "train"
    train_end = clean_df.index[train_end_mask][-1].tz_localize(None)

    print(f"[{args.plant}] training {args.model} on data up to {train_end}...")
    if args.model == "tft":
        model = train_tft(target, covariates, train_end)
    else:
        model = train_tsmixer(target, covariates, train_end)

    evaluate_and_save(model, target, covariates, target_scaler, clean_df, args.plant, args.model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
