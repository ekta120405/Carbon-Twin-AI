"""
src/layer3_anomaly/train_autoencoder_model.py

Trains and persists Layer 3.4's autoencoder for one or both plants. This
is the FIRST production training driver for this module -- previously it
was only ever trained transiently inside evaluation scripts, retrained
from scratch every single time it was needed (confirmed via grep before
this was written; see docs/TASK7_HANDOFF.md). Needed now because the
secondary-evidence escalation logic (src/layer3_anomaly/escalation.py)
depends on a real, reusable, saved model rather than a throwaway one.

Usage:
    python -m src.layer3_anomaly.train_autoencoder_model --plant plant_1
    python -m src.layer3_anomaly.train_autoencoder_model --plant plant_2
    python -m src.layer3_anomaly.train_autoencoder_model --plant all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from src.layer2_twin.physics_anchor import compute_physics_anchor
from src.layer2_twin.twin_model import PIGBQRTwin
from src.layer3_anomaly.residual import build_residual_table
from src.layer3_anomaly.autoencoder import train_and_calibrate, save_autoencoder

PLANTS = ["plant_1", "plant_2"]


def run(plant: str) -> None:
    with open(ROOT / "config" / "config.yaml") as f:
        config = yaml.safe_load(f)
    with open(ROOT / "config" / "plants.yaml") as f:
        plants_cfg = yaml.safe_load(f)
    meta = plants_cfg[plant]

    twin = PIGBQRTwin.load(str(ROOT / "reports" / "models" / plant))
    with open(ROOT / "data" / "processed" / f"{plant}_meta.yaml") as f:
        capacity_kw = yaml.safe_load(f)["ac_nameplate_kw"]

    df = pd.read_parquet(ROOT / "data" / "processed" / f"{plant}.parquet")
    anchor = compute_physics_anchor(df, meta["latitude"], meta["longitude"], capacity_kw)
    dfj = df.join(anchor)
    ts_table = build_residual_table(twin, dfj["physics_ac_kw"], dfj, capacity_kw, twin.feature_cols)

    daylight_dates = pd.Series(ts_table.index.date, index=ts_table.index)
    calib_dates = pd.Index(sorted(set(daylight_dates[ts_table["split"] == "calib"])))
    test_clean_dates = pd.Index(sorted(set(daylight_dates[ts_table["split"] == "test_clean"])))
    train_mask = ts_table["split"] == "train"

    ae_cfg = config["layer3"]["autoencoder"]
    print(f"[{plant}] training autoencoder ({len(train_mask[train_mask])} train rows)...")
    result, threshold, run_length = train_and_calibrate(
        ts_table, twin.feature_cols, train_mask, calib_dates, test_clean_dates,
        hidden=ae_cfg["hidden"], epochs=ae_cfg["epochs"], learning_rate=ae_cfg["learning_rate"],
    )
    print(f"[{plant}] calibrated: threshold={threshold:.4f}, run_length={run_length} "
          f"(final train loss={result.loss_history[-1]:.4f}, {len(result.loss_history)} epochs)")

    out_dir = ROOT / "reports" / "models" / plant
    save_autoencoder(result, ae_cfg["hidden"], threshold, run_length, str(out_dir))
    print(f"[{plant}] saved to {out_dir}/autoencoder.pt + autoencoder_meta.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plant", required=True, choices=PLANTS + ["all"])
    args = ap.parse_args()
    for plant in PLANTS if args.plant == "all" else [args.plant]:
        run(plant)
    return 0


if __name__ == "__main__":
    sys.exit(main())
