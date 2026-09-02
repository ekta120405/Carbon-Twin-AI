"""
EXPERIMENTAL. Honest split-validation of the "smart escalation" idea from
_experiment_ae_runlength.py / _experiment_cusum_smart.py.

PROBLEM BEING FIXED: every parameter chosen in those two scripts (the
near-threshold band width, the CUSUM escalation fraction) was picked by
directly testing its effect against the SAME 30 labeled replicates used
for final scoring -- exactly the "don't tune against TEST-INJECTED, touch
it once at the end" rule this project follows everywhere else (Overview
v2 SS8.1). The AE run-length itself IS honestly calibrated (from clean
calib+test_clean data only, zero label-peeking) -- only the escalation
BAND WIDTH and CUSUM FRACTION broke the rule.

FIX: split each plant's 30 replicates into a TUNE set (0-19) and a
HELD-OUT VALIDATION set (20-29). Sweep band-width x cusum-fraction
combinations using ONLY the tune set's labels to pick the best F1
configuration -- then evaluate that ONE chosen configuration on the
held-out set, which was never looked at while choosing anything. If the
gain survives on the untouched holdout, it's real. If it collapses back
toward baseline, it was overfit to the known 30 cases.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

import pandas as pd

from dashboard.lib.pipeline import PLANTS, load_plant_base
from src.evaluation.harness import score_all_replicates, _h_for_window
from src.evaluation.benchmark import merge_with_ground_truth, _metrics_block
from src.evaluation.ablations import train_plant_autoencoder
from src.evaluation._experiment_ae_runlength import ae_day_flag_with_runlength
from src.evaluation._experiment_cusum_smart import calibrate_ae_run_length
from src.layer2_twin.physics_anchor import compute_physics_anchor
from src.layer3_anomaly.residual import build_residual_table

TUNE_REPLICATES = list(range(0, 20))
HOLDOUT_REPLICATES = list(range(20, 30))

BAND_WIDTHS = [0.5, 0.6, 0.7, 0.8, 0.9]
CUSUM_FRACS = [0.0, 0.1, 0.2, 0.3, 0.5]


def build_full_df(plant: str, base: dict) -> tuple[pd.DataFrame, float, float, int]:
    twin, capacity_kw, meta = base["twin"], base["capacity_kw"], base["meta"]
    threshold = base["conformal_result"].threshold
    cusum_h = _h_for_window(base, 9)

    ae_result, ae_threshold = train_plant_autoencoder(plant, base)
    calibrated_rl = calibrate_ae_run_length(plant, base, ae_result, ae_threshold)

    scored = score_all_replicates(plant, base=base).reset_index()
    df = merge_with_ground_truth(scored, plant)

    ae_flags = {}
    for rep_id in range(30):
        rep_path = ROOT / "data" / "processed" / "injected" / plant / f"replicate_{rep_id:03d}.parquet"
        rep_df = pd.read_parquet(rep_path)
        anchor = compute_physics_anchor(rep_df, meta["latitude"], meta["longitude"], capacity_kw)
        dfj = rep_df.join(anchor)
        ts = build_residual_table(twin, dfj["physics_ac_kw"], dfj, capacity_kw, twin.feature_cols)
        day_flag = ae_day_flag_with_runlength(ts, ae_result, ae_threshold, calibrated_rl)
        for d, flagged in day_flag.items():
            ae_flags[(rep_id, d)] = bool(flagged)
    df["ae_flag"] = df.apply(lambda r: ae_flags.get((r["replicate"], r["date"]), False), axis=1)

    return df, threshold, cusum_h, calibrated_rl


def make_variant(df: pd.DataFrame, threshold: float, cusum_h: float, band_width: float, cusum_frac: float) -> pd.Series:
    near = (df["r_day"] > band_width * threshold) & (df["r_day"] <= threshold)
    elevated = df["cusum_s_t"] > (cusum_frac * cusum_h)
    corroborated = (df["ae_flag"] | elevated) & near
    return df["detected"] | corroborated


def main():
    plant_data = {}
    for plant in PLANTS:
        base = load_plant_base(plant)
        df, threshold, cusum_h, calibrated_rl = build_full_df(plant, base)
        print(f"[{plant}] AE run_length={calibrated_rl}, cusum_h={cusum_h}, threshold={threshold:.4f}")
        plant_data[plant] = dict(df=df, threshold=threshold, cusum_h=cusum_h)

    # --- Tune: sweep on TUNE_REPLICATES only, pick best pooled F1 ---
    tune_frames = {p: d["df"][d["df"]["replicate"].isin(TUNE_REPLICATES)] for p, d in plant_data.items()}
    holdout_frames = {p: d["df"][d["df"]["replicate"].isin(HOLDOUT_REPLICATES)] for p, d in plant_data.items()}

    best = None
    print("\n=== TUNE SET (replicates 0-19) sweep ===")
    for bw in BAND_WIDTHS:
        for cf in CUSUM_FRACS:
            variant_frames = []
            for p, d in plant_data.items():
                sub = tune_frames[p].copy()
                sub["variant"] = make_variant(sub, d["threshold"], d["cusum_h"], bw, cf)
                variant_frames.append(sub)
            pooled_tune = pd.concat(variant_frames)
            m = _metrics_block(pooled_tune, "variant")
            f1 = m["overall"]["f1"]
            spec = m["hard_negative_overall"]["specificity"]
            if best is None or f1 > best[0]:
                best = (f1, bw, cf, spec)
    print(f"Best on TUNE set: band_width={best[1]}, cusum_frac={best[2]}, tune_f1={best[0]:.3f}, tune_specificity={best[3]:.3f}")

    bw, cf = best[1], best[2]

    # --- Baseline on holdout ---
    holdout_pooled_base = pd.concat(holdout_frames.values())
    base_m = _metrics_block(holdout_pooled_base, "detected")
    print("\n=== HELD-OUT SET (replicates 20-29), NEVER used to choose bw/cf ===")
    print(f"baseline (production only) on holdout: recall={base_m['overall']['recall']:.3f} "
          f"precision={base_m['overall']['precision']:.3f} f1={base_m['overall']['f1']:.3f} "
          f"specificity={base_m['hard_negative_overall']['specificity']:.3f} "
          f"(n_positive={base_m['overall']['n_positive']}, n_negative={base_m['overall']['n_negative']})")

    # --- Chosen config on holdout ---
    variant_frames = []
    for p, d in plant_data.items():
        sub = holdout_frames[p].copy()
        sub["variant"] = make_variant(sub, d["threshold"], d["cusum_h"], bw, cf)
        variant_frames.append(sub)
    holdout_pooled_variant = pd.concat(variant_frames)
    hm = _metrics_block(holdout_pooled_variant, "variant")
    print(f"chosen config (bw={bw}, cf={cf}) on holdout: recall={hm['overall']['recall']:.3f} "
          f"precision={hm['overall']['precision']:.3f} f1={hm['overall']['f1']:.3f} "
          f"specificity={hm['hard_negative_overall']['specificity']:.3f}")

    # --- Also report chosen config on the FULL pooled set (all 30), for comparison to earlier reporting ---
    variant_frames_full = []
    for p, d in plant_data.items():
        sub = d["df"].copy()
        sub["variant"] = make_variant(sub, d["threshold"], d["cusum_h"], bw, cf)
        variant_frames_full.append(sub)
    full_pooled = pd.concat(variant_frames_full)
    fm = _metrics_block(full_pooled, "variant")
    fm_base = _metrics_block(full_pooled, "detected")
    print(f"\nFor reference -- chosen config on ALL 30 replicates pooled: recall={fm['overall']['recall']:.3f} "
          f"precision={fm['overall']['precision']:.3f} f1={fm['overall']['f1']:.3f} "
          f"specificity={fm['hard_negative_overall']['specificity']:.3f}")
    print(f"(vs. baseline on all 30: recall={fm_base['overall']['recall']:.3f} f1={fm_base['overall']['f1']:.3f} "
          f"specificity={fm_base['hard_negative_overall']['specificity']:.3f})")

    result = dict(
        chosen_band_width=bw, chosen_cusum_frac=cf,
        tune_f1=best[0],
        holdout_baseline=dict(overall=base_m["overall"], specificity=base_m["hard_negative_overall"]["specificity"]),
        holdout_chosen=dict(overall=hm["overall"], specificity=hm["hard_negative_overall"]["specificity"]),
        full_pooled_baseline=dict(overall=fm_base["overall"], specificity=fm_base["hard_negative_overall"]["specificity"]),
        full_pooled_chosen=dict(overall=fm["overall"], specificity=fm["hard_negative_overall"]["specificity"], per_type=fm["per_type"]),
    )
    with open(ROOT / "reports" / "tables" / "_experiment_split_validate.json", "w") as f:
        json.dump(result, f, indent=2, default=str)


if __name__ == "__main__":
    main()
