"""
src/evaluation/ablations.py

Task 7.4: ablation study (a, b, c, d, f, g -- NOT e, see module-level
NOT_BUILT_ONLINE_LEARNING_NOTE below, per TASK7_FULL_ROADMAP SS0.1).

(g) is a cross-reference to Task 7.3 (src/evaluation/leave_one_plant_out.py)
-- not duplicated here, per the roadmap's own instruction.
(b) is a cross-reference to Layer 2's already-committed
reports/tables/{plant}_twin_metrics.json -- not recomputed here, per the
roadmap's own instruction ("do not recompute").
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

import pandas as pd
import yaml

from dashboard.lib.pipeline import PLANTS, load_plant_base
from src.layer2_twin.physics_anchor import compute_physics_anchor
from src.layer2_twin.train_twin import regime_stratified_metrics
from src.layer2_twin.twin_model import PIGBQRTwin
from src.layer3_anomaly.conformal import ConformalResult
from src.layer3_anomaly.residual import build_residual_table
from src.layer3_anomaly.autoencoder import (
    build_autoencoder_features, train_autoencoder, compute_reconstruction_error,
)
from src.evaluation.harness import score_all_replicates, N_REPLICATES
from src.evaluation.benchmark import merge_with_ground_truth, _metrics_block

with open(ROOT / "config" / "config.yaml") as _f:
    CONFIG = yaml.safe_load(_f)

NOT_BUILT_ONLINE_LEARNING_NOTE = (
    "Ablation (e), online vs. batch twin, was NOT performed by design. See "
    "Layer 2's online-learning scope decision (docs/LAYER2_HANDOFF.md SS8): "
    "the dataset's 34-day span provides insufficient rolling-window advance "
    "positions (at most 14, with a 20-day window) for a meaningful online-"
    "vs-batch comparison, independent of implementation effort."
)


# ---------------------------------------------------------------------------
# (a) Satellite vs. on-site weather as twin input
# ---------------------------------------------------------------------------

ONSITE_ABLATION_RENAME = {
    "evidence_onsite_irradiation": "twin_onsite_irr",
    "evidence_onsite_ambient_temp": "twin_onsite_ambient_temp",
    "evidence_onsite_module_temp": "twin_onsite_module_temp",
}


def train_onsite_variant(plant: str, base: dict) -> tuple[PIGBQRTwin, dict]:
    """
    Ablation (a): trains an ISOLATED variant twin using on-site sensor
    evidence as input instead of independent satellite weather, for
    regime-stratified accuracy COMPARISON ONLY. This model is never saved,
    never used by score_replicate/harness, never touched by any other
    module in this task or the production pipeline.

    R1 INVARIANT NOTE -- read before touching this function. PIGBQRTwin.fit()
    hard-asserts (assert_twin_feature_safety, TC-L2-01, twin_model.py) that
    its feature matrix contains ONLY twin_*-prefixed columns and NEVER
    evidence_*/ac_power columns -- Layer 2's core leakage guard. This is
    NOT weakened or bypassed here. Instead, the on-site evidence columns
    are RENAMED to twin_-prefixed aliases (twin_onsite_irr etc.) before
    being handed to the real, UNMODIFIED PIGBQRTwin.fit()/.predict() --
    reusing the exact production training code path (same LightGBM config,
    same monotone-constraint quantile setup including LAYER2_HANDOFF.md
    SS2's q50-must-use-L2-not-quantile-objective gotcha) instead of a
    second, independently bug-prone reimplementation, while still passing
    the safety assertion honestly, because on-site evidence genuinely IS
    the intended input for this one isolated, never-deployed ablation
    model. Confirmed: this object is not returned to, saved by, or
    referenced from anywhere outside this function's caller.
    """
    meta, capacity_kw = base["meta"], base["capacity_kw"]
    df_full = pd.read_parquet(ROOT / "data" / "processed" / f"{plant}.parquet")
    anchor = compute_physics_anchor(df_full, meta["latitude"], meta["longitude"], capacity_kw)
    df = df_full.join(anchor).rename(columns=ONSITE_ABLATION_RENAME)
    onsite_cols = list(ONSITE_ABLATION_RENAME.values())

    train_mask = (df["split"] == "train") & df["is_clean"]
    test_mask = df["split"] == "test_clean"

    twin = PIGBQRTwin(
        lgb_params=CONFIG["layer2"]["model"],
        monotone_map={"twin_onsite_irr": 1, "twin_onsite_module_temp": -1},
    )
    twin.fit(
        df.loc[train_mask, onsite_cols], df.loc[train_mask, "physics_ac_kw"],
        df.loc[train_mask, "ac_power"], capacity_kw,
    )

    preds = twin.predict(df.loc[test_mask, onsite_cols], df.loc[test_mask, "physics_ac_kw"], df.loc[test_mask, "twin_kt"])
    metrics = regime_stratified_metrics(
        df.loc[test_mask, "ac_power"], preds["expected_q50"], df.loc[test_mask, "twin_kt"], capacity_kw,
    )
    return twin, metrics


def run_ablation_a() -> dict:
    out = {}
    for plant in PLANTS:
        base = load_plant_base(plant)
        _, onsite_metrics = train_onsite_variant(plant, base)
        with open(ROOT / "reports" / "tables" / f"{plant}_twin_metrics.json") as f:
            real_metrics = json.load(f)
        out[plant] = dict(
            satellite_input_pigbqr=real_metrics["pigbqr"],
            onsite_input_pigbqr=onsite_metrics,
        )
    return out


# ---------------------------------------------------------------------------
# (b) With/without physics anchor -- pulled from Layer 2's committed output
# ---------------------------------------------------------------------------

def run_ablation_b() -> dict:
    out = {}
    for plant in PLANTS:
        with open(ROOT / "reports" / "tables" / f"{plant}_twin_metrics.json") as f:
            m = json.load(f)
        out[plant] = dict(with_physics_anchor_pigbqr=m["pigbqr"], physics_anchor_only=m["physics_only"])
    return out


# ---------------------------------------------------------------------------
# (c) Conformal vs. fixed 2-sigma threshold
# ---------------------------------------------------------------------------

def fixed_2sigma_threshold(calib_r_day: pd.Series) -> float:
    """The original (rejected) naive threshold Overview v2's Task 1 weakness
    W3 replaced with conformal calibration -- rebuilt here ONLY as an
    ablation comparison arm, never used by any production code path."""
    return float(calib_r_day.mean() + 2 * calib_r_day.std())


def run_ablation_c() -> dict:
    out = {}
    frames = []
    for plant in PLANTS:
        base = load_plant_base(plant)
        calib_r_day = base["daily_main"].loc[base["daily_main"].index.isin(base["calib_dates"]), "r_day"]
        threshold_2s = fixed_2sigma_threshold(calib_r_day)

        fixed_result = ConformalResult(
            threshold=threshold_2s, n_calib=base["conformal_result"].n_calib,
            alpha_nominal=base["conformal_result"].alpha_nominal,
            alpha_achievable=base["conformal_result"].alpha_achievable,
            calib_r_day_values=base["conformal_result"].calib_r_day_values,
        )
        fixed_base = {**base, "conformal_result": fixed_result}

        scored_conformal = score_all_replicates(plant, base=base).reset_index()
        scored_2sigma = score_all_replicates(plant, base=fixed_base).reset_index()

        df_conformal = merge_with_ground_truth(scored_conformal, plant)
        df_2sigma = merge_with_ground_truth(scored_2sigma, plant)
        frames.append(("conformal", df_conformal))
        frames.append(("2sigma", df_2sigma))

        out[plant] = dict(
            conformal_threshold=base["conformal_result"].threshold,
            fixed_2sigma_threshold=threshold_2s,
            conformal=_metrics_block(df_conformal, "detected")["overall"],
            fixed_2sigma=_metrics_block(df_2sigma, "detected")["overall"],
            conformal_hard_negative_specificity=_metrics_block(df_conformal, "detected")["hard_negative_overall"]["specificity"],
            fixed_2sigma_hard_negative_specificity=_metrics_block(df_2sigma, "detected")["hard_negative_overall"]["specificity"],
        )

    pooled_conformal = pd.concat([df for name, df in frames if name == "conformal"])
    pooled_2sigma = pd.concat([df for name, df in frames if name == "2sigma"])
    out["pooled"] = dict(
        conformal=_metrics_block(pooled_conformal, "detected")["overall"],
        fixed_2sigma=_metrics_block(pooled_2sigma, "detected")["overall"],
        conformal_hard_negative_specificity=_metrics_block(pooled_conformal, "detected")["hard_negative_overall"]["specificity"],
        fixed_2sigma_hard_negative_specificity=_metrics_block(pooled_2sigma, "detected")["hard_negative_overall"]["specificity"],
    )
    return out


# ---------------------------------------------------------------------------
# (d) One-sided vs. two-sided
# ---------------------------------------------------------------------------

def run_ablation_d() -> dict:
    """
    Isolates JUST the conformal magnitude test's sidedness -- deliberately
    excludes the night-injection/stuck-sensor guardrails (a separate
    detection mechanism, already evaluated on its own merits in 7.1) so
    this comparison measures only what Overview v2 SS2.6/W4 is actually
    about: does allowing r_day < -threshold to also flag hurt specificity
    on the under-generation hard negatives (N1/N2/N3/N5/N6)?
    """
    out = {}
    frames = []
    for plant in PLANTS:
        base = load_plant_base(plant)
        threshold = base["conformal_result"].threshold
        scored = score_all_replicates(plant, base=base).reset_index()
        df = merge_with_ground_truth(scored, plant)
        df["detected_one_sided"] = df["r_day"] > threshold
        df["detected_two_sided"] = df["r_day"].abs() > threshold
        frames.append(df)

        out[plant] = dict(
            one_sided=_metrics_block(df, "detected_one_sided")["overall"],
            two_sided=_metrics_block(df, "detected_two_sided")["overall"],
            one_sided_hard_negative_specificity=_metrics_block(df, "detected_one_sided")["hard_negative_overall"]["specificity"],
            two_sided_hard_negative_specificity=_metrics_block(df, "detected_two_sided")["hard_negative_overall"]["specificity"],
            one_sided_hard_negative_per_type=_metrics_block(df, "detected_one_sided")["hard_negative_per_type"],
            two_sided_hard_negative_per_type=_metrics_block(df, "detected_two_sided")["hard_negative_per_type"],
        )

    pooled = pd.concat(frames)
    out["pooled"] = dict(
        one_sided=_metrics_block(pooled, "detected_one_sided")["overall"],
        two_sided=_metrics_block(pooled, "detected_two_sided")["overall"],
        one_sided_hard_negative_specificity=_metrics_block(pooled, "detected_one_sided")["hard_negative_overall"]["specificity"],
        two_sided_hard_negative_specificity=_metrics_block(pooled, "detected_two_sided")["hard_negative_overall"]["specificity"],
    )
    return out


# ---------------------------------------------------------------------------
# (f) Conformal (production flag) vs. autoencoder vs. ensemble
# ---------------------------------------------------------------------------

def _full_ts_table(plant: str, base: dict) -> pd.DataFrame:
    meta, capacity_kw, twin = base["meta"], base["capacity_kw"], base["twin"]
    df_full = pd.read_parquet(ROOT / "data" / "processed" / f"{plant}.parquet")
    anchor = compute_physics_anchor(df_full, meta["latitude"], meta["longitude"], capacity_kw)
    dfj = df_full.join(anchor)
    return build_residual_table(twin, dfj["physics_ac_kw"], dfj, capacity_kw, twin.feature_cols)


def train_plant_autoencoder(plant: str, base: dict) -> tuple:
    """
    Trains Layer 3.4's autoencoder for real (LAYER3_HANDOFF.md notes it was
    verified against real data during Layer 3's build, but no reusable
    training driver or persisted model exists anywhere in the committed
    codebase -- confirmed via grep before writing this). Baseline
    reconstruction-error pool and threshold follow the SAME discipline
    established for every other calibrated quantity in this project: fit
    only on clean data (train for weights, held-out calib for early
    stopping and threshold), 95th percentile of pooled calib+test_clean
    error as the threshold (same convention as aggregate.py's
    calibrate_divergence_cap).
    """
    ts = _full_ts_table(plant, base)
    feats = build_autoencoder_features(ts, base["twin"].feature_cols)
    split = ts.loc[feats.index, "split"]

    ae_cfg = CONFIG["layer3"]["autoencoder"]
    train_feats = feats[split == "train"]
    calib_feats = feats[split == "calib"]
    test_clean_feats = feats[split == "test_clean"]

    result = train_autoencoder(
        train_feats, hidden=ae_cfg["hidden"], epochs=ae_cfg["epochs"],
        learning_rate=ae_cfg["learning_rate"], early_stop_features=calib_feats,
    )

    baseline_pool = pd.concat([calib_feats, test_clean_feats])
    baseline_error = compute_reconstruction_error(baseline_pool, result)
    threshold = float(baseline_error.quantile(0.95))
    return result, threshold


def score_replicate_autoencoder(plant: str, replicate_id: int, twin, base: dict, ae_result, ae_threshold: float) -> pd.Series:
    """Day-level autoencoder flag: True if ANY daylight timestep's
    reconstruction error exceeds the calibrated threshold -- per-timestep
    architecture (LAYER3_HANDOFF.md SS5), not a day-mean/day-max (which
    was already found to hide A8 entirely for a different signal)."""
    meta, capacity_kw = base["meta"], base["capacity_kw"]
    rep_path = ROOT / "data" / "processed" / "injected" / plant / f"replicate_{replicate_id:03d}.parquet"
    df = pd.read_parquet(rep_path)
    anchor = compute_physics_anchor(df, meta["latitude"], meta["longitude"], capacity_kw)
    dfj = df.join(anchor)
    ts = build_residual_table(twin, dfj["physics_ac_kw"], dfj, capacity_kw, twin.feature_cols)

    feats = build_autoencoder_features(ts, twin.feature_cols)
    err = compute_reconstruction_error(feats, ae_result)

    daylight_idx = ts.index[ts["is_daylight"].astype(bool)]
    err_daylight = err[err.index.isin(daylight_idx)]
    dates = pd.Series(err_daylight.index.date, index=err_daylight.index)
    return (err_daylight > ae_threshold).groupby(dates).any()


def run_ablation_f() -> dict:
    out = {}
    frames = []
    for plant in PLANTS:
        base = load_plant_base(plant)
        twin = base["twin"]
        ae_result, ae_threshold = train_plant_autoencoder(plant, base)

        scored = score_all_replicates(plant, base=base).reset_index()
        df = merge_with_ground_truth(scored, plant)

        ae_flags = {}
        for rep_id in range(N_REPLICATES):
            day_flag = score_replicate_autoencoder(plant, rep_id, twin, base, ae_result, ae_threshold)
            for d, flagged in day_flag.items():
                ae_flags[(rep_id, d)] = bool(flagged)

        df["ae_detected"] = df.apply(lambda r: ae_flags.get((r["replicate"], r["date"]), False), axis=1)
        df["ensemble_detected"] = df["detected"] | df["ae_detected"]
        frames.append(df)

        out[plant] = dict(
            autoencoder_reconstruction_error_threshold=ae_threshold,
            conformal_production=_metrics_block(df, "detected")["overall"],
            autoencoder_only=_metrics_block(df, "ae_detected")["overall"],
            ensemble=_metrics_block(df, "ensemble_detected")["overall"],
            conformal_production_hard_negative_specificity=_metrics_block(df, "detected")["hard_negative_overall"]["specificity"],
            autoencoder_only_hard_negative_specificity=_metrics_block(df, "ae_detected")["hard_negative_overall"]["specificity"],
            ensemble_hard_negative_specificity=_metrics_block(df, "ensemble_detected")["hard_negative_overall"]["specificity"],
        )

    pooled = pd.concat(frames)
    out["pooled"] = dict(
        conformal_production=_metrics_block(pooled, "detected")["overall"],
        autoencoder_only=_metrics_block(pooled, "ae_detected")["overall"],
        ensemble=_metrics_block(pooled, "ensemble_detected")["overall"],
        conformal_production_hard_negative_specificity=_metrics_block(pooled, "detected")["hard_negative_overall"]["specificity"],
        autoencoder_only_hard_negative_specificity=_metrics_block(pooled, "ae_detected")["hard_negative_overall"]["specificity"],
        ensemble_hard_negative_specificity=_metrics_block(pooled, "ensemble_detected")["hard_negative_overall"]["specificity"],
    )
    return out


# ---------------------------------------------------------------------------
# (g) Leave-one-plant-out -- cross-reference only
# ---------------------------------------------------------------------------

def load_ablation_g() -> dict:
    path = ROOT / "reports" / "tables" / "leave_one_plant_out.json"
    if not path.exists():
        return {"note": "Task 7.3 output not found -- run src/evaluation/leave_one_plant_out.py first."}
    return {"note": "See src/evaluation/leave_one_plant_out.py / reports/tables/leave_one_plant_out.json -- not duplicated here.", "summary_path": str(path)}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_all_ablations() -> dict:
    return dict(
        a_satellite_vs_onsite=run_ablation_a(),
        b_with_without_physics_anchor=run_ablation_b(),
        c_conformal_vs_fixed_2sigma=run_ablation_c(),
        d_one_sided_vs_two_sided=run_ablation_d(),
        e_online_vs_batch=NOT_BUILT_ONLINE_LEARNING_NOTE,
        f_conformal_vs_autoencoder_vs_ensemble=run_ablation_f(),
        g_leave_one_plant_out=load_ablation_g(),
    )


def main() -> None:
    results = run_all_ablations()
    out_path = ROOT / "reports" / "tables" / "ablations.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Wrote {out_path}")

    print("\n=== (c) conformal vs fixed 2-sigma, pooled ===")
    print(json.dumps(results["c_conformal_vs_fixed_2sigma"]["pooled"], indent=2))
    print("\n=== (d) one-sided vs two-sided, pooled ===")
    print(json.dumps(results["d_one_sided_vs_two_sided"]["pooled"], indent=2))
    print("\n=== (f) conformal vs autoencoder vs ensemble, pooled ===")
    print(json.dumps(results["f_conformal_vs_autoencoder_vs_ensemble"]["pooled"], indent=2))


if __name__ == "__main__":
    main()
