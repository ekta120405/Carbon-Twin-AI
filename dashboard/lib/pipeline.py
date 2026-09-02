"""
dashboard/lib/pipeline.py

Layer 4.6 support module: wires Layers 2-4's real functions into cached,
dashboard-consumable context objects. Deliberately does NOT reimplement any
detection/explanation/carbon logic -- every computation here is a direct call
into src/layer{2,3,4}_*, in the exact sequence already verified by
score_with_fixes.py (project risk score) and verify_evidence_module.py
(evidence bundles). See docs/LAYER4_MODULES_1_TO_5_HANDOFF.md and
docs/LAYER3_HANDOFF.md before changing any wiring here -- the ordering
(e.g. conformal threshold fit once on the FULL real plant's calib days, then
reused across whichever data source is being viewed) is load-bearing, not
arbitrary.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
import yaml

from src.layer2_twin.twin_model import PIGBQRTwin
from src.layer2_twin.physics_anchor import compute_physics_anchor, compute_calibration_ratio
from src.layer3_anomaly.residual import (
    build_residual_table,
    compute_daily_energy_residual,
    compute_daily_night_injection_flag,
    compute_daily_stuck_sensor_flag,
)
from src.layer3_anomaly.conformal import fit_conformal_threshold, classify_days, tag_kt_regime
from src.layer3_anomaly.cusum import fit_cusum_reference, run_cusum, calibrate_h_from_clean_data
from src.layer3_anomaly.aggregate import (
    compute_daily_flag_severity_kwh,
    compute_divergence_severity_pct,
    calibrate_divergence_cap,
    compute_risk_score,
    RiskScoreInputs,
)
from src.layer4_xai.flag_shap import compute_day_level_evidence_features
from src.layer4_xai.carbon import build_carbon_table, aggregate_carbon_exposure, REQUIRED_CAVEATS
from src.layer4_xai.severity_ranking import rank_flags
from src.layer4_xai.evidence import build_evidence_bundles_for_plant
from src.layer3_anomaly.autoencoder import load_autoencoder, compute_daily_ae_flag
from src.layer3_anomaly.escalation import apply_secondary_evidence_escalation

with open(ROOT / "config" / "config.yaml") as _f:
    CONFIG = yaml.safe_load(_f)

ALPHA_NOMINAL = CONFIG["layer3"]["conformal"]["alpha"]
CSI_BINS = CONFIG["layer3"]["conformal"]["csi_bins"]
CUSUM_K = CONFIG["layer3"]["cusum"]["k"]
GRID_EMISSION_FACTOR = CONFIG["layer4"]["carbon"]["grid_emission_factor_tco2_per_mwh"]
CARBON_PRICE = CONFIG["layer4"]["carbon"]["carbon_price_usd_per_tco2"]
MAX_FLAGS_EXPLAINED = CONFIG["layer4"]["shap"]["max_flags_explained"]
ESCALATION_CFG = CONFIG["layer3"]["escalation"]

PLANTS = ["plant_1", "plant_2"]
N_REPLICATES = 30

FLAG_COLORS = {
    "flagged_overclaim": "#d62728",
    "flagged_night_injection": "#ff7f0e",
    "flagged_stuck_sensor": "#9467bd",
    "fault_underclaim": "#1f77b4",
    "normal": "#7f7f7f",
}

FLAG_LABELS = {
    "flagged_overclaim": "Flagged - overclaim (statistical)",
    "flagged_night_injection": "Flagged - night injection (guardrail)",
    "flagged_stuck_sensor": "Flagged - stuck sensor (guardrail)",
    "fault_underclaim": "Fault - underclaim (not fraud)",
    "normal": "Normal",
}

BAND_COLORS = {"Low": "#2ca02c", "Elevated": "#bcbd22", "High": "#ff7f0e", "Critical": "#d62728"}


def source_options(plant: str) -> list[tuple[str, str]]:
    """(source_key, human label) pairs for the sidebar source selector."""
    opts = [
        ("real_full", "Real data -- full history (34 days)"),
        ("real_clean", "Real data -- clean block (calib + test_clean, 10 days)"),
    ]
    replicate_dir = ROOT / "data" / "processed" / "injected" / plant
    n = len(sorted(replicate_dir.glob("replicate_*.parquet"))) if replicate_dir.exists() else 0
    for rep_id in range(n):
        opts.append((f"injected_{rep_id:03d}", f"Injected replicate {rep_id:03d}"))
    return opts


def _dates_index(df: pd.DataFrame) -> pd.Series:
    return pd.Series(df.index.date, index=df.index)


@st.cache_resource(show_spinner="Loading twin model and calibrating detectors...")
def load_plant_base(plant: str) -> dict:
    """
    Everything that is fixed PER PLANT, independent of which data source
    (real vs. which injected replicate) is currently being viewed: the
    loaded twin, the conformal threshold (calibrated ONLY on real calib
    days -- SS4 LAYER3_HANDOFF.md), the CUSUM reference stats (pooled real
    calib+test_clean -- cusum.py module docstring), the plant's own clean
    calibration-ratio baseline, and the divergence normalization cap.
    None of these may be recomputed per-source -- they are calibrated
    quantities and must stay fixed while different data is scored against
    them, exactly like score_with_fixes.py.
    """
    twin = PIGBQRTwin.load(str(ROOT / "reports" / "models" / plant))
    with open(ROOT / "data" / "processed" / f"{plant}_meta.yaml") as f:
        capacity_kw = yaml.safe_load(f)["ac_nameplate_kw"]
    with open(ROOT / "config" / "plants.yaml") as f:
        meta = yaml.safe_load(f)[plant]

    df_main = pd.read_parquet(ROOT / "data" / "processed" / f"{plant}.parquet")
    anchor_main = compute_physics_anchor(df_main, meta["latitude"], meta["longitude"], capacity_kw)
    df_main_j = df_main.join(anchor_main)
    ts_main = build_residual_table(twin, df_main_j["physics_ac_kw"], df_main_j, capacity_kw, twin.feature_cols)
    daily_main = compute_daily_energy_residual(ts_main, ac_nameplate_kw=capacity_kw)

    daylight_main = ts_main[ts_main["is_daylight"]]
    split_main = daylight_main.groupby(daylight_main.index.date)["split"].agg(lambda s: s.mode().iloc[0])
    calib_dates = split_main[split_main == "calib"].index
    test_clean_dates = split_main[split_main == "test_clean"].index
    calib_test_clean_dates = calib_dates.union(test_clean_dates)

    conformal_result = fit_conformal_threshold(
        daily_main.loc[daily_main.index.isin(calib_dates), "r_day"], alpha_nominal=ALPHA_NOMINAL
    )
    clean_cr_baseline = compute_calibration_ratio(
        df_main_j["ac_power"], df_main_j["physics_ac_kw"], df_main_j["twin_kt"]
    )
    divergence_cap = calibrate_divergence_cap(ts_main, calib_test_clean_dates, percentile=0.95)

    reference_pool = pd.concat([
        daily_main.loc[daily_main.index.isin(calib_dates), "r_day"],
        daily_main.loc[daily_main.index.isin(test_clean_dates), "r_day"],
    ])
    cusum_ref = fit_cusum_reference(reference_pool)

    evidence_main = compute_day_level_evidence_features(ts_main, daily_main, capacity_kw)
    clean_reference_pool = evidence_main[evidence_main.index.isin(calib_test_clean_dates)]

    # Secondary-evidence escalation's autoencoder -- persisted by
    # src/layer3_anomaly/train_autoencoder_model.py. Loaded once per plant
    # here (this function is st.cache_resource-cached) rather than
    # retrained on every call, per docs/TASK7_HANDOFF.md's finding that no
    # reusable trained model existed before this. None if the model
    # hasn't been trained yet -- callers must treat that as "escalation
    # unavailable," never silently skip the whole scoring pipeline.
    ae_model_dir = ROOT / "reports" / "models" / plant
    autoencoder = None
    if (ae_model_dir / "autoencoder.pt").exists():
        ae_result, ae_threshold, ae_run_length = load_autoencoder(str(ae_model_dir))
        autoencoder = dict(result=ae_result, threshold=ae_threshold, run_length=ae_run_length)

    return dict(
        twin=twin,
        capacity_kw=capacity_kw,
        meta=meta,
        ts_main=ts_main,
        daily_main=daily_main,
        conformal_result=conformal_result,
        clean_cr_baseline=clean_cr_baseline,
        divergence_cap=divergence_cap,
        reference_pool=reference_pool,
        cusum_ref=cusum_ref,
        clean_reference_pool=clean_reference_pool,
        calib_dates=calib_dates,
        test_clean_dates=test_clean_dates,
        calib_test_clean_dates=calib_test_clean_dates,
        autoencoder=autoencoder,
    )


def _load_source_df(plant: str, source_key: str, base: dict) -> pd.DataFrame:
    if source_key == "real_full":
        return pd.read_parquet(ROOT / "data" / "processed" / f"{plant}.parquet")
    if source_key == "real_clean":
        df_full = pd.read_parquet(ROOT / "data" / "processed" / f"{plant}.parquet")
        mask = _dates_index(df_full).isin(base["calib_test_clean_dates"])
        return df_full[mask]
    if source_key.startswith("injected_"):
        rep_id = int(source_key.split("_")[1])
        return pd.read_parquet(
            ROOT / "data" / "processed" / "injected" / plant / f"replicate_{rep_id:03d}.parquet"
        )
    raise ValueError(f"Unknown source_key: {source_key!r}")


@st.cache_resource(show_spinner="Scoring selected data source...")
def get_scored_context(plant: str, source_key: str) -> dict:
    """
    Runs Layers 2-4 on ONE selected data source (real full history, real
    clean block, or one injected replicate), reusing the PER-PLANT
    calibrated quantities from load_plant_base -- never refitting the
    conformal threshold, CUSUM reference, or calibration-ratio baseline
    against the source being viewed itself (that would be evaluating the
    detector against its own calibration, not a real held-out test).
    CUSUM's h is the one exception recalibrated here, by design: h is only
    valid for the specific window length it was calibrated for
    (docs/LAYER3_HANDOFF.md SS7 / cusum.py module docstring), so it is
    freshly calibrated for whatever window length this source actually is.
    """
    base = load_plant_base(plant)
    twin = base["twin"]
    capacity_kw = base["capacity_kw"]
    meta = base["meta"]

    df = _load_source_df(plant, source_key, base)
    anchor = compute_physics_anchor(df, meta["latitude"], meta["longitude"], capacity_kw)
    df = df.join(anchor)
    ts_table = build_residual_table(twin, df["physics_ac_kw"], df, capacity_kw, twin.feature_cols)
    daily = compute_daily_energy_residual(ts_table, ac_nameplate_kw=capacity_kw)

    night_flags = compute_daily_night_injection_flag(ts_table, capacity_kw)
    stuck_flags = compute_daily_stuck_sensor_flag(ts_table)
    labels = classify_days(daily["r_day"], base["conformal_result"], night_flags, stuck_flags)

    daylight_kt = ts_table[ts_table["is_daylight"].astype(bool)]
    daily_mean_kt = daylight_kt.groupby(_dates_index(daylight_kt))["twin_kt"].mean()
    daily["kt_regime"] = tag_kt_regime(daily_mean_kt.reindex(daily.index), CSI_BINS)

    window_len = max(len(daily), 2)
    h_calib = calibrate_h_from_clean_data(
        base["reference_pool"], k=CUSUM_K, window_len=window_len, target_fpr=0.05
    )
    h = h_calib["recommended_h"]
    cusum_res = run_cusum(daily.sort_index()["r_day"], base["cusum_ref"], k=CUSUM_K, h=h)
    max_s = max(cusum_res.s_values) if cusum_res.s_values else 0.0

    # Secondary-evidence escalation (src/layer3_anomaly/escalation.py) --
    # upgrades a 'normal' day to 'flagged_overclaim' when it's a
    # near-miss on the primary test AND a secondary signal (autoencoder
    # run-length flag, or CUSUM's own statistic) corroborates. MUST run
    # before severity/frac_flagged/carbon/ranking are computed below, so
    # every downstream number (risk score, carbon exposure, evidence
    # bundles) reflects the escalated flags, not the pre-escalation ones.
    # See config.yaml's layer3.escalation section for the calibration
    # story (honest split-validated, not a free/zero-cost result).
    if ESCALATION_CFG.get("enabled") and base.get("autoencoder") is not None:
        ae = base["autoencoder"]
        ae_day_flag = compute_daily_ae_flag(
            ts_table, twin.feature_cols, ae["result"], ae["threshold"], ae["run_length"],
        )
        cusum_s_t_series = pd.Series(cusum_res.s_values, index=pd.Index(cusum_res.dates))
        labels = apply_secondary_evidence_escalation(
            daily, labels, base["conformal_result"].threshold,
            cusum_s_t_series, h, ae_day_flag,
            band_width=ESCALATION_CFG["band_width"], cusum_frac=ESCALATION_CFG["cusum_frac"],
        )

    severity = compute_daily_flag_severity_kwh(daily, labels, ts_table)
    flagged_mask = labels.str.startswith("flagged")
    frac_flagged = float(flagged_mask.mean()) if len(labels) else 0.0
    total_claimed = float(daily["claimed_energy_kwh"].sum())
    cum_severity_frac = (
        float(severity[flagged_mask].sum() / total_claimed) if total_claimed > 0 else 0.0
    )

    warnings = []
    try:
        cr = compute_calibration_ratio(df["ac_power"], df["physics_ac_kw"], df["twin_kt"])
    except ValueError as e:
        cr = base["clean_cr_baseline"]
        warnings.append(f"Calibration ratio could not be computed for this source ({e}); "
                         f"falling back to the plant's clean baseline for display.")

    divergence = compute_divergence_severity_pct(ts_table, labels)

    risk_inputs = RiskScoreInputs(
        frac_flagged=frac_flagged,
        cum_severity_frac=cum_severity_frac,
        cusum_max_s=max_s,
        cusum_h=h,
        calibration_ratio=cr,
        calibration_ratio_clean_baseline=base["clean_cr_baseline"],
        divergence_flagged_only=divergence,
    )
    risk_result = compute_risk_score(risk_inputs, divergence_cap=base["divergence_cap"])

    carbon_table = build_carbon_table(daily, labels, ts_table, GRID_EMISSION_FACTOR, CARBON_PRICE)
    carbon_agg = aggregate_carbon_exposure(carbon_table)

    ranked = rank_flags(
        daily, labels, ts_table, base["conformal_result"], capacity_kw,
        GRID_EMISSION_FACTOR, CARBON_PRICE,
    )

    evidence_bundles = build_evidence_bundles_for_plant(
        plant, twin, ts_table, daily, labels, base["clean_reference_pool"], capacity_kw,
        GRID_EMISSION_FACTOR, CARBON_PRICE, max_flags_explained=MAX_FLAGS_EXPLAINED,
    )
    bundles_by_date = {b["date"]: b for b in evidence_bundles}

    return dict(
        df=df,
        ts_table=ts_table,
        daily=daily,
        labels=labels,
        severity=severity,
        frac_flagged=frac_flagged,
        cum_severity_frac=cum_severity_frac,
        cusum_result=cusum_res,
        cusum_h=h,
        calibration_ratio=cr,
        divergence=divergence,
        risk_result=risk_result,
        carbon_table=carbon_table,
        carbon_agg=carbon_agg,
        ranked=ranked,
        evidence_bundles=evidence_bundles,
        bundles_by_date=bundles_by_date,
        warnings=warnings,
    )


@st.cache_data(show_spinner=False)
def load_ground_truth(plant: str) -> pd.DataFrame:
    """Frozen injector ground truth (data/ground_truth/) -- read-only, bounded
    sanity-check use only (docs/LAYER4_ROADMAP.md SS2.8), never for tuning."""
    path = ROOT / "data" / "ground_truth" / f"{plant}_labels.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
