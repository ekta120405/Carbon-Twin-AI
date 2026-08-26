"""
src/layer4_xai/evidence.py

Layer 4.4: Evidence bundle (Overview v2 Module 4.4; docs/LAYER4_ROADMAP.md
SS4.4) -- combines 4.1 (Twin-SHAP) + 4.2 (Flag-SHAP) + 4.3 (carbon) into
one exportable, auditor-facing record per flagged day. No new modeling --
purely an assembly/serialization step, same nature as 4.3.

ONE CARRYOVER FROM 4.1 THAT MUST NOT GET LOST HERE: for
flagged_night_injection days, the physical_explanation panel describes the
day's peak-residual DAYLIGHT timestamp (residual.py's r_day is structurally
blind to night-time claims -- see twin_shap.py's
pick_representative_timestamp() docstring), which is NOT the timestamp that
actually triggered the flag. Left unlabeled, a reader would reasonably
assume the physical_explanation panel explains the flagged event itself.
This module adds an explicit known_caveats entry for exactly this case so
the mismatch is visible on the bundle, not just in a code comment two
modules upstream.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.layer2_twin.twin_model import PIGBQRTwin
from src.layer3_anomaly.aggregate import compute_daily_flag_severity_kwh
from src.layer4_xai.twin_shap import TwinShapExplainer, pick_representative_timestamp, DEFAULT_KT_SWITCH
from src.layer4_xai.flag_shap import (
    compute_day_level_evidence_features,
    explain_flagged_day,
    STUCK_SENSOR_CAVEAT,
)
from src.layer4_xai.carbon import compute_carbon_exposure, REQUIRED_CAVEATS

NIGHT_INJECTION_TWIN_SHAP_CAVEAT = (
    "The physical_explanation panel below describes conditions at this "
    "day's largest-residual DAYLIGHT timestamp, not the flagged night-time "
    "event -- residual.py's daily residual is structurally computed from "
    "daylight rows only (see docs/LAYER3_HANDOFF.md night-injection "
    "discussion), so no daylight timestamp can represent the actual "
    "violation. The actual flagged event is described in "
    "suspicion_explanation below, which correctly points at the specific "
    "night-time timestamp and claimed power that triggered this flag."
)


def _physical_explanation_dict(expl) -> dict:
    d = {"mode": expl.mode, "twin_kt": expl.twin_kt, "note": expl.note}
    if expl.mode == "shap":
        d["physics_ac_kw"] = expl.physics_ac_kw
        d["twin_correction_kw"] = expl.twin_correction_kw
        d["expected_q50_kw"] = expl.expected_q50_kw
        d["top_features_kw"] = expl.top_features_kw
    else:
        d["physics_ac_kw"] = expl.physics_ac_kw
        d["expected_q50_kw"] = expl.expected_q50_kw
    return d


def _suspicion_explanation_dict(expl) -> dict:
    if hasattr(expl, "top_drivers"):  # StatisticalFlagExplanation (Path A)
        return {
            "mode": "statistical",
            "top_drivers": expl.top_drivers,
            "raw_values": expl.raw_values,
            "reference_n": expl.reference_n,
        }
    else:  # GuardrailFlagExplanation (Path B)
        return {
            "mode": "guardrail",
            "text": expl.text,
            "trigger_values": expl.trigger_values,
        }


def build_evidence_bundle(
    plant: str,
    date,
    flag_type: str,
    physical_expl,
    suspicion_expl,
    severity_kwh: float,
    emission_factor_tco2_per_mwh: float,
    carbon_price_usd_per_tco2: float,
) -> dict:
    """
    Assembles one flagged day's full record. `physical_expl` is a
    twin_shap.TimestampExplanation, `suspicion_expl` is either a
    flag_shap.StatisticalFlagExplanation or .GuardrailFlagExplanation
    (dispatched automatically by _suspicion_explanation_dict).
    """
    carbon = compute_carbon_exposure(severity_kwh, emission_factor_tco2_per_mwh, carbon_price_usd_per_tco2)

    known_caveats = list(REQUIRED_CAVEATS)
    if flag_type == "flagged_stuck_sensor":
        known_caveats.append(STUCK_SENSOR_CAVEAT)
    if flag_type == "flagged_night_injection":
        known_caveats.append(NIGHT_INJECTION_TWIN_SHAP_CAVEAT)

    return {
        "plant": plant,
        "date": str(date),
        "flag_type": flag_type,
        "severity_kwh": severity_kwh,
        "physical_explanation": _physical_explanation_dict(physical_expl),
        "suspicion_explanation": _suspicion_explanation_dict(suspicion_expl),
        "carbon_exposure": {
            "excess_tco2e": carbon.excess_tco2e,
            "value_at_risk_usd": carbon.value_at_risk_usd,
        },
        "known_caveats": known_caveats,
    }


def build_evidence_bundles_for_plant(
    plant: str,
    twin: PIGBQRTwin,
    ts_table: pd.DataFrame,
    daily: pd.DataFrame,
    labels: pd.Series,
    clean_reference_pool: pd.DataFrame,
    ac_nameplate_kw: float,
    emission_factor_tco2_per_mwh: float,
    carbon_price_usd_per_tco2: float,
    kt_switch: float = DEFAULT_KT_SWITCH,
    max_flags_explained: int = 500,
    dt_hours: float = 0.25,
) -> list[dict]:
    """
    Full 4.1+4.2+4.3 assembly for every flagged day in `labels`
    (fault_underclaim/normal excluded -- not fraud flags, see
    flag_shap.explain_flagged_day's ValueError for the same rule).
    """
    explainer = TwinShapExplainer(twin, kt_switch=kt_switch)
    evidence_table = compute_day_level_evidence_features(ts_table, daily, ac_nameplate_kw, dt_hours)
    severity_kwh_series = compute_daily_flag_severity_kwh(daily, labels, ts_table, dt_hours)

    ts_dates = pd.Series(ts_table.index.date, index=ts_table.index)
    flagged_dates = labels[labels.str.startswith("flagged")].index[:max_flags_explained]

    bundles = []
    for date in flagged_dates:
        flag_type = labels.loc[date]
        day_table = ts_table[ts_dates == date]
        if len(day_table) == 0:
            continue

        rep_ts = pick_representative_timestamp(day_table)
        row = day_table.loc[rep_ts]
        physical_expl = explainer.explain_timestamp(row, physics_ac_kw=row["physics_ac_kw"], twin_kt=row["twin_kt"])

        suspicion_expl = explain_flagged_day(
            date, flag_type, ts_table, evidence_table=evidence_table,
            clean_reference_pool=clean_reference_pool, ac_nameplate_kw=ac_nameplate_kw,
        )

        severity_kwh = float(severity_kwh_series.get(date, 0.0))
        bundle = build_evidence_bundle(
            plant, date, flag_type, physical_expl, suspicion_expl, severity_kwh,
            emission_factor_tco2_per_mwh, carbon_price_usd_per_tco2,
        )
        bundles.append(bundle)
    return bundles


def save_evidence_bundles(bundles: list[dict], out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(bundles, f, indent=2, default=str)
