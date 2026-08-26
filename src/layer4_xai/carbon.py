"""
src/layer4_xai/carbon.py

Layer 4.3: carbon accounting (Overview v2 Module 4.3; docs/LAYER4_ROADMAP.md
SS4.3).

WHAT THIS MODULE IS, PRECISELY: a pure unit conversion on Layer 3's
already-computed, already flag-type-aware excess energy
(src/layer3_anomaly/aggregate.py:compute_daily_flag_severity_kwh). Carbon
credits for a grid-connected solar project are arithmetically downstream of
verified kWh -- one credit ~= one tonne CO2e avoided, computed as
(kWh / 1000) * grid_emission_factor. There is no independent "carbon fraud"
mechanism: if a plant over-claims kWh by X%, it over-claims carbon credits
by exactly X%, because the conversion is a fixed multiplier. Consequently
there is nothing non-deterministic here for SHAP or any other attribution
method to explain (see roadmap SS2.4 and the Overview v2 SHAP discussion) --
this module is intentionally simple arithmetic, not a modeling step.

DO NOT recompute excess energy here. This is the single most important
carryover from Layers 2/3 (roadmap SS2.4): a day-wide or naively-averaged
excess-energy number reintroduces the exact silent-dilution bug found and
fixed three separate times in Layer 3 (see docs/LAYER3_HANDOFF.md SS0, SS7)
-- most concretely, using residual.py's raw daylight-only
`excess_energy_kwh` for a flagged_night_injection day would sum unrelated
ordinary daytime variation instead of the actual fabricated night energy
(inflated ~500 kWh -> ~173,000 kWh in the real case that motivated the fix).
This module always calls compute_daily_flag_severity_kwh() and treats its
output as the sole source of excess-energy magnitude.

TERMINOLOGY: outputs are "tCO2e AT RISK" / "value AT RISK", not "credits
over-issued" or "fraud confirmed" -- the flagged excess is Layer 3's
statistical suspicion (achievable alpha ~16.7% at the conformal layer,
see docs/LAYER3_HANDOFF.md SS4; known false-positive overlap for the A3/A8
guardrails, see SS3 and roadmap SS2.6), not an adjudicated fact. Report
language built on this module's output must preserve that distinction.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.layer3_anomaly.aggregate import compute_daily_flag_severity_kwh

# ---------------------------------------------------------------------------
# Caveats that MUST accompany any report/paper text using these numbers
# (roadmap SS2.7) -- kept here as text, not just a config comment, so any
# consumer of this module (evidence bundles, dashboard, paper-generation
# scripts) can pull the exact required wording rather than re-deriving it
# and risking drift.
# ---------------------------------------------------------------------------

VINTAGE_MISMATCH_CAVEAT = (
    "Vintage mismatch: the grid emission factor is CEA CO2 Baseline Database "
    "v21.0 (Nov 2025, Combined Margin, FY2024-25); the dataset's actual "
    "generation is from 2020. This is a stated methodological "
    "simplification -- demonstrating the verification approach, not issuing "
    "a real registered credit -- not an oversight."
)

PRICE_UNCERTAINTY_CAVEAT = (
    "Price uncertainty: the voluntary carbon market is opaque and "
    "unorganized, with wide price dispersion ($0.25-$27/tCO2e observed "
    "range for this credit category). The configured carbon price is a "
    "representative point estimate at the low end of the $4-6/tCO2e range "
    "typical for grid-connected renewable-energy avoidance credits "
    "specifically (reflecting the documented EMDE pricing discount, "
    "~28% below advanced-economy credits, applicable to an India-origin "
    "project) -- not a quoted market rate."
)

REQUIRED_CAVEATS = (VINTAGE_MISMATCH_CAVEAT, PRICE_UNCERTAINTY_CAVEAT)


@dataclass
class CarbonExposure:
    excess_kwh: float
    excess_tco2e: float
    value_at_risk_usd: float
    emission_factor_tco2_per_mwh: float
    carbon_price_usd_per_tco2: float


def compute_carbon_exposure(
    excess_kwh: float,
    emission_factor_tco2_per_mwh: float,
    carbon_price_usd_per_tco2: float,
) -> CarbonExposure:
    """
    Pure arithmetic: excess_kwh -> tCO2e -> $ value at risk.

    `excess_kwh` MUST be a flag-type-aware severity value already produced
    by compute_daily_flag_severity_kwh() (or 0.0 for an unflagged day) --
    never a raw/naive excess-energy number. This function does not know or
    care which flag type produced the value; that decision already happened
    upstream in Layer 3, per the flag-type-aware severity design (see module
    docstring and docs/LAYER3_HANDOFF.md SS7).
    """
    if excess_kwh < 0:
        raise ValueError(
            f"excess_kwh must be >= 0 (a severity magnitude, not a signed "
            f"residual); got {excess_kwh}. If this fires, check the caller "
            f"isn't passing raw r_day/excess_energy_kwh instead of "
            f"compute_daily_flag_severity_kwh()'s output."
        )
    excess_tco2e = (excess_kwh / 1000.0) * emission_factor_tco2_per_mwh
    value_at_risk_usd = excess_tco2e * carbon_price_usd_per_tco2
    return CarbonExposure(
        excess_kwh=excess_kwh,
        excess_tco2e=excess_tco2e,
        value_at_risk_usd=value_at_risk_usd,
        emission_factor_tco2_per_mwh=emission_factor_tco2_per_mwh,
        carbon_price_usd_per_tco2=carbon_price_usd_per_tco2,
    )


def build_carbon_table(
    daily: pd.DataFrame,
    labels: pd.Series,
    ts_table: pd.DataFrame,
    emission_factor_tco2_per_mwh: float,
    carbon_price_usd_per_tco2: float,
    dt_hours: float = 0.25,
) -> pd.DataFrame:
    """
    Day-level carbon exposure table for one plant/period.

    `daily`, `labels`, `ts_table` are exactly the objects Layer 3 already
    produces: `daily` from residual.compute_daily_energy_residual(),
    `labels` from conformal.classify_days() (sharing `daily`'s date index),
    `ts_table` from residual.build_residual_table(). This function calls
    compute_daily_flag_severity_kwh() itself -- callers should NOT pass in
    a pre-computed severity series from anywhere else, to guarantee the
    flag-type-aware definition is always what's used here.

    Returns one row per date in `daily`, columns:
        flag_type, severity_kwh, excess_tco2e, value_at_risk_usd
    fault_underclaim/normal days carry 0.0 exposure by construction of
    compute_daily_flag_severity_kwh (this is intentional, not a filter
    applied here -- no fraud severity to report for those flag types).
    """
    severity_kwh = compute_daily_flag_severity_kwh(daily, labels, ts_table, dt_hours)

    rows = []
    for date in daily.index:
        flag_type = labels.get(date, "normal")
        sev = float(severity_kwh.get(date, 0.0))
        exp = compute_carbon_exposure(sev, emission_factor_tco2_per_mwh, carbon_price_usd_per_tco2)
        rows.append(
            {
                "date": date,
                "flag_type": flag_type,
                "severity_kwh": exp.excess_kwh,
                "excess_tco2e": exp.excess_tco2e,
                "value_at_risk_usd": exp.value_at_risk_usd,
            }
        )
    out = pd.DataFrame(rows).set_index("date")
    out.index.name = daily.index.name
    return out


def aggregate_carbon_exposure(carbon_table: pd.DataFrame) -> dict:
    """
    Project/period-level totals, flagged days only.

    Non-flagged rows are already 0.0 by construction of
    compute_daily_flag_severity_kwh, so summing the whole table vs. summing
    only flagged rows gives the identical total -- the explicit filter here
    exists to keep `n_flagged_days` and `by_flag_type` meaningful and to
    stay correct if that upstream 0.0 guarantee ever changes, not because
    it changes the totals today.
    """
    flagged = carbon_table[carbon_table["flag_type"].str.startswith("flagged")]
    by_type = (
        flagged.groupby("flag_type")[["excess_tco2e", "value_at_risk_usd"]]
        .sum()
        .to_dict("index")
        if len(flagged) > 0
        else {}
    )
    return {
        "n_flagged_days": int(len(flagged)),
        "total_excess_tco2e": float(flagged["excess_tco2e"].sum()) if len(flagged) else 0.0,
        "total_value_at_risk_usd": float(flagged["value_at_risk_usd"].sum()) if len(flagged) else 0.0,
        "by_flag_type": by_type,
    }
