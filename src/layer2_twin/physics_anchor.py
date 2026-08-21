"""
src/layer2_twin/physics_anchor.py

Layer 2.1: the physics anchor (Overview v2 SS5.1).

A deterministic, non-learned estimate of expected AC output, computed purely
from independent weather + basic system assumptions -- zero training data
required. Serves three purposes:
  1. Defines the physics ceiling used in Layer 1.4 QC (already implemented,
     simplified form, in quality.py).
  2. Provides the TARGET BASELINE for the PI-GBQR twin: the twin only learns
     the residual correction on top of this, not the whole physical
     relationship from scratch (this is what makes ~900 training rows enough).
  3. Provides the calibration-ratio diagnostic (SS5.4): claimed/physics_anchor
     on clear-sky timestamps -- a plant that can only be fitted by assuming
     >100% system efficiency is itself evidence of systematic inflation.

Uses pvlib's full chain: solar position -> GHI decomposition (Erbs) ->
plane-of-array transposition -> Faiman cell temperature -> PVWatts DC -> AC.

Real tilt/azimuth are unknown (Task 4 limitation -- no engineering spec
available for either plant). Tilt defaults to latitude (a standard rule-of-
thumb for fixed-tilt ground-mount arrays); azimuth defaults to 180 deg
(due south, correct for northern-hemisphere fixed installations). Document
this assumption in the paper alongside the coordinate-confidence asymmetry
from Task 4.
"""
from __future__ import annotations

import pandas as pd


def compute_physics_anchor(
    df: pd.DataFrame,
    latitude: float,
    longitude: float,
    ac_nameplate_kw: float,
    tilt_deg: float | None = None,
    azimuth_deg: float = 180.0,
    gamma_pdc: float = -0.004,
    dc_ac_ratio: float = 1.10,
    system_losses: float = 0.14,
    eta_inv_nom: float = 0.96,
    ghi_col: str = "twin_ghi",
) -> pd.DataFrame:
    """
    Returns a DataFrame indexed like `df`, columns:
        physics_poa_global, physics_cell_temp, physics_dc_kw, physics_ac_kw

    `physics_ac_kw` is P_physics(t) -- the value the PI-GBQR twin's target
    is defined relative to (Overview v2 SS5.1).
    """
    import pvlib

    tilt = tilt_deg if tilt_deg is not None else abs(latitude)

    # 1. Solar position
    solpos = pvlib.solarposition.get_solarposition(df.index, latitude, longitude)

    # 2. GHI -> DNI/DHI decomposition (Erbs model -- needs only GHI + zenith,
    #    appropriate since NASA POWER's DNI/DHI are themselves modeled
    #    quantities at 1-degree resolution, not meaningfully more "real"
    #    than a decomposition from GHI alone at this spatial scale).
    erbs = pvlib.irradiance.erbs(df[ghi_col], solpos["zenith"], df.index)

    # 3. Transpose to plane-of-array
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth_deg,
        solar_zenith=solpos["apparent_zenith"],
        solar_azimuth=solpos["azimuth"],
        dni=erbs["dni"],
        ghi=df[ghi_col],
        dhi=erbs["dhi"],
        model="isotropic",
    )

    # 4. Cell temperature (Faiman -- needs only POA, ambient temp, wind speed)
    cell_temp = pvlib.temperature.faiman(
        poa_global=poa["poa_global"], temp_air=df["twin_t2m"], wind_speed=df["twin_ws2m"]
    )

    # 5. PVWatts DC
    pdc0_array = ac_nameplate_kw * dc_ac_ratio  # array DC nameplate, oversized per config
    dc_kw = pvlib.pvsystem.pvwatts_dc(
        g_poa_effective=poa["poa_global"], temp_cell=cell_temp, pdc0=pdc0_array, gamma_pdc=gamma_pdc
    )

    # 6. PVWatts AC (inverter clipping + efficiency curve)
    inverter_pdc0 = ac_nameplate_kw / eta_inv_nom  # DC input at which inverter hits nominal efficiency
    ac_kw = pvlib.inverter.pvwatts(pdc=dc_kw, pdc0=inverter_pdc0, eta_inv_nom=eta_inv_nom)

    # 7. System losses (soiling, wiring, availability) as a single derate
    #    factor on final AC output -- a simplification of PVWatts' more
    #    granular loss breakdown, appropriate for an ANCHOR model.
    ac_kw_final = (ac_kw * (1 - system_losses)).clip(lower=0)

    return pd.DataFrame({
        "physics_poa_global": poa["poa_global"],
        "physics_cell_temp": cell_temp,
        "physics_dc_kw": dc_kw,
        "physics_ac_kw": ac_kw_final,
    }, index=df.index)


def compute_calibration_ratio(
    claimed_kw: pd.Series, physics_ac_kw: pd.Series, clearsky_index: pd.Series,
    min_kt: float = 0.90,
) -> float:
    """
    Layer 2.4: CR = median(claimed) / median(physics_anchor) on clear-sky
    timestamps. CR meaningfully > 1 across an extended period is itself
    evidence of systematic over-claiming, independent of any per-timestamp test.
    """
    mask = clearsky_index >= min_kt
    aligned = pd.DataFrame({"claimed": claimed_kw, "physics": physics_ac_kw}).loc[mask].dropna()
    aligned = aligned[aligned["physics"] > 1e-6]
    if len(aligned) < 20:
        raise ValueError(f"Only {len(aligned)} clear-sky samples -- too few for a stable CR estimate")
    return float(aligned["claimed"].median() / aligned["physics"].median())
