"""
src/layer1_data/features.py

Layer 1.5: feature engineering (Overview v2 SS4.6).

CRITICAL INVARIANT: twin features and evidence features are built by
SEPARATE functions returning SEPARATE, prefixed column sets. This is what
Test TC-L2-01 (Task 7) checks mechanically -- it is not just a convention,
the twin's training code asserts on the `twin_` prefix and will refuse to
train if any `evidence_` or raw claimed-generation column is present.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_twin_features(df: pd.DataFrame, latitude: float, longitude: float) -> pd.DataFrame:
    """
    Independent-weather-only features. NEVER include claimed generation,
    on-site sensor readings, or anything derived from them.

    Expects df to already contain (from downscale.py / NASA POWER):
        ghi_15min, clearsky_ghi_15min, kt_15min, t2m, rh2m, ws2m, ps, prectotcorr
    """
    import pvlib

    solpos = pvlib.solarposition.get_solarposition(df.index, latitude, longitude)
    airmass = pvlib.atmosphere.get_relative_airmass(solpos["apparent_zenith"])

    out = pd.DataFrame(index=df.index)
    out["twin_ghi"] = df["ghi_15min"]
    out["twin_clearsky_ghi"] = df["clearsky_ghi_15min"]
    out["twin_kt"] = df["kt_15min"]
    out["twin_t2m"] = df["t2m"]
    out["twin_rh2m"] = df["rh2m"]
    out["twin_ws2m"] = df["ws2m"]
    out["twin_ps"] = df["ps"]
    out["twin_prectotcorr"] = df["prectotcorr"]

    out["twin_solar_zenith"] = solpos["apparent_zenith"]
    out["twin_solar_elevation"] = 90.0 - solpos["apparent_zenith"]
    out["twin_solar_azimuth"] = solpos["azimuth"]
    out["twin_airmass"] = airmass.clip(upper=10).fillna(10)

    # cell temperature estimate (same simple NOCT model as quality.py, kept
    # consistent so the twin's temperature feature matches the QC ceiling's assumption)
    out["twin_cell_temp"] = df["t2m"] + (45.0 - 20.0) / 800.0 * df["ghi_15min"].clip(lower=0)

    hour = out.index.hour + out.index.minute / 60
    out["twin_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["twin_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    doy = out.index.dayofyear
    out["twin_doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["twin_doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # short lags: thermal inertia means cell temp / GHI a few steps ago still
    # matters. These are lags of INDEPENDENT weather, not of claimed generation
    # -- still safe under the R1 invariant.
    out["twin_ghi_lag1"] = out["twin_ghi"].shift(1)
    out["twin_ghi_lag2"] = out["twin_ghi"].shift(2)
    out["twin_kt_lag1"] = out["twin_kt"].shift(1)

    return out


def build_evidence_features(df: pd.DataFrame, nasa_ghi_col: str = "ghi_15min",
                             ac_nameplate_kw: float = 1.0) -> pd.DataFrame:
    """
    Untrusted / evidence features -- used ONLY by Layer 4 (flag-SHAP) to
    explain WHY something is suspicious. Never passed to the twin.

    Expects df to contain: claimed on-site sensor columns
        (onsite_irradiation, onsite_ambient_temp, onsite_module_temp)
    plus the independent NASA columns for computing divergence.
    """
    out = pd.DataFrame(index=df.index)

    out["evidence_onsite_irradiation"] = df["onsite_irradiation"]
    out["evidence_onsite_ambient_temp"] = df["onsite_ambient_temp"]
    out["evidence_onsite_module_temp"] = df["onsite_module_temp"]

    # divergence: the fingerprint of on-site sensor tampering (weather spoofing, A5)
    # onsite_irradiation is a fraction (0-1) in the Kaggle schema; NASA GHI is W/m2.
    # Normalise onsite to W/m2 scale using the peak observed clear-sky value as a
    # rough reference, so the divergence is in comparable units.
    onsite_ghi_approx = df["onsite_irradiation"] * df[nasa_ghi_col].quantile(0.99)
    out["evidence_irr_divergence"] = onsite_ghi_approx - df[nasa_ghi_col]
    out["evidence_irr_divergence_pct"] = (
        out["evidence_irr_divergence"] / df[nasa_ghi_col].replace(0, np.nan)
    ).clip(-5, 5)

    out["evidence_temp_divergence"] = df["onsite_ambient_temp"] - df["t2m"]

    # day-over-day yield discontinuity: sudden jump vs. the plant's own recent trend
    daily_yield = df.get("daily_yield")
    if daily_yield is not None:
        out["evidence_yield_discontinuity"] = daily_yield.diff().fillna(0)

    return out
