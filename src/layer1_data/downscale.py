"""
src/layer1_data/downscale.py

Layer 1.3: clear-sky-index-preserving temporal downscaling (Overview v2 SS4.4).

NASA POWER hourly GHI cannot be naively forward-filled or linearly interpolated
to 15-minute resolution -- that destroys the diurnal shape (irradiance rises and
falls within each hour, especially near sunrise/sunset, which is exactly when
your daylight mask and physics ceiling are most sensitive).

Method:
  1. Compute clear-sky GHI at 15-min resolution using pvlib (exact solar geometry).
  2. Compute the hourly hourly clear-sky index  Kt = GHI_nasa_hourly / GHI_clearsky_hourly.
  3. Interpolate Kt (not GHI) to 15-min -- cloud opacity varies smoothly, solar
     geometry does not, so this is the right variable to interpolate.
  4. Reconstruct  GHI_15min = Kt_interpolated * GHI_clearsky_15min.

This preserves the exact diurnal curve shape while smoothly distributing the
hourly cloud signal across the sub-hourly steps.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_clearsky_ghi(times_utc: pd.DatetimeIndex, latitude: float, longitude: float,
                          altitude: float = 300.0) -> pd.Series:
    """
    Clear-sky GHI via pvlib's Ineichen model.

    Requires pvlib (see requirements.txt). This function is the ONLY place in
    Layer 1 that should import pvlib directly -- keeping the dependency
    localised makes it easy to swap models (Ineichen -> Haurwitz) later.
    """
    import pvlib

    site = pvlib.location.Location(latitude, longitude, tz="UTC", altitude=altitude)
    clearsky = site.get_clearsky(times_utc, model="ineichen")
    return clearsky["ghi"]


def downscale_ghi_csi_preserving(
    ghi_hourly: pd.Series,
    latitude: float,
    longitude: float,
    target_freq: str = "15min",
) -> pd.DataFrame:
    """
    Downscale hourly NASA POWER GHI to `target_freq` while preserving solar
    geometry, per the clear-sky-index method described in the module docstring.

    Parameters
    ----------
    ghi_hourly : pd.Series, tz-aware UTC index, hourly frequency
        NASA POWER's ALLSKY_SFC_SW_DWN.
    latitude, longitude : float
        Resolved plant coordinates (Task 4).

    Returns
    -------
    pd.DataFrame indexed at `target_freq`, columns:
        ghi_15min, clearsky_ghi_15min, kt_15min
    """
    assert ghi_hourly.index.tz is not None, "ghi_hourly index must be tz-aware UTC"

    # Step 1: clear-sky GHI at BOTH resolutions (hourly for Kt, fine for reconstruction)
    clearsky_hourly = compute_clearsky_ghi(ghi_hourly.index, latitude, longitude)

    fine_index = pd.date_range(
        ghi_hourly.index.min(), ghi_hourly.index.max(), freq=target_freq, tz="UTC"
    )
    clearsky_fine = compute_clearsky_ghi(fine_index, latitude, longitude)

    # Step 2: hourly clear-sky index. Guard division by near-zero clear-sky
    # (pre-dawn/post-dusk) -- Kt is undefined there and should stay NaN, not
    # blow up to a huge or negative number.
    kt_hourly = ghi_hourly / clearsky_hourly.replace(0, np.nan)
    kt_hourly = kt_hourly.clip(lower=0, upper=1.2)  # 1.2 allows brief cloud-edge enhancement

    # Step 3: interpolate Kt to fine resolution. Kt is smooth; GHI is not.
    kt_fine = kt_hourly.reindex(kt_hourly.index.union(fine_index)).interpolate(
        method="time", limit_direction="both"
    ).reindex(fine_index)

    # Night-time: clear-sky GHI ~ 0, Kt undefined -> force GHI to exactly 0
    # rather than propagating NaN or interpolation noise into the dark hours.
    is_night = clearsky_fine < 1.0  # W/m2, effectively zero
    kt_fine = kt_fine.where(~is_night, 0.0)

    # Step 4: reconstruct
    ghi_fine = (kt_fine * clearsky_fine).clip(lower=0)
    ghi_fine = ghi_fine.where(~is_night, 0.0)

    return pd.DataFrame({
        "ghi_15min": ghi_fine,
        "clearsky_ghi_15min": clearsky_fine,
        "kt_15min": kt_fine,
    })
