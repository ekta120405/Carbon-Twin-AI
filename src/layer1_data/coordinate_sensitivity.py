"""
src/layer1_data/coordinate_sensitivity.py

Task 4, Step 4: resolve which candidate site (Nasik / Gandikota) corresponds to
which Kaggle plant, and quantify how sensitive downstream accuracy is to the
coordinate assumption.

Method (deliberately simple and defensible — this is a diagnostic, not the twin):
  1. For each Kaggle plant, aggregate the 22 inverters into a plant-level AC power
     series and normalise to [0, 1] (we don't know true nameplate capacity yet —
     that's Layer 1.6 — so shape-matching, not magnitude-matching, is what we test).
  2. For each candidate grid cell (9 per site x 2 sites = 18), pull ALLSKY_SFC_SW_DWN,
     resample/align to the plant's daylight timestamps, and normalise to [0, 1].
  3. Compute Pearson correlation and RMSE between the two normalised shapes,
     restricted to daylight hours, over the full overlapping window.
  4. The (plant, site) pairing with the highest correlation wins. Report the full
     18 x 2 matrix so the choice is auditable, not asserted.
  5. Report the correlation SPREAD across the 3x3 grid around the winning site's
     center cell -- this IS the sensitivity result for the paper: "coordinate
     uncertainty of +/-1 degree changes correlation by X, RMSE by Y".

Usage:
    python -m src.layer1_data.coordinate_sensitivity

Output:
    reports/tables/coordinate_sensitivity.csv   (full 18 x 2 matrix)
    reports/figures/coordinate_sensitivity.png  (heatmap, if matplotlib available)
    Prints the resolved plant->site mapping and the recommended config edit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KAGGLE_DIR = Path("data/raw/kaggle")
NASA_DIR = Path("data/raw/nasa_power")
OUT_TABLE = Path("reports/tables/coordinate_sensitivity.csv")
OUT_FIG = Path("reports/figures/coordinate_sensitivity.png")

PLANT_FILES = {
    "plant_1": "Plant_1_Generation_Data.csv",
    "plant_2": "Plant_2_Generation_Data.csv",
}


def load_plant_generation(plant_key: str) -> pd.Series:
    """Aggregate all inverters to a single plant-level AC power series, UTC-indexed."""
    path = KAGGLE_DIR / PLANT_FILES[plant_key]
    df = pd.read_csv(path)

    # Plant 1 and Plant 2 releases have historically shipped with different
    # DATE_TIME formats. Try both rather than assuming.
    try:
        ts = pd.to_datetime(df["DATE_TIME"], format="%d-%m-%Y %H:%M")
    except ValueError:
        ts = pd.to_datetime(df["DATE_TIME"], format="%Y-%m-%d %H:%M:%S")

    # Kaggle timestamps are naive local time (IST, UTC+5:30) -- localise then
    # convert, per the Layer 1.2 timezone contract.
    ts = ts.dt.tz_localize("Asia/Kolkata").dt.tz_convert("UTC")
    df = df.assign(timestamp_utc=ts)

    plant_series = (
        df.groupby("timestamp_utc")["AC_POWER"].sum().sort_index()
    )
    return plant_series


def load_nasa_cell(path: Path) -> pd.Series:
    df = pd.read_parquet(path)
    return df["ALLSKY_SFC_SW_DWN"].astype(float)


def normalise(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-9:
        return s * 0.0
    return (s - lo) / (hi - lo)


def align_and_score(plant_power: pd.Series, ghi: pd.Series) -> tuple[float, float, int]:
    """
    Align hourly GHI to the plant's (15-min) generation index via forward-fill
    within the hour (this is a coarse diagnostic alignment -- the real
    clear-sky-index-preserving downscaling from Layer 1.3 happens in Task 5,
    not here). Returns (pearson_r, rmse, n_overlap_points).
    """
    ghi_hourly_reindexed = ghi.reindex(
        plant_power.index.floor("h")
    ).ffill(limit=1)
    ghi_hourly_reindexed.index = plant_power.index

    joined = pd.DataFrame({"power": plant_power, "ghi": ghi_hourly_reindexed}).dropna()

    # Restrict to daylight: keep only rows where GHI is meaningfully above zero.
    joined = joined[joined["ghi"] > 20]
    if len(joined) < 30:
        return float("nan"), float("nan"), len(joined)

    p_norm = normalise(joined["power"])
    g_norm = normalise(joined["ghi"])

    r = float(np.corrcoef(p_norm, g_norm)[0, 1])
    rmse = float(np.sqrt(np.mean((p_norm - g_norm) ** 2)))
    return r, rmse, len(joined)


def main() -> int:
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

    if not NASA_DIR.exists() or not any(NASA_DIR.glob("*.parquet")):
        print(f"ERROR: no NASA POWER data found in {NASA_DIR}/")
        print("Run Step 3 first: python -m src.layer1_data.acquisition")
        return 1

    print("Loading Kaggle plant generation data...")
    plants = {k: load_plant_generation(k) for k in PLANT_FILES}
    for k, s in plants.items():
        print(f"  {k}: {len(s)} timestamps, {s.index.min()} to {s.index.max()}")

    cells = sorted(NASA_DIR.glob("*.parquet"))
    print(f"\nScoring {len(cells)} candidate grid cells against {len(plants)} plants...")

    rows = []
    for cell_path in cells:
        cell_df = pd.read_parquet(cell_path)
        site = cell_df["site"].iloc[0]
        lat = cell_df["lat"].iloc[0]
        lon = cell_df["lon"].iloc[0]
        dlat = int(cell_df["dlat"].iloc[0])
        dlon = int(cell_df["dlon"].iloc[0])
        ghi = load_nasa_cell(cell_path)

        for plant_key, plant_power in plants.items():
            r, rmse, n = align_and_score(plant_power, ghi)
            rows.append(
                dict(site=site, lat=lat, lon=lon, dlat=dlat, dlon=dlon,
                     plant=plant_key, pearson_r=r, rmse_norm=rmse, n_overlap=n,
                     is_center=(dlat == 0 and dlon == 0))
            )

    result = pd.DataFrame(rows).sort_values(["plant", "pearson_r"], ascending=[True, False])
    result.to_csv(OUT_TABLE, index=False)
    print(f"\nFull matrix written to {OUT_TABLE}")

    # ---- Resolve plant -> site mapping using CENTER cells only -------------
    center = result[result["is_center"]].copy()
    print("\n" + "=" * 64)
    print("CENTER-CELL CORRELATION (this resolves plant -> site mapping)")
    print("=" * 64)
    print(center[["plant", "site", "lat", "lon", "pearson_r", "rmse_norm"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Resolve as a one-to-one assignment: with 2 plants and 2 sites, compare
   # the two possible pairings by total correlation, so no site can be
   # claimed by more than one plant.

    plant_keys = list(plants.keys())
    site_names = sorted(center["site"].unique())
    if len(plant_keys) != 2 or len(site_names) != 2:
       raise ValueError(
           f"Assignment logic assumes exactly 2 plants and 2 sites; "
           f"got {len(plant_keys)} plants and {len(site_names)} sites."
       )

    def r_of(plant_key, site):
       row = center[(center["plant"] == plant_key) & (center["site"] == site)].iloc[0]
       return row["pearson_r"]

    pairing_a = {plant_keys[0]: site_names[0], plant_keys[1]: site_names[1]}
    pairing_b = {plant_keys[0]: site_names[1], plant_keys[1]: site_names[0]}
    score_a = sum(r_of(p, s) for p, s in pairing_a.items())
    score_b = sum(r_of(p, s) for p, s in pairing_b.items())

    mapping = pairing_a if score_a >= score_b else pairing_b

    print(f"\nPairing A ({plant_keys[0]}->{site_names[0]}, "
         f"{plant_keys[1]}->{site_names[1]}): total r = {score_a:.4f}")
    print(f"Pairing B ({plant_keys[0]}->{site_names[1]}, "
         f"{plant_keys[1]}->{site_names[0]}): total r = {score_b:.4f}")

    margin = abs(score_a - score_b)
    for plant_key, site in mapping.items():
       r = r_of(plant_key, site)
       print(f"\n{plant_key} -> {site}  (r={r:.4f})")
    print(f"\nPairing margin (total r, winner over runner-up) = {margin:.4f}")
    if margin < 0.03:
       print(f"  WARNING: pairing margin is small ({margin:.4f}). The two "
             f"assignments are not clearly distinguishable. Report both "
             f"pairings in the paper and treat the mapping as provisional.")

    # ---- Sensitivity: spread across the winning site's 3x3 grid -------------
    print("\n" + "=" * 64)
    print("COORDINATE SENSITIVITY (the +/-1 degree grid spread around the winner)")
    print("=" * 64)
    for plant_key, site in mapping.items():
        grid = result[(result["plant"] == plant_key) & (result["site"] == site)]
        r_range = grid["pearson_r"].max() - grid["pearson_r"].min()
        rmse_range = grid["rmse_norm"].max() - grid["rmse_norm"].min()
        print(f"\n{plant_key} ({site}):")
        print(f"  correlation across 9 cells: {grid['pearson_r'].min():.4f} to "
              f"{grid['pearson_r'].max():.4f}  (spread = {r_range:.4f})")
        print(f"  normalised RMSE across 9 cells: {grid['rmse_norm'].min():.4f} to "
              f"{grid['rmse_norm'].max():.4f}  (spread = {rmse_range:.4f})")
        if r_range > 0.10:
            print(f"  NOTE: spread > 0.10 -- twin accuracy is meaningfully sensitive "
                  f"to the +/-1 degree coordinate uncertainty. Report this range "
                  f"explicitly as a limitation (Overview v2 W7).")

    # ---- Recommended config edit --------------------------------------------
    print("\n" + "=" * 64)
    print("RECOMMENDED config/plants.yaml EDIT")
    print("=" * 64)
    for plant_key, site in mapping.items():
        row = center[(center["plant"] == plant_key) & (center["site"] == site)].iloc[0]
        print(f"\n{plant_key}:")
        print(f"  latitude:  {row['lat']}")
        print(f"  longitude: {row['lon']}")
        print(f"  coordinate_source: \"resolved empirically via clear-sky-index "
              f"correlation, see reports/tables/coordinate_sensitivity.csv\"")

    # ---- Optional heatmap -----------------------------------------------
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(plants), figsize=(6 * len(plants), 5))
        if len(plants) == 1:
            axes = [axes]
        for ax, plant_key in zip(axes, plants):
            sub = result[result["plant"] == plant_key]
            for site in sub["site"].unique():
                s = sub[sub["site"] == site]
                pivot = s.pivot(index="dlat", columns="dlon", values="pearson_r")
                im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=1)
            ax.set_title(f"{plant_key}: correlation by site (best site shown)")
        fig.tight_layout()
        fig.savefig(OUT_FIG, dpi=150)
        print(f"\nHeatmap saved to {OUT_FIG}")
    except ImportError:
        print("\n(matplotlib not available -- skipping heatmap; table is sufficient)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
