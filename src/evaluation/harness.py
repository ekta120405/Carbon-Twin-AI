"""
src/evaluation/harness.py

Task 7.0: shared batch-scoring harness. Every other 7.x module (7.1-7.5)
calls score_replicate()/score_dataframe() instead of reimplementing the
plant -> flags pipeline -- the same "one wiring layer, reused everywhere"
principle already established for the dashboard (dashboard/lib/pipeline.py,
see docs/LAYER4_DASHBOARD_HANDOFF.md SS1).

Reuses dashboard.lib.pipeline.load_plant_base() directly for every
per-plant CALIBRATED quantity (conformal threshold + achievable alpha,
CUSUM reference mean/std, calibration-ratio clean baseline, divergence
normalization cap) -- all of these are fit ONLY on real calib/test_clean
data by that function already, and this harness never re-fits them against
injected/ground-truth data (TASK7_FULL_ROADMAP SS1.6). The one exception,
carried over unchanged from the dashboard: CUSUM's `h` is recalibrated
fresh per distinct WINDOW LENGTH being scored (never reused across window
lengths -- LAYER3_HANDOFF.md SS7), but still calibrated only from clean
reference data, cached in-process per (plant, window_len) since the
calibration is deterministic (fixed seed) and moderately expensive (2000
bootstrap resamples) -- no reason to recompute it 30 times per plant when
every injected replicate is the same 9-day window length.

Unlike dashboard.lib.pipeline.get_scored_context, this harness does NOT
build carbon exposure, severity ranking, or evidence bundles/SHAP
explanations -- a full precision/recall benchmark over 30 replicates x 2
plants needs only the day-level classification + severity numbers, not
per-flag explanations. If a future module needs those too, extend
score_dataframe's return dict rather than duplicating the pipeline again
elsewhere.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from dashboard.lib.pipeline import load_plant_base, CUSUM_K
from src.layer2_twin.physics_anchor import compute_physics_anchor
from src.layer3_anomaly.residual import (
    build_residual_table,
    compute_daily_energy_residual,
    compute_daily_night_injection_flag,
    compute_daily_stuck_sensor_flag,
)
from src.layer3_anomaly.conformal import classify_days
from src.layer3_anomaly.cusum import run_cusum, calibrate_h_from_clean_data
from src.layer3_anomaly.aggregate import compute_daily_flag_severity_kwh

N_REPLICATES = 30
REPLICATE_WINDOW_DAYS = 9  # frozen injector's eligible pool span (config/anomalies.yaml)

_H_CACHE: dict[tuple[int, int], float] = {}


def _h_for_window(base: dict, window_len: int) -> float:
    """
    CUSUM h, calibrated fresh per distinct window length -- an h calibrated
    for one window length carries no valid false-positive-rate guarantee
    when applied to a different one (LAYER3_HANDOFF.md SS7 / cusum.py's own
    module docstring: this exact bug was found and fixed once already, a
    genuinely clean plant_2 scored 30.7/"Elevated" using a 9-day h over a
    34-day sequence instead of 20.2/"Low"). Cached per (id(base), window_len)
    in-process -- deterministic given the fixed default seed, so recomputing
    it 30 times for 30 same-length replicates would be pure waste.
    """
    key = (id(base), window_len)
    if key not in _H_CACHE:
        calib = calibrate_h_from_clean_data(
            base["reference_pool"], k=CUSUM_K, window_len=window_len, target_fpr=0.05
        )
        _H_CACHE[key] = calib["recommended_h"]
    return _H_CACHE[key]


def score_dataframe(df: pd.DataFrame, twin, base: dict, capacity_kw: float, meta: dict) -> dict:
    """
    Runs Layer 2 (physics anchor + twin prediction) and Layer 3.1-3.3 +
    3.6's flag-type-aware severity on an ALREADY-LOADED dataframe (one
    injected replicate, or any other window sharing the same schema as
    data/processed/{plant}.parquet).

    Returns a dict: ts_table, daily, labels, severity, cusum (DataFrame
    indexed by date with cusum_s_t/cusum_flagged), h (the CUSUM threshold
    actually used for this call).
    """
    anchor = compute_physics_anchor(df, meta["latitude"], meta["longitude"], capacity_kw)
    df = df.join(anchor)
    ts_table = build_residual_table(twin, df["physics_ac_kw"], df, capacity_kw, twin.feature_cols)
    daily = compute_daily_energy_residual(ts_table, ac_nameplate_kw=capacity_kw)

    night_flags = compute_daily_night_injection_flag(ts_table, capacity_kw)
    stuck_flags = compute_daily_stuck_sensor_flag(ts_table)
    labels = classify_days(daily["r_day"], base["conformal_result"], night_flags, stuck_flags)

    severity = compute_daily_flag_severity_kwh(daily, labels, ts_table)

    window_len = max(len(daily), 2)
    h = _h_for_window(base, window_len)
    cusum_res = run_cusum(daily.sort_index()["r_day"], base["cusum_ref"], k=CUSUM_K, h=h)
    cusum_frame = pd.DataFrame(
        {"cusum_s_t": cusum_res.s_values, "cusum_flagged": cusum_res.flagged},
        index=pd.Index(cusum_res.dates, name=daily.index.name),
    )

    return dict(ts_table=ts_table, daily=daily, labels=labels, severity=severity,
                cusum=cusum_frame, h=h)


def score_replicate(plant: str, replicate_id: int, twin=None, base=None) -> pd.DataFrame:
    """
    Task 7.0's core entry point. Runs ONE injected replicate through the
    full detection pipeline and returns a per-day DataFrame indexed by
    date, columns: claimed_energy_kwh, expected_energy_kwh,
    excess_energy_kwh, r_day, daylight_hours, n_daylight_samples,
    flag_type, severity_kwh, cusum_s_t, cusum_flagged, plant, replicate.

    Pass `twin`/`base` (from a prior load_plant_base(plant) call) to avoid
    reloading the model per replicate in a loop -- though load_plant_base
    itself is cached (st.cache_resource works fine outside a running
    Streamlit app, verified during the 4.6 dashboard build), so omitting
    them just costs a cheap cache lookup, not a real reload.
    """
    if base is None:
        base = load_plant_base(plant)
    if twin is None:
        twin = base["twin"]

    rep_path = ROOT / "data" / "processed" / "injected" / plant / f"replicate_{replicate_id:03d}.parquet"
    df = pd.read_parquet(rep_path)

    result = score_dataframe(df, twin, base, base["capacity_kw"], base["meta"])
    out = result["daily"].copy()
    out["flag_type"] = result["labels"]
    out["severity_kwh"] = result["severity"]
    out = out.join(result["cusum"][["cusum_s_t", "cusum_flagged"]])
    out.index.name = "date"
    out["plant"] = plant
    out["replicate"] = replicate_id
    return out


def score_all_replicates(plant: str, replicate_ids=None, base=None) -> pd.DataFrame:
    """Convenience: concatenates score_replicate() over many replicates
    (default: all N_REPLICATES) for one plant, reusing one loaded base."""
    if base is None:
        base = load_plant_base(plant)
    if replicate_ids is None:
        replicate_ids = range(N_REPLICATES)
    frames = [score_replicate(plant, rep_id, twin=base["twin"], base=base) for rep_id in replicate_ids]
    return pd.concat(frames)


def load_ground_truth(plant: str, replicate_id: int | None = None) -> pd.DataFrame:
    """data/ground_truth/{plant}_labels.parquet -- sealed, read-only
    (TASK7_FULL_ROADMAP SS1). Optionally filtered to one replicate."""
    path = ROOT / "data" / "ground_truth" / f"{plant}_labels.parquet"
    gt = pd.read_parquet(path)
    if replicate_id is not None:
        gt = gt[gt["replicate"] == replicate_id]
    return gt
