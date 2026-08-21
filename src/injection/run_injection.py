"""
src/injection/run_injection.py

OWNED BY: Ekta (per current team allocation — Injection Framework).
THE FREEZE BOUNDARY.

Orchestrates the bootstrap-replicated injection described in
config/anomalies.yaml. Produces:

    data/processed/injected/<plant>_replicate_<NNN>.parquet
        -- full 9-day pool with ac_power (and evidence_onsite_irradiation
           where relevant) modified per that replicate's random assignment.
           Everything else (twin_* features) is UNTOUCHED -- the twin's
           inputs describe real weather; only the claimed/evidence columns
           are the injected "lie".

    data/ground_truth/<plant>_labels.parquet
        -- SEALED. One row per (replicate, date): anomaly_type, params,
           is_positive, is_hard_negative. Layer 3 / evaluation code reads
           this ONLY at final scoring time, never during development.

FREEZE PROTOCOL (do this every time, no exceptions):
    1. Run this script.
    2. Inspect the summary printout + a few sampled days manually.
    3. git add config/anomalies.yaml src/injection/ data/processed/injected/ \\
              data/ground_truth/
    4. git commit -m "Freeze injector v1.0"
    5. git tag injector-v1.0
    6. git push --tags
    7. From this point, DO NOT edit config/anomalies.yaml, taxonomy.py, or
       hard_negatives.py without bumping to v1.1, re-running in full, and
       re-tagging. Whoever tunes Layer 3 must work only against tagged output.

Usage:
    python -m src.injection.run_injection --plant plant_1
    python -m src.injection.run_injection --plant plant_2
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.injection import taxonomy as tax
from src.injection import hard_negatives as hn


def derive_plant_seed(base_seed: int, plant_key: str) -> int:
    """
    Fold plant_key into the base seed via a stable hash (NOT Python's
    built-in hash(), which is randomized per-process unless PYTHONHASHSEED
    is fixed -- hashlib gives the same result on every machine and every run).

    Without this, `np.random.default_rng(seed + rep)` produces the IDENTICAL
    stream for every plant that shares the same seed and replicate count --
    not just the same anomaly TYPE per day-slot, but bit-identical sampled
    PARAMETERS too, since the RNG call sequence is otherwise plant-agnostic.
    That silently correlates the two plants' "independent" evaluation
    replicates, which specifically undermines pooled precision/recall and
    the leave-one-plant-out generalisation check (Overview v2 SS4.8/SS9).
    """
    digest = hashlib.sha256(f"{base_seed}:{plant_key}".encode()).hexdigest()
    return int(digest, 16) % (2**31 - 1)


def load_config(path: str = "config/anomalies.yaml") -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def slice_day(df: pd.DataFrame, date) -> pd.DataFrame:
    return df[df.index.date == date]


def assign_replicate_labels(
    eligible_dates: list, rng: np.random.Generator, cfg: dict
) -> dict:
    """
    For ONE replicate: decide per-day what happens, over whatever dates are
    passed in (A6's reserved block, if any, has already been removed from
    this list by the caller). Returns
    {date: ("clean"|"positive"|"hard_negative", type_or_None)}.
    """
    n = len(eligible_dates)
    assignment = {d: ("clean", None) for d in eligible_dates}

    if n == 0:
        return assignment

    n_positive = max(1, int(round(n * cfg["per_replicate_positive_frac"])))
    n_hard_neg = max(1, int(round(n * cfg["per_replicate_hard_negative_frac"])))

    shuffled = list(eligible_dates)
    rng.shuffle(shuffled)

    positive_types = [t for t in tax.POSITIVE_TYPES if t != "A6_slow_drift"]  # A6 reserved separately, before this function runs
    for i, d in enumerate(shuffled[:n_positive]):
        chosen_type = positive_types[rng.integers(0, len(positive_types))]
        assignment[d] = ("positive", chosen_type)

    for d in shuffled[n_positive:n_positive + n_hard_neg]:
        chosen_type = hn.HARD_NEGATIVE_TYPES[rng.integers(0, len(hn.HARD_NEGATIVE_TYPES))]
        assignment[d] = ("hard_negative", chosen_type)

    return assignment


def run_replicate(
    full_df: pd.DataFrame,
    eligible_dates: list,
    capacity_kw: float,
    replicate_id: int,
    rng: np.random.Generator,
    cfg: dict,
) -> tuple[pd.DataFrame, list[dict]]:
    """Runs ONE bootstrap replicate over the full eligible pool.

    A6_slow_drift is handled BEFORE the normal per-day assignment: it
    consumes a contiguous block of `ramp_days` days, not a single day like
    the other 7 positive types, so it must reserve its block first and let
    the remaining days go through the normal per-day shuffle.
    """
    # .index.date returns a numpy array (no .isin); wrap in a Series aligned
    # to the index to get a proper boolean mask.
    date_mask = pd.Series(full_df.index.date, index=full_df.index).isin(eligible_dates)
    result_df = full_df[date_mask].copy()
    labels = []

    # --- A6 slow-drift block reservation ---
    # ~1-in-8 replicates get an A6 block, consistent with A6 being 1 of the
    # 8 positive types (the other 7 are drawn per-day at roughly equal odds
    # via assign_replicate_labels below).
    a6_dates = None
    remaining_dates = list(eligible_dates)
    if rng.random() < 1.0 / 8.0:
        lo, hi = cfg["positive"]["A6_slow_drift"]["ramp_days"]
        max_len = min(hi, len(eligible_dates))
        min_len = min(lo, max_len)
        if max_len >= 1:
            block_len = int(rng.integers(min_len, max_len + 1))
            max_start = len(eligible_dates) - block_len
            if max_start >= 0:
                start = int(rng.integers(0, max_start + 1))
                a6_dates = eligible_dates[start:start + block_len]
                remaining_dates = [d for d in eligible_dates if d not in set(a6_dates)]

    if a6_dates:
        day_frames = [slice_day(full_df, d) for d in a6_dates]
        a6_results = tax.apply_a6_slow_drift(day_frames, rng, cfg["positive"]["A6_slow_drift"])
        for date, r in zip(a6_dates, a6_results):
            day = slice_day(full_df, date)
            if day.empty:
                continue
            result_df.loc[day.index, "ac_power"] = r["ac_power"]
            labels.append({"replicate": replicate_id, "date": str(date),
                            "kind": "positive", "type": "A6_slow_drift", **r["meta"]})

    # --- normal per-day assignment over whatever days A6 didn't claim ---
    assignment = assign_replicate_labels(remaining_dates, rng, cfg)

    for date, (kind, atype) in assignment.items():
        day = slice_day(full_df, date)
        if day.empty:
            continue

        if kind == "clean":
            labels.append({"replicate": replicate_id, "date": str(date),
                            "kind": "clean", "type": None})
            continue

        if kind == "positive":
            params = cfg["positive"][atype]
            if atype == "A3_night_injection":
                r = tax.apply_a3_night_injection(day, rng, params, capacity_kw)
            elif atype == "A4_day_swap":
                # source = highest-power day in the FULL dataset (any split
                # outside eligible pool is fine -- we only borrow its SHAPE)
                daily_power = full_df.groupby(full_df.index.date)["ac_power"].mean()
                source_date = daily_power.idxmax()
                source_day = slice_day(full_df, source_date)
                r = tax.apply_a4_day_swap(day, source_day)
            elif atype == "A5_weather_spoofing":
                r = tax.apply_a5_weather_spoofing(day, rng, params)
            elif atype == "A7_capacity_violation":
                r = tax.apply_a7_capacity_violation(day, rng, params, capacity_kw)
            elif atype == "A1_constant_overclaim":
                r = tax.apply_a1_constant_overclaim(day, rng, params)
            elif atype == "A2_peak_hour_shaving":
                r = tax.apply_a2_peak_hour_shaving(day, rng, params)
            elif atype == "A8_flatline":
                r = tax.apply_a8_flatline(day, rng, params)
            else:
                continue

            result_df.loc[day.index, "ac_power"] = r["ac_power"]
            if r.get("evidence_onsite_irradiation") is not None:
                result_df.loc[day.index, "evidence_onsite_irradiation"] = r["evidence_onsite_irradiation"]
            labels.append({"replicate": replicate_id, "date": str(date),
                            "kind": "positive", "type": atype, **r["meta"]})

        elif kind == "hard_negative":
            params = cfg["hard_negatives"].get(atype, {})
            if atype in ("N1_inverter_outage", "N5_heavy_cloud_day", "N7_cloud_enhancement"):
                # selection-based: label the REAL day, no modification
                labels.append({"replicate": replicate_id, "date": str(date),
                                "kind": "hard_negative", "type": atype, "note": "real event, unmodified"})
                continue
            elif atype == "N2_soiling_loss":
                day_position = eligible_dates.index(date) + 1
                r = hn.synthesize_n2_soiling_loss(day, day_position, params)
            elif atype == "N3_curtailment":
                r = hn.synthesize_n3_curtailment(day, rng, params, capacity_kw)
            elif atype == "N4_sensor_dropout":
                r = hn.synthesize_n4_sensor_dropout(day, rng, params)
            elif atype == "N6_extreme_heat":
                r = hn.synthesize_n6_extreme_heat(day, rng, params)
            else:
                continue

            result_df.loc[day.index, "ac_power"] = r["ac_power"]
            labels.append({"replicate": replicate_id, "date": str(date),
                            "kind": "hard_negative", "type": atype, **r["meta"]})

    return result_df, labels


def run(plant_key: str, cfg_path: str = "config/anomalies.yaml") -> None:
    cfg = load_config(cfg_path)
    seed = derive_plant_seed(cfg["seed"], plant_key)
    base_rng = np.random.default_rng(seed)

    processed_path = Path(f"data/processed/{plant_key}.parquet")
    full_df = pd.read_parquet(processed_path)
    meta_path = Path(f"data/processed/{plant_key}_meta.yaml")
    with open(meta_path) as fh:
        capacity_kw = yaml.safe_load(fh)["ac_nameplate_kw"]

    lo, hi = cfg["eligible_days"]
    first_day = full_df.index.normalize().min()
    day_number = (full_df.index.normalize() - first_day).days + 1
    eligible_mask = (day_number >= lo) & (day_number <= hi)
    eligible_dates = sorted(set(full_df.index[eligible_mask].date))

    print(f"[{plant_key}] eligible pool: {len(eligible_dates)} days "
          f"({eligible_dates[0]} to {eligible_dates[-1]})")
    print(f"[{plant_key}] capacity (frozen from Layer 1): {capacity_kw:.1f} kW")

    n_replicates = cfg["n_replicates"]
    out_dir = Path(f"data/processed/injected/{plant_key}")
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_dir = Path("data/ground_truth")
    gt_dir.mkdir(parents=True, exist_ok=True)

    all_labels = []
    for rep in range(n_replicates):
        # independent, reproducible seed per replicate, derived from the
        # PLANT-SPECIFIC seed -- so the whole run is fully reproducible from
        # `seed` + `plant_key` alone, replicates aren't identical to each
        # other, and the two plants no longer share a seed stream.
        rep_rng = np.random.default_rng(seed + rep)
        result_df, labels = run_replicate(full_df, eligible_dates, capacity_kw, rep, rep_rng, cfg)
        result_df.to_parquet(out_dir / f"replicate_{rep:03d}.parquet")
        all_labels.extend(labels)

    labels_df = pd.DataFrame(all_labels)
    labels_df.to_parquet(gt_dir / f"{plant_key}_labels.parquet")

    print(f"[{plant_key}] {n_replicates} replicates written to {out_dir}/")
    print(f"[{plant_key}] ground truth: {len(labels_df)} labeled day-instances "
          f"-> {gt_dir}/{plant_key}_labels.parquet")
    print(f"[{plant_key}] label distribution:")
    print(labels_df["kind"].value_counts().to_string())
    print(f"[{plant_key}] positive type distribution:")
    print(labels_df[labels_df["kind"] == "positive"]["type"].value_counts().to_string())
    print(f"[{plant_key}] hard-negative type distribution:")
    print(labels_df[labels_df["kind"] == "hard_negative"]["type"].value_counts().to_string())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plant", required=True, choices=["plant_1", "plant_2"])
    ap.add_argument("--config", default="config/anomalies.yaml")
    args = ap.parse_args()
    run(args.plant, args.config)
    print("\nNext: inspect output, then follow the FREEZE PROTOCOL in this file's docstring.")
    return 0


if __name__ == "__main__":
    sys.exit(main())