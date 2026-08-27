"""
src/evaluation/benchmark.py

Task 7.1: full benchmark -- precision/recall/F1/AUC-PR per anomaly type
(A1-A8), overall metrics, and hard-negative specificity per type (N1-N7)
and overall, across all N_REPLICATES replicates x both plants. Uses the
7.0 harness (src/evaluation/harness.py) for all scoring -- this module
only joins scored predictions against data/ground_truth/ and computes
metrics; it reimplements no detection logic.

Sealed ground truth (data/ground_truth/{plant}_labels.parquet) is read
here for the first and only time this task touches it, per TASK7_FULL_
ROADMAP SS1.6 / Overview v2 SS8.1 -- nothing in Layer 3 is recalibrated
against it.

METHODOLOGY NOTE, stated plainly (same discipline as every other honest
scoping call in this project): a "detected" day is defined as ANY
flagged_* classification (flagged_overclaim / flagged_night_injection /
flagged_stuck_sensor), regardless of whether the mechanism that fired
"matches" the injected anomaly's mechanism -- Layer 3's three detection
paths are not individually labeled with which injected type they target
at prediction time, only whether a day is flagged at all. Consequently:
  - Per-type RECALL (A1-A8) is well-defined: TP_i / n_i, the fraction of
    that type's injected days that were flagged by ANY mechanism.
  - Per-type PRECISION is NOT independently well-defined per type, because
    false positives have no injected type of their own. This module
    reports precision_i using a SHARED false-positive pool (all clean +
    hard-negative days flagged, pooled once) reused across every type's
    precision calculation -- read precision_i as "precision if the only
    question were distinguishing type Ai from non-fraud days", not as a
    fully independent per-type figure. This is stated explicitly in the
    output JSON's "methodology_note" field, not just here.
  - AUC-PR uses severity_kwh (Layer 3.6's flag-type-aware magnitude,
    0.0 for unflagged days) as the ranking score against binary
    is_positive labels -- comparable across all three detection
    mechanisms since severity_kwh is already flag-type-aware (SS1.5 of
    this task's roadmap).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

import pandas as pd
from sklearn.metrics import average_precision_score

from dashboard.lib.pipeline import PLANTS, load_plant_base
from src.evaluation.harness import score_all_replicates, load_ground_truth

METHODOLOGY_NOTE = (
    "Every metrics block is reported under TWO detection definitions: "
    "'production_flag_only' (Layer 3.3's classify_days() flag_type -- "
    "overclaim/night-injection/stuck-sensor -- what the dashboard and "
    "evidence bundles actually use) and 'including_cusum' (also counts "
    "CUSUM's own day-level flag, which is architecturally SEPARATE from "
    "classify_days() -- conformal.py never consults it -- but is Overview "
    "v2 SS3.5's stated intended mechanism for A6_slow_drift specifically; "
    "reporting only the production flag would silently understate A6 "
    "recall). A day is 'detected' if flagged by ANY counted mechanism, "
    "regardless of whether that mechanism 'matches' the injected anomaly. "
    "Per-type recall is independently well-defined; per-type precision "
    "reuses a SHARED clean+hard-negative false-positive pool across all "
    "types (false positives have no injected type of their own) -- read "
    "precision_i as 'precision vs. non-fraud days for this type "
    "specifically', not a fully independent per-type statistic. AUC-PR "
    "ranks by severity_kwh (flag-type-aware magnitude, unaffected by the "
    "CUSUM question) against binary is_positive labels."
)


def build_eval_frame(plant: str, base: dict | None = None) -> pd.DataFrame:
    """One row per (replicate, date): scored prediction joined against
    sealed ground truth. `kind` in {clean, positive, hard_negative}."""
    if base is None:
        base = load_plant_base(plant)
    scored = score_all_replicates(plant, base=base).reset_index()

    gt = load_ground_truth(plant).copy()
    gt["date"] = pd.to_datetime(gt["date"]).dt.date

    merged = scored.merge(gt[["replicate", "date", "kind", "type"]], on=["replicate", "date"], how="left")
    missing = merged["kind"].isna().sum()
    if missing:
        raise ValueError(
            f"{missing} scored day(s) for {plant} had no matching ground-truth label -- "
            f"every eligible day should be labeled clean/positive/hard_negative by the "
            f"frozen injector. Investigate before trusting any metric below."
        )

    merged["detected"] = merged["flag_type"].astype(str).str.startswith("flagged")
    # CUSUM (3.5) is architecturally SEPARATE from classify_days()'s flag_type --
    # conformal.py's classify_day() never consults cusum_flagged, it only feeds
    # 3.6's aggregate risk score as one of five components. But Overview v2 SS3.5
    # explicitly frames CUSUM as the actual intended detection mechanism for A6
    # (persistent low-magnitude drift too small to trip any single-day threshold).
    # Reporting ONLY `detected` (the production classify_days() flag, which is
    # what the dashboard/evidence bundles actually use) would silently understate
    # A6 recall by omitting the one mechanism designed to catch it -- so both
    # definitions are computed and reported, never just one.
    merged["detected_with_cusum"] = merged["detected"] | merged["cusum_flagged"].fillna(False)
    merged["is_positive"] = merged["kind"] == "positive"
    merged["is_hard_negative"] = merged["kind"] == "hard_negative"
    merged["plant"] = plant
    return merged


def _prf1(tp: int, fp: int, fn: int) -> tuple[float | None, float | None, float | None]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    return precision, recall, f1


def compute_overall_metrics(df: pd.DataFrame, detected_col: str = "detected") -> dict:
    y_true = df["is_positive"].astype(int)
    y_pred = df[detected_col].astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision, recall, f1 = _prf1(tp, fp, fn)

    auc_pr = None
    if 0 < y_true.sum() < len(y_true):
        auc_pr = float(average_precision_score(y_true, df["severity_kwh"]))

    return dict(
        n=len(df), n_positive=int(y_true.sum()), n_negative=int((1 - y_true).sum()),
        tp=tp, fp=fp, fn=fn, tn=tn,
        precision=precision, recall=recall, f1=f1, auc_pr=auc_pr,
    )


def compute_per_type_metrics(df: pd.DataFrame, detected_col: str = "detected") -> tuple[dict, dict]:
    negative_mask = ~df["is_positive"]  # clean + hard_negative, the shared FP pool
    fp_pool_flagged = int((negative_mask & df[detected_col]).sum())
    fp_pool_total = int(negative_mask.sum())

    out = {}
    for t in sorted(df.loc[df["is_positive"], "type"].dropna().unique()):
        sub = df[df["type"] == t]
        n = len(sub)
        tp = int(sub[detected_col].sum())
        precision, recall, f1 = _prf1(tp, fp_pool_flagged, n - tp)
        out[t] = dict(n=n, n_detected=tp, precision=precision, recall=recall, f1=f1)

    return out, dict(shared_fp_pool_flagged=fp_pool_flagged, shared_fp_pool_total=fp_pool_total)


def compute_hard_negative_specificity(df: pd.DataFrame, detected_col: str = "detected") -> tuple[dict, dict]:
    out = {}
    for t in sorted(df.loc[df["is_hard_negative"], "type"].dropna().unique()):
        sub = df[df["type"] == t]
        n = len(sub)
        tn = int((~sub[detected_col]).sum())
        out[t] = dict(n=n, n_correctly_unflagged=tn, specificity=(tn / n if n > 0 else None))

    hn = df[df["is_hard_negative"]]
    n = len(hn)
    tn = int((~hn[detected_col]).sum())
    overall = dict(n=n, n_correctly_unflagged=tn, specificity=(tn / n if n > 0 else None))
    return out, overall


def _metrics_block(df: pd.DataFrame, detected_col: str) -> dict:
    per_type, fp_pool = compute_per_type_metrics(df, detected_col)
    hn_per_type, hn_overall = compute_hard_negative_specificity(df, detected_col)
    return dict(
        overall=compute_overall_metrics(df, detected_col),
        per_type=per_type,
        shared_fp_pool=fp_pool,
        hard_negative_per_type=hn_per_type,
        hard_negative_overall=hn_overall,
    )


def run_benchmark() -> dict:
    """
    Computes metrics under TWO detection definitions, both reported, never
    just one: `detected` (Layer 3.3's production classify_days() flag_type
    -- what the dashboard/evidence bundles actually use) and
    `detected_with_cusum` (also counting CUSUM's own day-level flag).
    Overview v2 SS3.5 frames CUSUM as the actual intended mechanism for A6
    (persistent low-magnitude drift too small to trip any single-day
    threshold) -- CUSUM is architecturally NOT part of classify_days()
    (conformal.py never consults it), so reporting only `detected` would
    silently understate A6 recall by omitting the one signal designed to
    catch it. See build_eval_frame()'s comment for the same point.
    """
    per_plant = {}
    frames = []
    for plant in PLANTS:
        base = load_plant_base(plant)
        df = build_eval_frame(plant, base=base)
        frames.append(df)
        per_plant[plant] = dict(
            production_flag_only=_metrics_block(df, "detected"),
            including_cusum=_metrics_block(df, "detected_with_cusum"),
        )

    pooled_df = pd.concat(frames)
    pooled = dict(
        production_flag_only=_metrics_block(pooled_df, "detected"),
        including_cusum=_metrics_block(pooled_df, "detected_with_cusum"),
    )

    return dict(methodology_note=METHODOLOGY_NOTE, per_plant=per_plant, pooled=pooled)


def main() -> None:
    results = run_benchmark()

    out_path = ROOT / "reports" / "tables" / "task7_benchmark_full.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Wrote {out_path}")
    print("\n=== POOLED OVERALL (production flag only) ===")
    print(json.dumps(results["pooled"]["production_flag_only"]["overall"], indent=2))
    print("\n=== POOLED OVERALL (including CUSUM) ===")
    print(json.dumps(results["pooled"]["including_cusum"]["overall"], indent=2))
    print("\n=== POOLED HARD-NEGATIVE SPECIFICITY (production flag only) ===")
    print(json.dumps(results["pooled"]["production_flag_only"]["hard_negative_overall"], indent=2))
    print(
        "\nSanity check vs. the already-known bounded result "
        "(LAYER3_HANDOFF.md SS8: specificity 0.929 at n=28) -- "
        "should be broadly consistent, not wildly different."
    )


if __name__ == "__main__":
    main()
