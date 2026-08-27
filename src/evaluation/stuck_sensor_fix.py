"""
src/evaluation/stuck_sensor_fix.py

Task 7.5: re-attempts the stuck-sensor / N3-curtailment discriminator idea
floated (but not built or tested) in docs/LAYER3_HANDOFF.md SS3, now that
the full 30-replicate x 2-plant scale is available (tens of A8/N3 examples
each, vs. the n=1-2 that made the FIRST attempted fix -- a different idea,
based on residual sign-consistency -- unreliable and led to it being
reverted after a severe regression, 8/8->4/8 and 7/7->1/7).

THE IDEA (LAYER3_HANDOFF.md SS3): curtailment caps are set BELOW capacity
by construction (hard_negatives.py's synthesize_n3_curtailment clips
`ac_power` to `cap_frac_of_capacity` in [0.5, 0.8]) and only clip values
that would otherwise EXCEED the cap -- so during a real N3 event, the
physical ceiling (physics_ceiling_fraud_kw, config.yaml's strict 1.05x
tolerance, the one Layer 1 reserves for Layer 3 max-severity checks) stays
AT OR ABOVE the frozen/capped value for the ENTIRE clipped run, by
construction. A genuinely stuck sensor has no such constraint -- if
irradiance drops (afternoon fade, passing cloud) far enough that the
physical ceiling itself falls BELOW the frozen value, that frozen value
has become physically impossible, which only a truly stuck/replayed
reading (not a legitimate cap) can produce.

DISCRIMINATOR: for each run flag_stuck_sensor() flags, additionally
require that physics_ceiling_fraud_kw drops below the frozen value at
least once during the run's exact span (find_stuck_run_details() from
flag_shap.py, reused rather than rederiving the run bounds a third time).

THIS FILE DOES NOT MODIFY src/layer3_anomaly/residual.py. Built here as an
isolated, side-by-side comparison first -- exactly per this task's own
methodology (verify before merging) and TASK7_FULL_ROADMAP SS0.3's
explicit instruction: "If it still regresses A8 detection meaningfully,
revert again and document the negative result." Whether to actually merge
this into residual.py is decided AFTER running evaluate_stuck_sensor_
variants() below, based on the real numbers -- see run() for the decision
and its stated reasoning.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

import pandas as pd

from dashboard.lib.pipeline import PLANTS, load_plant_base
from src.layer3_anomaly.residual import compute_daily_stuck_sensor_flag
from src.layer4_xai.flag_shap import find_stuck_run_details
from src.evaluation.harness import score_dataframe, N_REPLICATES, load_ground_truth

CEILING_COL = "physics_ceiling_fraud_kw"


def stuck_sensor_v2_day_flags(
    ts_table: pd.DataFrame, run_length_threshold: int = 8, ceiling_col: str = CEILING_COL,
) -> pd.Series:
    """
    Day-level stuck-sensor flag, v2: same base rule as
    residual.flag_stuck_sensor() (N=8 consecutive identical daylight
    ac_power values), PLUS requires the strict physics ceiling to drop
    below the frozen value at some point during the run's exact span. See
    module docstring for why this discriminates genuine freeze from
    legitimate curtailment. Returns one boolean per date present in
    ts_table (True only for runs that pass BOTH the base rule and this
    discriminator).
    """
    base_flags = compute_daily_stuck_sensor_flag(ts_table, run_length_threshold)
    date_series = pd.Series(ts_table.index.date, index=ts_table.index)

    result = pd.Series(False, index=base_flags.index)
    for d in base_flags.index[base_flags]:
        day_ts = ts_table[date_series == d]
        details = find_stuck_run_details(day_ts, run_length_threshold)
        if details is None:
            continue
        run_slice = day_ts.loc[details["start_ts"]:details["end_ts"], ceiling_col]
        result[d] = bool((run_slice < details["frozen_value_kw"]).any())
    return result


def evaluate_stuck_sensor_variants() -> pd.DataFrame:
    """
    Scores every replicate x both plants with BOTH the old (base) and new
    (ceiling-discriminated) stuck-sensor rule, joined against sealed
    ground truth. One row per (plant, replicate, date) that has a
    ground-truth label -- old_flag/new_flag are the two rules' day-level
    booleans for that date (independent of whether a stuck-sensor run was
    even present -- most days are False under both).
    """
    rows = []
    for plant in PLANTS:
        base = load_plant_base(plant)
        twin, capacity_kw, meta = base["twin"], base["capacity_kw"], base["meta"]
        gt = load_ground_truth(plant).copy()
        gt["date"] = pd.to_datetime(gt["date"]).dt.date

        for rep_id in range(N_REPLICATES):
            rep_path = ROOT / "data" / "processed" / "injected" / plant / f"replicate_{rep_id:03d}.parquet"
            df = pd.read_parquet(rep_path)
            result = score_dataframe(df, twin, base, capacity_kw, meta)
            ts_table = result["ts_table"]

            old_flags = compute_daily_stuck_sensor_flag(ts_table)
            new_flags = stuck_sensor_v2_day_flags(ts_table)

            rep_gt = gt[gt["replicate"] == rep_id]
            for _, row in rep_gt.iterrows():
                d = row["date"]
                rows.append(dict(
                    plant=plant, replicate=rep_id, date=str(d), kind=row["kind"], type=row["type"],
                    old_flag=bool(old_flags.get(d, False)), new_flag=bool(new_flags.get(d, False)),
                ))
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    a8 = df[df["type"] == "A8_flatline"]
    n3 = df[df["type"] == "N3_curtailment"]
    other_hard_neg = df[(df["kind"] == "hard_negative") & (df["type"] != "N3_curtailment")]

    def _rate(sub: pd.DataFrame, col: str, want_true: bool) -> dict:
        n = len(sub)
        if n == 0:
            return {"n": 0, "rate": None}
        hit = int((sub[col] == want_true).sum())
        return {"n": n, "n_hit": hit, "rate": hit / n}

    return dict(
        a8_recall_old=_rate(a8, "old_flag", True),
        a8_recall_new=_rate(a8, "new_flag", True),
        n3_specificity_old=_rate(n3, "old_flag", False),
        n3_specificity_new=_rate(n3, "new_flag", False),
        other_hard_negative_specificity_old=_rate(other_hard_neg, "old_flag", False),
        other_hard_negative_specificity_new=_rate(other_hard_neg, "new_flag", False),
    )


DECISION_TEMPLATE = (
    "DECISION: {decision}. A8 recall {a8_old_rate:.3f} (n={a8_n}) -> "
    "{a8_new_rate:.3f}; N3 specificity {n3_old_rate:.3f} (n={n3_n}) -> "
    "{n3_new_rate:.3f}. {reasoning}"
)


def run() -> dict:
    raw = evaluate_stuck_sensor_variants()
    summary = summarize(raw)

    a8_old = summary["a8_recall_old"]["rate"]
    a8_new = summary["a8_recall_new"]["rate"]
    n3_old = summary["n3_specificity_old"]["rate"]
    n3_new = summary["n3_specificity_new"]["rate"]

    # Decision rule, stated explicitly: the fix is worth keeping only if it
    # improves N3 specificity WITHOUT a "meaningful" A8 recall cost. Per
    # TASK7_FULL_ROADMAP SS0.3 ("if it still regresses A8 detection
    # meaningfully, revert"), a >=10 percentage-point A8 recall drop counts
    # as meaningful -- the same order of magnitude as the FIRST fix's
    # regression (50-86 percentage points, an unambiguous case), but strict
    # enough that a genuinely damaging trade isn't waved through just
    # because it's smaller than that first attempt's.
    a8_drop = (a8_old - a8_new) if (a8_old is not None and a8_new is not None) else None
    n3_gain = (n3_new - n3_old) if (n3_old is not None and n3_new is not None) else None

    if a8_drop is not None and a8_drop >= 0.10:
        decision = "REVERTED"
        reasoning = (
            f"A8 recall dropped by {a8_drop:.3f} ({a8_drop*100:.1f} pp), meeting the "
            f"revert threshold (>=10pp) even at full 30-replicate scale. "
            f"src/layer3_anomaly/residual.py is NOT modified."
        )
    elif n3_gain is not None and n3_gain > 0:
        decision = "MERGE-CANDIDATE (not yet applied to production residual.py -- flagged for review)"
        reasoning = (
            f"N3 specificity improved by {n3_gain:.3f} ({n3_gain*100:.1f} pp) with an "
            f"acceptable A8 recall cost ({0 if a8_drop is None else a8_drop:.3f}). "
            f"Still NOT auto-applied to production code -- this module reports the "
            f"finding; merging into residual.py is a separate, deliberate follow-up "
            f"step, not automated here."
        )
    else:
        decision = "REVERTED (no N3 specificity improvement found)"
        reasoning = "The discriminator did not improve N3 specificity at full scale."

    decision_text = DECISION_TEMPLATE.format(
        decision=decision,
        a8_old_rate=a8_old or 0.0, a8_n=summary["a8_recall_old"]["n"],
        a8_new_rate=a8_new or 0.0,
        n3_old_rate=n3_old or 0.0, n3_n=summary["n3_specificity_old"]["n"],
        n3_new_rate=n3_new or 0.0,
        reasoning=reasoning,
    )

    return dict(summary=summary, decision=decision, decision_text=decision_text, raw_n=len(raw))


def main() -> None:
    result = run()
    out_path = ROOT / "reports" / "tables" / "stuck_sensor_fix_reattempt.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Wrote {out_path}")
    print()
    print(result["decision_text"])
    print()
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
