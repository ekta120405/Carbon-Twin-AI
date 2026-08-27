"""
src/evaluation/leave_one_plant_out.py

Task 7.3: leave-one-plant-out generalization. Scores plant_2's 30
replicates using plant_1's trained twin, and vice versa, while keeping
every Layer 3 calibrated quantity (conformal threshold, CUSUM reference,
calibration-ratio clean baseline, divergence cap) tied to the plant
actually being SCORED (i.e., the plant being audited), never to the twin's
origin plant. This directly answers the roadmap's stated question: does
using the scored plant's own thresholds remain valid when a different
plant's twin produced the underlying prediction? Only the twin (the
learned weather -> residual-correction model) is swapped; nothing about
how "anomalous" is defined for the scored plant changes.

REAL GOTCHA FOUND BEFORE WRITING THIS, not assumed -- confirmed against
src/layer2_twin/twin_model.py's actual predict() code:

    preds[q] = physics_ac_kw.values + y_hat * self.ac_nameplate_kw

`self.ac_nameplate_kw` is the TWIN'S OWN TRAINING-TIME capacity (baked
into the loaded object, reports/models/<plant>/twin_meta.json), not
whichever plant is currently being scored. y_hat is a capacity-NORMALIZED
residual fraction (build_target(): y = (claimed - physics_anchor) /
ac_nameplate_kw) -- to correctly de-normalize it back into kW units for
the plant actually being scored, the multiplier must be the SCORED
plant's capacity, not the twin's origin plant's. Naively calling
plant_1_twin.predict(X_plant_2, physics_ac_kw_plant_2) would silently
rescale the learned correction by plant_1's capacity (27,986 kW) instead
of plant_2's (23,579 kW, a real but modest ~19% mismatch for this
project's two plants specifically -- NOT the ~10x figure that applies to
the two plants' raw per-inverter DC power magnitudes, LAYER1 notes; the
estimated ac_nameplate_kw constants are much closer since both are
derived the same way, 99.5th-percentile summed AC power at Kt>0.9). Any
mismatch here, large or small, is a silent unit error, not a genuine test
of whether the twin's LEARNED weather->residual PATTERN generalizes.

FIX: never call a foreign twin's .predict() directly. Shallow-copy the
twin object (dataclasses.replace / copy.copy -- never mutate the cached
original in place, it's shared via dashboard.lib.pipeline's
st.cache_resource across every other 7.x module) and override
ac_nameplate_kw to the SCORED plant's own capacity before predicting.
`models_` (the actual trained boosters) and `feature_cols` are left
untouched -- confirmed both plants' twins share IDENTICAL feature_cols
(verified against the real twin_meta.json files), so no column mismatch.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

import pandas as pd

from dashboard.lib.pipeline import PLANTS, load_plant_base
from src.evaluation.harness import score_replicate, N_REPLICATES
from src.evaluation.benchmark import merge_with_ground_truth, _metrics_block


def _twin_rescaled_for_plant(source_twin, target_capacity_kw: float):
    """See module docstring -- shallow-copies the twin and overrides
    ac_nameplate_kw so the learned correction de-normalizes into the
    SCORED plant's kW scale, not the twin's own training-time plant's."""
    rescaled = copy.copy(source_twin)
    rescaled.ac_nameplate_kw = target_capacity_kw
    return rescaled


def _other_plant(plant: str) -> str:
    others = [p for p in PLANTS if p != plant]
    assert len(others) == 1, f"expected exactly one other plant, got {others}"
    return others[0]


def score_cross_plant(scored_plant: str, replicate_ids=None) -> pd.DataFrame:
    """
    Scores `scored_plant`'s replicates using the OTHER plant's twin,
    keeping every Layer 3 calibrated quantity (threshold, CUSUM ref, CR
    baseline, divergence cap) tied to `scored_plant`'s own calibration.
    """
    if replicate_ids is None:
        replicate_ids = range(N_REPLICATES)

    own_base = load_plant_base(scored_plant)
    foreign_plant = _other_plant(scored_plant)
    foreign_base = load_plant_base(foreign_plant)
    cross_twin = _twin_rescaled_for_plant(foreign_base["twin"], own_base["capacity_kw"])

    frames = [
        score_replicate(scored_plant, rep_id, twin=cross_twin, base=own_base)
        for rep_id in replicate_ids
    ]
    scored = pd.concat(frames).reset_index()
    return scored


def run_leave_one_plant_out() -> dict:
    results = {}
    for plant in PLANTS:
        foreign_plant = _other_plant(plant)
        scored = score_cross_plant(plant)
        df = merge_with_ground_truth(scored, plant)
        results[plant] = dict(
            twin_source=foreign_plant,
            metrics=_metrics_block(df, "detected"),
            metrics_including_cusum=_metrics_block(df, "detected_with_cusum"),
        )
    return results


def _compare_to_own_twin_baseline(cross_results: dict) -> dict:
    """
    Pulls the already-committed same-plant (own-twin) benchmark numbers
    from reports/tables/task7_benchmark_full.json (Task 7.1) as the
    baseline for comparison -- does NOT recompute them, per this task's
    own "don't recompute what's already built and verified" discipline.
    """
    bench_path = ROOT / "reports" / "tables" / "task7_benchmark_full.json"
    if not bench_path.exists():
        return {"note": "Task 7.1 benchmark output not found -- run src/evaluation/benchmark.py first."}
    with open(bench_path) as f:
        bench = json.load(f)

    comparison = {}
    for plant in PLANTS:
        own = bench["per_plant"][plant]["production_flag_only"]["overall"]
        cross = cross_results[plant]["metrics"]["overall"]
        comparison[plant] = dict(
            own_twin=dict(precision=own["precision"], recall=own["recall"], f1=own["f1"], auc_pr=own["auc_pr"]),
            cross_twin=dict(precision=cross["precision"], recall=cross["recall"], f1=cross["f1"], auc_pr=cross["auc_pr"]),
            own_twin_hard_negative_specificity=bench["per_plant"][plant]["production_flag_only"]["hard_negative_overall"]["specificity"],
            cross_twin_hard_negative_specificity=cross_results[plant]["metrics"]["hard_negative_overall"]["specificity"],
        )
    return comparison


def main() -> None:
    cross_results = run_leave_one_plant_out()
    comparison = _compare_to_own_twin_baseline(cross_results)

    out = dict(
        methodology_note=(
            "Each plant is scored using the OTHER plant's trained twin, "
            "with the twin's ac_nameplate_kw overridden to the SCORED "
            "plant's own capacity before prediction (see module docstring "
            "for why this override is necessary, not optional). Every "
            "Layer 3 calibrated quantity (conformal threshold, CUSUM "
            "reference, calibration-ratio baseline, divergence cap) stays "
            "tied to the plant being scored/audited, never to the twin's "
            "origin plant -- only the prediction mechanism is swapped."
        ),
        cross_plant_results=cross_results,
        comparison_vs_own_twin=comparison,
    )

    out_path = ROOT / "reports" / "tables" / "leave_one_plant_out.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Wrote {out_path}")

    print("\n=== Own-twin vs. cross-twin comparison ===")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
