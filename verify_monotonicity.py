"""
verify_monotonicity.py

Run this AFTER training the PI-GBQR twin, before deciding whether to weaken
the monotone constraints. The known LightGBM issue (#3371) reports that
monotone_constraints + objective="quantile" CAN produce non-monotonic
output -- it does not say this ALWAYS happens, and it does not say training
fails. This script checks whether it actually happened on YOUR trained
model, on YOUR data, rather than assuming the worst-case from a 2020 bug
report applies here.

Checks BOTH the raw monotone-constrained q50 booster's own output AND the
pipeline's final post-sort expected_q50. These are NOT the same thing:
twin.predict() sorts q10/q50/q90 row-wise to enforce non-crossing, so if
the unconstrained q10 or q90 model crosses past the true q50 at some sweep
point (confirmed to happen for q90 in prior sklearn-stand-in testing:
16/49 steps), the sort can silently substitute a tail prediction into the
"q50" slot -- making an unconstrained-tail crossing event look like a
constrained-median violation. Reporting both lets us attribute any
violation correctly before deciding whether anything needs fixing.

Usage:
    python verify_monotonicity.py --plant plant_1
"""
import argparse
import numpy as np
import pandas as pd


def audit_monotonicity(twin, base_row: pd.Series, feature_name: str,
                        direction: int, n_points: int = 100,
                        value_range: tuple[float, float] | None = None) -> dict:
    """
    Sweeps ONE feature across its observed range while holding all others
    fixed at `base_row`'s values, and checks whether BOTH the raw
    monotone-constrained q50 booster's own output AND the pipeline's final
    post-sort expected_q50 are monotonic in the direction the constraint
    was supposed to enforce.

    direction: +1 (should be non-decreasing) or -1 (should be non-increasing)
    """
    if value_range is None:
        lo, hi = 0.0, 1000.0
    else:
        lo, hi = value_range

    sweep_df = pd.DataFrame([base_row.to_dict()] * n_points)
    sweep_df[feature_name] = np.linspace(lo, hi, n_points)
    sweep_df = sweep_df[twin.feature_cols]

    physics_placeholder = pd.Series(0.0, index=sweep_df.index)

    # Post-sort pipeline output (what Layer 3 actually consumes)
    preds = twin.predict(sweep_df, physics_placeholder)
    post_sort_curve = preds["expected_q50"].values

    # Raw q50 booster output, bypassing the sort entirely -- isolates
    # whether the monotone-constrained model itself is well-behaved
    raw_q50 = twin.models_[0.5].predict(sweep_df)
    raw_curve = physics_placeholder.values + raw_q50 * twin.ac_nameplate_kw

    def _score(curve):
        diffs = np.diff(curve) * direction
        violations = diffs[diffs < -1e-6]
        violation_rate = len(violations) / len(diffs)
        max_violation = float(-violations.min()) if len(violations) else 0.0
        return violation_rate, max_violation

    post_rate, post_max = _score(post_sort_curve)
    raw_rate, raw_max = _score(raw_curve)

    return {
        "feature": feature_name,
        "n_steps": len(post_sort_curve) - 1,
        "post_sort_violation_rate": post_rate,
        "post_sort_max_violation": post_max,
        "raw_q50_violation_rate": raw_rate,
        "raw_q50_max_violation": raw_max,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plant", required=True, choices=["plant_1", "plant_2"])
    args = ap.parse_args()

    import sys
    sys.path.insert(0, ".")
    from src.layer2_twin.twin_model import PIGBQRTwin
    import lightgbm as lgb
    import json

    model_dir = f"reports/models/{args.plant}"
    with open(f"{model_dir}/twin_meta.json") as fh:
        meta = json.load(fh)

    twin = PIGBQRTwin(feature_cols=meta["feature_cols"], ac_nameplate_kw=meta["ac_nameplate_kw"])
    twin.models_ = {}
    for q in (0.1, 0.5, 0.9):
        booster = lgb.Booster(model_file=f"{model_dir}/twin_q{int(q*100)}.txt")
        class _Wrapped:
            def __init__(self, b): self.b = b
            def predict(self, X): return self.b.predict(X)
        twin.models_[q] = _Wrapped(booster)

    df = pd.read_parquet(f"data/processed/{args.plant}.parquet")
    train_row = df[df["split"] == "train"].iloc[len(df)//2]

    checks = [
        ("twin_ghi", 1, (df["twin_ghi"].min(), df["twin_ghi"].max())),
        ("twin_clearsky_ghi", 1, (df["twin_clearsky_ghi"].min(), df["twin_clearsky_ghi"].max())),
        ("twin_kt", 1, (0.0, 1.2)),
        ("twin_cell_temp", -1, (df["twin_cell_temp"].min(), df["twin_cell_temp"].max())),
    ]

    print(f"\nMonotonicity audit for {args.plant} (raw constrained q50 vs post-sort pipeline output):\n")

    any_raw_violation = False
    for feature, direction, value_range in checks:
        result = audit_monotonicity(twin, train_row, feature, direction, value_range=value_range)
        raw_flag = "VIOLATION" if result["raw_q50_violation_rate"] > 0.02 else "clean"
        post_flag = "VIOLATION" if result["post_sort_violation_rate"] > 0.02 else "clean"
        print(f"  {feature:20s} "
              f"raw_q50: rate={result['raw_q50_violation_rate']:.3f} max={result['raw_q50_max_violation']:.2f}kW [{raw_flag}]  "
              f"post_sort: rate={result['post_sort_violation_rate']:.3f} max={result['post_sort_max_violation']:.2f}kW [{post_flag}]")
        if result["raw_q50_violation_rate"] > 0.02:
            any_raw_violation = True

    print()
    if any_raw_violation:
        print("RAW VIOLATIONS FOUND -- the monotone-constrained q50 model itself is")
        print("misbehaving. This is unexpected; check for a column-order mismatch between")
        print("X.columns and monotone_constraints, or another issue with the L2 fix.")
    else:
        print("RAW q50 clean. Any post-sort violations above are confined to the")
        print("crossing-correction touching unconstrained tails (q10/q90), not the")
        print("constrained median estimator -- report this distinction precisely in the paper.")


if __name__ == "__main__":
    main()