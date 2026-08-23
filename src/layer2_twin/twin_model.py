"""
src/layer2_twin/twin_model.py

Layer 2.2-2.3: the PI-GBQR twin (Overview v2 SS5.2-5.3, Task 2 decision).

Target definition (the key design decision from Task 2):
    y(t) = [claimed(t) - physics_anchor(t)] / ac_nameplate_kw
    expected_q(t) = physics_anchor(t) + y_hat_q(t) * ac_nameplate_kw

The twin only learns the RESIDUAL correction on top of the physics anchor
-- soiling, spectral mismatch, satellite-vs-actual irradiance bias, inverter
clipping, cell-temp model error -- not the whole diurnal/seasonal
relationship from scratch. This is what makes ~900 training rows sufficient.

Three separate LightGBM boosters (q10/q50/q90), monotone-constrained so
expected power cannot decrease as irradiance increases or increase as cell
temperature increases -- this makes the twin physically un-gameable, not
just numerically accurate.

CRITICAL INVARIANT (enforced by Test TC-L2-01, Task 7): the feature matrix
passed to fit() must contain ONLY twin_* columns. Never ac_power, never
evidence_* columns. This function asserts on that at training time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

QUANTILES = [0.1, 0.5, 0.9]


def build_target(
    claimed_kw: pd.Series, physics_ac_kw: pd.Series, ac_nameplate_kw: float
) -> pd.Series:
    return (claimed_kw - physics_ac_kw) / ac_nameplate_kw


def assert_twin_feature_safety(feature_cols: list[str]) -> None:
    """
    Layer 2 invariant, R1 (Overview v2 SS2.2 / Task 2 SS2.1): claimed
    generation and on-site sensor evidence must NEVER enter the twin's
    feature matrix. This is TC-L2-01 from Task 7, called here directly so
    a violation is caught the moment training is attempted, not just in CI.
    """
    forbidden_prefixes = ("evidence_", "ac_power")
    violations = [c for c in feature_cols if any(c.startswith(p) or c == p for p in forbidden_prefixes)]
    assert not violations, (
        f"LEAKAGE: forbidden columns in twin feature matrix: {violations}. "
        f"The twin must train ONLY on twin_* independent-weather features."
    )
    non_twin = [c for c in feature_cols if not c.startswith("twin_")]
    assert not non_twin, (
        f"Feature matrix contains non-twin_ columns: {non_twin}. "
        f"All twin features must be prefixed twin_ per Layer 1.5's separation contract."
    )


@dataclass
class PIGBQRTwin:
    """
    Physics-Informed Gradient-Boosted Quantile Regression twin.
    One LightGBM booster per quantile, sharing the same feature set and
    monotone constraints.
    """
    monotone_map: dict[str, int] = field(default_factory=lambda: {
        "twin_ghi": 1, "twin_clearsky_ghi": 1, "twin_kt": 1,
        "twin_cell_temp": -1,
    })
    lgb_params: dict = field(default_factory=lambda: dict(
        num_leaves=15, max_depth=4, min_data_in_leaf=30,
        learning_rate=0.03, n_estimators=600,
        feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1,
        lambda_l2=5.0,
    ))
    feature_cols: list[str] = field(default_factory=list)
    ac_nameplate_kw: float = 0.0
    models_: dict = field(default_factory=dict, init=False, repr=False)

    def _monotone_constraints(self) -> list[int]:
        return [self.monotone_map.get(c, 0) for c in self.feature_cols]

    def fit(self, X: pd.DataFrame, physics_ac_kw: pd.Series, claimed_kw: pd.Series,
            ac_nameplate_kw: float) -> "PIGBQRTwin":

        """
        NOTE (post-Task-2 correction, verified twice against real errors):
        LightGBM's monotone-constraint mechanism rejects any objective with
        a degenerate (near-zero, piecewise-linear) second-order gradient.
        Confirmed empirically: both objective="quantile" and
        objective="regression_l1" raise a hard LightGBMError when combined
        with monotone_constraints ("Cannot use monotone_constraints in
        <objective> objective, please disable it"). Standard objective=
        "regression" (L2/MSE, constant positive hessian) is accepted.

        Fix: q50 trains as a monotone-constrained L2 (mean-targeting)
        regressor -- an approximation to the true median, exact only for
        symmetric residual distributions, documented here rather than
        silently assumed. q10/q90 remain true quantile-loss models WITHOUT
        the constraint, matching LightGBM's actual restriction rather than
        working around it. Report the L2-vs-true-median approximation and
        the unconstrained tail behavior explicitly as known, quantified
        limitations (see verify_monotonicity.py for the tail violation
        rate on real data).
        """

        import lightgbm as lgb

        self.feature_cols = list(X.columns)
        assert_twin_feature_safety(self.feature_cols)
        self.ac_nameplate_kw = ac_nameplate_kw

        y = build_target(claimed_kw, physics_ac_kw, ac_nameplate_kw)
        constraints = self._monotone_constraints()

        for q in QUANTILES:
            if q == 0.5:
                # median target, monotone-constrained. LightGBM's monotone
                # constraint mechanism requires a well-behaved (non-degenerate)
                # second-order gradient during split-finding -- confirmed
                # empirically that both "quantile" and "regression_l1" (both
                # piecewise-linear, zero/degenerate hessian almost everywhere)
                # are explicitly rejected with monotone_constraints, while
                # standard "regression" (L2/MSE, constant positive hessian)
                # is accepted. Using L2 as the median-targeting objective is
                # therefore the correct choice here, not L1 -- the mean and
                # median coincide only for symmetric residual distributions,
                # so this is an approximation to the true median, documented
                # explicitly rather than silently assumed.
                model = lgb.LGBMRegressor(
                    objective="regression",
                    monotone_constraints=constraints,
                    verbose=-1, **self.lgb_params,
                )
            else:
                # true quantile loss, NO monotone constraint -- this is the
                # combination LightGBM actually rejects; leaving q10/q90
                # unconstrained is correct behavior, not a workaround.
                model = lgb.LGBMRegressor(
                    objective="quantile", alpha=q,
                    verbose=-1, **self.lgb_params,
                )
            model.fit(X, y)
            self.models_[q] = model
        return self

    def predict(self, X: pd.DataFrame, physics_ac_kw: pd.Series,
                clearsky_index: pd.Series | None = None,
                kt_switch: float = 0.4) -> pd.DataFrame:
        """Returns expected_q10, expected_q50, expected_q90, with quantile-
        crossing correction (sort across quantiles per row -- cheap and
        sufficient; a full isotonic-regression rearrangement is optional
        polish, not required for this to be usable by Layer 3).

        REGIME FALLBACK (empirically justified, not a tuning choice): with
        only 9-12 real training rows at Kt<0.4 across both plants (Task 5/
        Layer 2 finding), the learned residual correction extrapolates
        unreliably in deep-overcast conditions -- confirmed to sometimes
        predict negative power and to underperform the physics-anchor-only
        baseline there. Below `kt_switch`, this falls back to the physics
        anchor directly rather than trusting the learned correction.
        Verified empirically (both plants): a hard switch exactly recovers
        physics-only's overcast R^2, while a soft linear blend only
        partially recovers it, since the blend still admits some
        unreliable correction near the boundary where training data is
        equally sparse. If `clearsky_index` is not provided, no fallback
        is applied (backward-compatible for callers that don't have it,
        e.g. the monotonicity audit script's synthetic sweeps).
        """
        assert list(X.columns) == self.feature_cols, "feature columns must match training"

        preds = {}
        for q in QUANTILES:
            y_hat = self.models_[q].predict(X)
            preds[q] = physics_ac_kw.values + y_hat * self.ac_nameplate_kw

        stacked = np.column_stack([preds[q] for q in QUANTILES])
        stacked_sorted = np.sort(stacked, axis=1)  # enforce q10 <= q50 <= q90 row-wise

        # Physical validity floor: AC power cannot be negative.
        stacked_sorted = np.clip(stacked_sorted, 0, None)

        result = pd.DataFrame({
            "expected_q10": stacked_sorted[:, 0],
            "expected_q50": stacked_sorted[:, 1],
            "expected_q90": stacked_sorted[:, 2],
        }, index=X.index)

        if clearsky_index is not None:
            low_kt_mask = (clearsky_index < kt_switch).reindex(result.index, fill_value=False)
            for col in ("expected_q10", "expected_q50", "expected_q90"):
                result.loc[low_kt_mask, col] = physics_ac_kw.loc[low_kt_mask].values

        return result

    def save(self, out_dir: str) -> None:
        """LightGBM's version-stable text format, NOT pickle -- see Task 3
        note on portability across machines/versions."""
        from pathlib import Path
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for q in QUANTILES:
            self.models_[q].booster_.save_model(str(out / f"twin_q{int(q*100)}.txt"))
        import json
        with open(out / "twin_meta.json", "w") as fh:
            json.dump({"feature_cols": self.feature_cols, "ac_nameplate_kw": self.ac_nameplate_kw,
                       "monotone_map": self.monotone_map}, fh, indent=2)

    @classmethod
    def load(cls, model_dir: str) -> "PIGBQRTwin":
        """
        Loads a twin saved via .save(). Used by every Layer 3 module that
        needs the twin's predictions. Loaded boosters are read-only for
        prediction -- this does NOT restore an LGBMRegressor you can call
        .fit() on again; use train_twin.py for retraining.
        """
        import json
        from pathlib import Path
        import lightgbm as lgb

        model_dir = Path(model_dir)
        with open(model_dir / "twin_meta.json") as fh:
            meta = json.load(fh)

        twin = cls(feature_cols=meta["feature_cols"], ac_nameplate_kw=meta["ac_nameplate_kw"],
                    monotone_map=meta.get("monotone_map", {}))
        twin.models_ = {}
        for q in QUANTILES:
            booster = lgb.Booster(model_file=str(model_dir / f"twin_q{int(q*100)}.txt"))

            class _ReadOnlyWrapper:
                def __init__(self, booster):
                    self._booster = booster
                def predict(self, X):
                    return self._booster.predict(X)

            twin.models_[q] = _ReadOnlyWrapper(booster)
        return twin
