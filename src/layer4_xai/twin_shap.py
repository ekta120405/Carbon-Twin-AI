"""
src/layer4_xai/twin_shap.py

Layer 4.1: Twin-SHAP -- "why did we expect this much?" (Overview v2 Module
4.1; docs/LAYER4_ROADMAP.md SS4.1).

TWO GOTCHAS FOUND AND VERIFIED WHILE BUILDING THIS (neither is in the
roadmap's sketch -- confirmed empirically against the real repo before
writing this module, not assumed):

1. UNITS: PIGBQRTwin's raw LightGBM boosters predict `y_hat`, the
   RESIDUAL-CORRECTION TARGET as a FRACTION of ac_nameplate_kw (see
   src/layer2_twin/twin_model.py build_target():
   y = (claimed - physics_anchor) / ac_nameplate_kw). Confirmed by direct
   reconstruction: `explainer.shap_values(X).sum(axis=1) + explainer.
   expected_value == raw_booster.predict(X)` exactly, and that quantity is
   NOT expected_q50 in kW -- it's the small residual fraction (e.g.
   -0.008), not the ~kW-scale power value. SHAP values must be multiplied
   by ac_nameplate_kw to be meaningful as kW contributions, and even then
   they explain only the twin's LEARNED CORRECTION, not the physics
   anchor itself (physics_ac_kw is deterministic pvlib/PVWatts arithmetic,
   not a model output -- there is nothing for SHAP to attribute there).
   Report language must say "the twin's correction on top of the physics
   anchor," never "why we expected X kWh" unqualified, or it implies SHAP
   explains the whole prediction when it only explains one additive term.

2. LOW-Kt FALLBACK: twin.predict()'s kt_switch=0.4 fallback (Layer 2/3,
   already documented) means the booster's learned correction is DISCARDED
   ENTIRELY for clearsky_index < kt_switch -- Layer 3 uses physics_ac_kw
   directly there. Confirmed on a real plant_1 row (twin_kt=0.357):
   the fallback-corrected expected_q50 was 2229.9 kW; the raw
   booster-driven value (as if the fallback didn't exist) was 1318.6 kW --
   a ~900 kW difference on a single timestep. Running TreeSHAP on a
   low-Kt row's raw booster output would confidently explain a number
   Layer 3 never actually used. This module checks twin_kt against
   kt_switch BEFORE attempting any SHAP call and returns a rule-based
   fallback explanation instead for those rows -- the same "don't force
   statistical attribution onto a deterministic path" discipline the
   roadmap already applies to the A3/A8 guardrails (SS2.5), just applied
   here to a different deterministic path the roadmap didn't call out.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import shap

from src.layer2_twin.twin_model import PIGBQRTwin

DEFAULT_KT_SWITCH = 0.4  # MUST match twin.predict()'s kt_switch and
# config.yaml's layer3.conformal.csi_bins[1] -- see residual.py's
# build_residual_table docstring. Do not change independently.


@dataclass
class TimestampExplanation:
    timestamp: pd.Timestamp
    mode: str  # "shap" or "physics_anchor_fallback"
    physics_ac_kw: float
    twin_correction_kw: float | None       # None if fallback mode
    expected_q50_kw: float                 # physics_ac_kw + twin_correction_kw (or == physics_ac_kw in fallback)
    twin_kt: float
    top_features_kw: list[tuple[str, float]] = field(default_factory=list)  # [(feature, contribution_kw), ...] sorted by |contribution|, empty in fallback mode
    note: str = ""


def _unwrap_booster(twin: PIGBQRTwin, quantile: float = 0.5):
    """
    twin.models_[q] is a _ReadOnlyWrapper when loaded via PIGBQRTwin.load()
    (see docs/LAYER4_ROADMAP.md SS2.3) -- shap.TreeExplainer needs the raw
    lightgbm.basic.Booster, not the wrapper. Verified attribute name
    `._booster` against the real twin_model.py source, not assumed.
    """
    model = twin.models_[quantile]
    if hasattr(model, "_booster"):
        return model._booster
    # in-memory (freshly-trained, not loaded) case: sklearn wrapper exposes booster_
    if hasattr(model, "booster_"):
        return model.booster_
    raise AttributeError(
        f"Could not find a raw Booster on twin.models_[{quantile}] "
        f"(type={type(model)}). Expected ._booster (loaded) or .booster_ "
        f"(freshly trained). Re-verify against twin_model.py if this fires -- "
        f"it means the wrapper's interface changed."
    )


class TwinShapExplainer:
    """
    Wraps a TreeExplainer built once per plant (on q50's raw booster) and
    reused across many rows -- TreeSHAP is fast per-call, but there's no
    reason to reconstruct the explainer for every timestamp.
    """

    def __init__(self, twin: PIGBQRTwin, kt_switch: float = DEFAULT_KT_SWITCH):
        self.twin = twin
        self.kt_switch = kt_switch
        raw_booster = _unwrap_booster(twin, 0.5)
        self.explainer = shap.TreeExplainer(raw_booster)
        self.feature_cols = twin.feature_cols
        self.ac_nameplate_kw = twin.ac_nameplate_kw

    def explain_timestamp(
        self,
        x_row: pd.Series,
        physics_ac_kw: float,
        twin_kt: float,
        top_k: int = 5,
    ) -> TimestampExplanation:
        """
        Explains ONE timestamp. `x_row` must contain exactly
        self.feature_cols (same contract as twin.predict()).
        """
        if twin_kt < self.kt_switch:
            # Fallback path -- see module docstring, gotcha 2. Nothing to
            # attribute: the learned correction was never applied here.
            return TimestampExplanation(
                timestamp=x_row.name,
                mode="physics_anchor_fallback",
                physics_ac_kw=float(physics_ac_kw),
                twin_correction_kw=None,
                expected_q50_kw=float(physics_ac_kw),
                twin_kt=float(twin_kt),
                top_features_kw=[],
                note=(
                    f"twin_kt={twin_kt:.3f} < kt_switch={self.kt_switch}: the "
                    f"twin's learned correction is not used here (Layer 2/3 "
                    f"low-Kt fallback, see docs/LAYER2_HANDOFF.md SS5 / "
                    f"twin_model.py predict()). expected_q50 is the physics "
                    f"anchor directly -- no SHAP attribution applies because "
                    f"no learned model output was used for this timestamp."
                ),
            )

        X = pd.DataFrame(
            [x_row[self.feature_cols].to_numpy(dtype=float)],
            columns=self.feature_cols,
            index=[x_row.name],
        )
        shap_row = self.explainer.shap_values(X)[0]  # (n_features,), in y_hat (fraction) units
        base_value = self.explainer.expected_value

        # Convert fraction-of-capacity units to kW -- see module docstring,
        # gotcha 1. This is a linear rescaling of an additive decomposition,
        # so it stays a valid decomposition after scaling.
        shap_kw = shap_row * self.ac_nameplate_kw
        base_kw = base_value * self.ac_nameplate_kw
        twin_correction_kw = float(shap_kw.sum() + base_kw)
        expected_q50_kw = float(physics_ac_kw) + twin_correction_kw

        ranked = sorted(
            zip(self.feature_cols, shap_kw), key=lambda kv: abs(kv[1]), reverse=True
        )[:top_k]

        return TimestampExplanation(
            timestamp=x_row.name,
            mode="shap",
            physics_ac_kw=float(physics_ac_kw),
            twin_correction_kw=twin_correction_kw,
            expected_q50_kw=expected_q50_kw,
            twin_kt=float(twin_kt),
            top_features_kw=[(f, float(v)) for f, v in ranked],
            note=(
                "SHAP explains the twin's learned correction on top of the "
                "physics anchor, not the total expected_q50 -- the physics "
                "anchor's own value is deterministic pvlib/PVWatts arithmetic, "
                "not a model output, and carries no SHAP attribution."
            ),
        )


def pick_representative_timestamp(day_ts_table: pd.DataFrame) -> pd.Timestamp:
    """
    For a flagged DAY, pick the single daylight timestamp with the largest
    |residual| as the representative instance to explain -- explaining
    every 15-min slot of a flagged day is mostly redundant (weather doesn't
    change fast enough to give distinct stories) and the roadmap's scope
    control (SS4.1: "only compute for flagged periods... no reason to
    explain thousands of clean timesteps") already establishes the
    principle of not over-explaining; this applies the same principle
    within a single flagged day. Restricted to is_daylight rows since
    residual is only meaningful there (residual.py).
    """
    daylight = day_ts_table[day_ts_table["is_daylight"].astype(bool)]
    if len(daylight) == 0:
        raise ValueError("No daylight rows in this day's table -- cannot pick a representative timestamp.")
    idx = daylight["residual"].abs().idxmax()
    return idx


def explain_flagged_days(
    twin: PIGBQRTwin,
    ts_table: pd.DataFrame,
    labels: pd.Series,
    kt_switch: float = DEFAULT_KT_SWITCH,
    max_flags_explained: int = 500,
    top_k: int = 5,
) -> dict[object, TimestampExplanation]:
    """
    Explains one representative timestamp per flagged day, for all flag
    types (the physical "what should have happened" context is useful
    regardless of WHY a day was flagged -- that's 4.2's separate job).
    `ts_table` must have expected_q10/50/90, physics_ac_kw, twin_kt,
    residual, is_daylight columns (residual.build_residual_table()'s
    output). `labels` is conformal.classify_days()'s output.

    Scope control per config.yaml layer4.shap.explain_flagged_only /
    max_flags_explained: only flagged days, capped, per roadmap SS4.1 --
    TreeSHAP is cheap per-call but there's no reason to explain clean days
    nobody will look at.
    """
    explainer = TwinShapExplainer(twin, kt_switch=kt_switch)
    flagged_dates = labels[labels.str.startswith("flagged")].index[:max_flags_explained]

    ts_dates = pd.Series(ts_table.index.date, index=ts_table.index)
    results = {}
    for date in flagged_dates:
        day_table = ts_table[ts_dates == date]
        if len(day_table) == 0:
            continue
        rep_ts = pick_representative_timestamp(day_table)
        row = day_table.loc[rep_ts]
        expl = explainer.explain_timestamp(
            row, physics_ac_kw=row["physics_ac_kw"], twin_kt=row["twin_kt"], top_k=top_k,
        )
        results[date] = expl
    return results
