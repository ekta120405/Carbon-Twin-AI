"""
View 3 -- per-flag detail: 4.1 (Twin-SHAP) + 4.2 (Flag-SHAP) explanation
panels, plus 4.3's carbon exposure and 4.5's severity/confidence ranking.
Consumes 4.4's evidence bundles directly (src/layer4_xai/evidence.py) --
no recomputation happens in this file.

Branches on physical_explanation.mode and suspicion_explanation.mode per
docs/LAYER4_MODULES_1_TO_5_HANDOFF.md SS1/SS2 -- a "shap" physical mode
gets a feature-contribution chart, "physics_anchor_fallback" gets the note
text only (no misleading empty chart); a "statistical" suspicion mode gets
a z-score chart, "guardrail" gets the deterministic template text + caveat.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import plotly.graph_objects as go
import streamlit as st

from dashboard.lib.pipeline import get_scored_context, load_plant_base
from dashboard.lib.ui import (
    sidebar_controls, render_caveats, flag_color, flag_label, fmt_kwh, fmt_usd, fmt_tco2e,
)

st.set_page_config(page_title="Flag Detail -- CarbonTwinAI", page_icon="🌞", layout="wide")

plant, source_key = sidebar_controls()
st.title("3. Flag Detail")

base = load_plant_base(plant)
ctx = get_scored_context(plant, source_key)
ranked = ctx["ranked"]
bundles_by_date = ctx["bundles_by_date"]

if len(ranked) == 0:
    st.info("No flagged days in this data source -- nothing to explain. "
            "Try an injected replicate from the sidebar.")
    st.stop()

st.caption(
    "Flags are listed most-valuable-at-risk first (4.5's severity ranking). "
    "`confidence_ratio` is mechanism-specific -- 'fractional excess past "
    "this flag's own trigger', not a unified statistic comparable across "
    "flag types (docs/LAYER4_MODULES_1_TO_5_HANDOFF.md SS5)."
)

option_labels = {
    str(d): f"{d} -- {flag_label(ranked.loc[d, 'flag_type'])} -- {fmt_usd(ranked.loc[d, 'value_at_risk_usd'])}"
    for d in ranked.index
}
options = list(option_labels.keys())
choice = st.selectbox("Select a flagged day", options, format_func=lambda s: option_labels[s])
sel_date = ranked.index[options.index(choice)]
row = ranked.loc[sel_date]
bundle = bundles_by_date.get(str(sel_date))

if bundle is None:
    from dashboard.lib.pipeline import MAX_FLAGS_EXPLAINED
    st.error(
        f"No evidence bundle found for {sel_date} (capped by "
        f"layer4.shap.max_flags_explained={MAX_FLAGS_EXPLAINED}). This "
        f"should not normally happen for a flagged day within the cap."
    )
    st.stop()

st.subheader(f"{sel_date} -- {flag_label(bundle['flag_type'])}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Severity", fmt_kwh(bundle["severity_kwh"]))
c2.metric("Carbon at risk", fmt_tco2e(bundle["carbon_exposure"]["excess_tco2e"]))
c3.metric("Value at risk", fmt_usd(bundle["carbon_exposure"]["value_at_risk_usd"]))
conf = row["confidence_ratio"]
c4.metric("Confidence ratio", f"{conf:.2f}" if conf is not None else "n/a",
          help="Fractional excess past this flag's own trigger condition; "
               "mechanism-specific, not a p-value (see caption above).")

render_caveats(list(bundle["known_caveats"]), title="Caveats for this flag", expanded=True)

st.divider()
left, right = st.columns(2)

with left:
    st.markdown("#### Physical explanation -- \"why did we expect this much?\" (4.1 Twin-SHAP)")
    phys = bundle["physical_explanation"]
    if phys["mode"] == "shap":
        st.markdown(
            f"Physics anchor: **{phys['physics_ac_kw']:.1f} kW** &nbsp;+&nbsp; "
            f"Twin's learned correction: **{phys['twin_correction_kw']:+.1f} kW** "
            f"&nbsp;=&nbsp; Expected (q50): **{phys['expected_q50_kw']:.1f} kW**"
        )
        feats = phys["top_features_kw"]
        if feats:
            names = [f[0] for f in feats][::-1]
            vals = [f[1] for f in feats][::-1]
            colors = ["#2ca02c" if v >= 0 else "#d62728" for v in vals]
            fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=colors))
            fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                               xaxis_title="Contribution to twin's correction (kW)")
            st.plotly_chart(fig, width='stretch')
        st.caption(
            "SHAP explains only the twin's learned correction ON TOP OF the "
            "physics anchor -- never the physics anchor itself, which is "
            "deterministic pvlib/PVWatts arithmetic with no model behind it."
        )
    else:
        st.info(phys["note"])
        st.markdown(f"Expected output used by Layer 3: **{phys['expected_q50_kw']:.1f} kW** "
                    f"(physics anchor directly, twin_kt={phys['twin_kt']:.3f})")

with right:
    st.markdown("#### Suspicion explanation -- \"why is this flagged?\" (4.2 Flag-SHAP)")
    susp = bundle["suspicion_explanation"]
    if susp["mode"] == "statistical":
        drivers = susp["top_drivers"]
        names = [d[0] for d in drivers][::-1]
        zscores = [d[1] for d in drivers][::-1]
        colors = ["#d62728" if abs(z) >= 2 else "#ff7f0e" if abs(z) >= 1 else "#7f7f7f" for z in zscores]
        fig2 = go.Figure(go.Bar(x=zscores, y=names, orientation="h", marker_color=colors))
        fig2.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10),
                            xaxis_title=f"z-score vs. clean reference (n={susp['reference_n']})")
        st.plotly_chart(fig2, width='stretch')
        st.dataframe(
            {"raw value": {k: round(v, 4) for k, v in susp["raw_values"].items()}},
            width='stretch',
        )
        if any(d[0] == "yield_jump_frac" and abs(d[1]) == max(abs(x[1]) for x in drivers) for d in drivers):
            st.caption(
                "Note: `yield_jump_frac` is the top driver in most overclaim "
                "flags but is partially circular -- an overclaim day almost "
                "definitionally differs from the day before. "
                "`irr_divergence_pct_mean` is the independent on-site-vs-"
                "satellite check."
            )
    else:
        st.warning(susp["text"])
        st.json(susp["trigger_values"])

render_caveats(
    [
        "Vintage mismatch: the grid emission factor is dated FY2024-25 (Nov "
        "2025 edition); the dataset's generation is from 2020 -- a stated "
        "methodological simplification.",
        "Price uncertainty: the voluntary carbon market is opaque with wide "
        "price dispersion ($0.25-$27/tCO2e observed); $5.00/tCO2e is a "
        "representative low-end estimate, not a quoted market rate.",
        "Terminology: figures are 'at risk', never 'confirmed fraud' -- "
        "Layer 3's flags are calibrated statistical suspicion (achievable "
        "alpha ~16.7%), not adjudicated fact.",
    ],
    title="Carbon accounting caveats (mandatory)", expanded=True,
)
