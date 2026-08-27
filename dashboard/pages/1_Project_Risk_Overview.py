"""
View 1 -- Project risk overview. Consumes 3.6's RiskScoreResult directly
(src/layer3_anomaly/aggregate.py:compute_risk_score) -- no new computation
happens in this file, per docs/LAYER4_MODULES_1_TO_5_HANDOFF.md SS7.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.lib.pipeline import get_scored_context, load_plant_base
from dashboard.lib.ui import sidebar_controls, render_caveats, CARRYOVER_CAVEATS, band_color, fmt_usd, fmt_tco2e

st.set_page_config(page_title="Project Risk Overview -- CarbonTwinAI", page_icon="🌞", layout="wide")

plant, source_key = sidebar_controls()
st.title("1. Project Risk Overview")

base = load_plant_base(plant)
ctx = get_scored_context(plant, source_key)
risk = ctx["risk_result"]

for w in ctx["warnings"]:
    st.warning(w)

left, right = st.columns([1, 1.4])

with left:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk.score,
        number={"suffix": " / 100"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": band_color(risk.band)},
            "steps": [
                {"range": [0, 25], "color": "#eafaea"},
                {"range": [25, 50], "color": "#faf7d6"},
                {"range": [50, 75], "color": "#fde3cc"},
                {"range": [75, 100], "color": "#fbdada"},
            ],
        },
        title={"text": f"{plant} -- {risk.band} risk"},
    ))
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=60, b=10))
    st.plotly_chart(fig, width='stretch')

with right:
    st.subheader("Five-component breakdown")
    st.caption(
        "Each of Layer 3.6's five inputs is normalized to [0,1] before "
        "weighting -- weights and caps are documented judgment calls, not "
        "empirically optimized (docs/LAYER3_HANDOFF.md SS7)."
    )
    weights = {"frac_flagged": 0.30, "severity": 0.30, "cusum": 0.20,
               "calibration_ratio": 0.12, "divergence": 0.08}
    comp_df = pd.DataFrame({
        "component": list(risk.components.keys()),
        "normalized_value": list(risk.components.values()),
        "weight": [weights[k] for k in risk.components.keys()],
    })
    comp_df["contribution_to_score"] = comp_df["normalized_value"] * comp_df["weight"] * 100
    fig2 = go.Figure(go.Bar(
        x=comp_df["contribution_to_score"], y=comp_df["component"], orientation="h",
        marker_color="#1f77b4", text=comp_df["contribution_to_score"].round(1), textposition="outside",
    ))
    fig2.update_layout(height=320, margin=dict(l=20, r=20, t=10, b=10),
                        xaxis_title="Contribution to score (points, out of 100)")
    st.plotly_chart(fig2, width='stretch')

st.divider()

c1, c2, c3, c4 = st.columns(4)
n_flagged = int(ctx["labels"].str.startswith("flagged").sum())
n_total = len(ctx["labels"])
c1.metric("Days flagged", f"{n_flagged} / {n_total}")
c2.metric("Cumulative severity", f"{ctx['cum_severity_frac']*100:.2f}% of claimed energy")
c3.metric("Carbon at risk", fmt_tco2e(ctx["carbon_agg"]["total_excess_tco2e"]))
c4.metric("Value at risk", fmt_usd(ctx["carbon_agg"]["total_value_at_risk_usd"]))

st.subheader("Flag mix")
counts = ctx["labels"].value_counts()
if len(counts):
    from dashboard.lib.pipeline import FLAG_COLORS, FLAG_LABELS
    order = [k for k in FLAG_LABELS if k in counts.index]
    fig3 = go.Figure(go.Bar(
        x=[FLAG_LABELS[k] for k in order], y=[int(counts[k]) for k in order],
        marker_color=[FLAG_COLORS[k] for k in order],
    ))
    fig3.update_layout(height=280, margin=dict(l=20, r=20, t=10, b=10), yaxis_title="days")
    st.plotly_chart(fig3, width='stretch')
else:
    st.info("No days in the selected data source.")

st.subheader("Raw score inputs")
st.json({
    "frac_flagged": ctx["frac_flagged"],
    "cum_severity_frac": ctx["cum_severity_frac"],
    "cusum_max_s": max(ctx["cusum_result"].s_values) if ctx["cusum_result"].s_values else 0.0,
    "cusum_h (recalibrated for this window length)": ctx["cusum_h"],
    "calibration_ratio": ctx["calibration_ratio"],
    "calibration_ratio_clean_baseline": base["clean_cr_baseline"],
    "divergence_flagged_only": ctx["divergence"],
    "achievable_conformal_alpha": base["conformal_result"].alpha_achievable,
    "conformal_threshold_r_day": base["conformal_result"].threshold,
})

render_caveats(CARRYOVER_CAVEATS, title="Read before interpreting this score", expanded=True)
