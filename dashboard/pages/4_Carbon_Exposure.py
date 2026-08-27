"""
View 4 -- carbon and financial exposure summary (4.3, src/layer4_xai/
carbon.py). Displays project/period totals from aggregate_carbon_exposure()
and the per-day carbon table from build_carbon_table() -- both mandatory
caveats (vintage mismatch, price uncertainty) are always rendered on this
page per carbon.py's REQUIRED_CAVEATS, not just computed silently.
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

from src.layer4_xai.carbon import REQUIRED_CAVEATS
from dashboard.lib.pipeline import get_scored_context, load_plant_base, GRID_EMISSION_FACTOR, CARBON_PRICE, FLAG_COLORS, FLAG_LABELS
from dashboard.lib.ui import sidebar_controls, render_caveats, fmt_kwh, fmt_usd, fmt_tco2e

st.set_page_config(page_title="Carbon Exposure -- CarbonTwinAI", page_icon="🌞", layout="wide")

plant, source_key = sidebar_controls()
st.title("4. Carbon & Financial Exposure")

base = load_plant_base(plant)
ctx = get_scored_context(plant, source_key)
carbon_table = ctx["carbon_table"]
agg = ctx["carbon_agg"]

render_caveats(list(REQUIRED_CAVEATS), title="Mandatory caveats for every number on this page", expanded=True)

st.markdown(
    f"Configured grid emission factor: **{GRID_EMISSION_FACTOR} tCO2e/MWh** "
    f"(CEA CO2 Baseline Database v21.0, Combined Margin, FY2024-25) &nbsp;|&nbsp; "
    f"Configured carbon price: **${CARBON_PRICE}/tCO2e**"
)

c1, c2, c3 = st.columns(3)
c1.metric("Flagged days", agg["n_flagged_days"])
c2.metric("Total carbon at risk", fmt_tco2e(agg["total_excess_tco2e"]))
c3.metric("Total value at risk", fmt_usd(agg["total_value_at_risk_usd"]))

st.subheader("Exposure by flag type")
if agg["by_flag_type"]:
    rows = []
    for ft, vals in agg["by_flag_type"].items():
        rows.append({"flag_type": FLAG_LABELS.get(ft, ft), "excess_tco2e": vals["excess_tco2e"],
                     "value_at_risk_usd": vals["value_at_risk_usd"]})
    by_type_df = pd.DataFrame(rows)
    fig = go.Figure(go.Bar(
        x=by_type_df["flag_type"], y=by_type_df["value_at_risk_usd"],
        marker_color=[FLAG_COLORS.get(ft, "#888") for ft in agg["by_flag_type"].keys()],
        text=by_type_df["value_at_risk_usd"].round(2), textposition="outside",
    ))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Value at risk (USD)")
    st.plotly_chart(fig, width='stretch')
    st.dataframe(by_type_df, width='stretch', hide_index=True)
else:
    st.info("No flagged days in this data source -- $0 exposure. "
            "(This is the expected result on genuinely clean real data.)")

st.subheader("Per-day exposure")
display_table = carbon_table.copy()
display_table["value_at_risk_usd"] = display_table["value_at_risk_usd"].round(2)
display_table["excess_tco2e"] = display_table["excess_tco2e"].round(4)
display_table["severity_kwh"] = display_table["severity_kwh"].round(1)
st.dataframe(display_table, width='stretch')

flagged_only = carbon_table[carbon_table["flag_type"].str.startswith("flagged")]
if len(flagged_only):
    fig2 = go.Figure(go.Bar(
        x=[str(d) for d in flagged_only.index], y=flagged_only["value_at_risk_usd"],
        marker_color=[FLAG_COLORS.get(ft, "#888") for ft in flagged_only["flag_type"]],
    ))
    fig2.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                        yaxis_title="Value at risk (USD)", xaxis_title="Date")
    st.plotly_chart(fig2, width='stretch')
