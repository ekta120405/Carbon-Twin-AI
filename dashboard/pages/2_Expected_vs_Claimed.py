"""
View 2 -- expected-vs-claimed time series, twin uncertainty band, and flags
overlaid (Layers 3.1-3.3). Honest labeling note: the shaded band is the
twin's own q10-q90 quantile band, NOT a per-timestep conformal interval --
Layer 3's conformal calibration operates on the DAILY r_day aggregate, not
per-15-min predictions (docs/LAYER3_HANDOFF.md SS4). The daily panel below
shows the actual conformal-calibrated quantity (r_day vs. threshold), which
is where the real statistical guarantee lives.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from dashboard.lib.pipeline import get_scored_context, load_plant_base, FLAG_COLORS, FLAG_LABELS
from dashboard.lib.ui import sidebar_controls, render_caveats, Q50_CAVEAT

st.set_page_config(page_title="Expected vs Claimed -- CarbonTwinAI", page_icon="🌞", layout="wide")

plant, source_key = sidebar_controls()
st.title("2. Expected vs. Claimed Generation")

base = load_plant_base(plant)
ctx = get_scored_context(plant, source_key)
ts = ctx["ts_table"]
daily = ctx["daily"]
labels = ctx["labels"]
threshold = base["conformal_result"].threshold

st.caption(
    "Shaded band = twin's q10-q90 uncertainty (NOT a per-timestep conformal "
    "interval -- conformal calibration is applied at the DAILY level; see "
    "the lower panel). Vertical bands mark flagged days."
)

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=False, row_heights=[0.62, 0.38],
    subplot_titles=("15-minute power (kW)", "Daily normalised residual r_day vs. conformal threshold"),
    vertical_spacing=0.12,
)

fig.add_trace(go.Scatter(
    x=ts.index, y=ts["expected_q90"], mode="lines", line=dict(width=0),
    showlegend=False, hoverinfo="skip",
), row=1, col=1)
fig.add_trace(go.Scatter(
    x=ts.index, y=ts["expected_q10"], mode="lines", line=dict(width=0),
    fill="tonexty", fillcolor="rgba(31,119,180,0.15)", name="Twin q10-q90 band",
), row=1, col=1)
fig.add_trace(go.Scatter(
    x=ts.index, y=ts["expected_q50"], mode="lines", line=dict(color="#1f77b4", width=1.5),
    name="Twin expected (q50, central estimate)",
), row=1, col=1)
fig.add_trace(go.Scatter(
    x=ts.index, y=ts["physics_ac_kw"], mode="lines", line=dict(color="#2ca02c", width=1, dash="dot"),
    name="Physics anchor (deterministic)",
), row=1, col=1)
fig.add_trace(go.Scatter(
    x=ts.index, y=ts["ac_power"], mode="lines", line=dict(color="#d62728", width=1.5),
    name="Claimed (ac_power)",
), row=1, col=1)

for date, flag_type in labels.items():
    if not str(flag_type).startswith("flagged"):
        continue
    start = pd.Timestamp(date, tz=ts.index.tz)
    end = start + pd.Timedelta(days=1)
    fig.add_vrect(x0=start, x1=end, fillcolor=FLAG_COLORS.get(flag_type, "#888"),
                  opacity=0.15, line_width=0, row=1, col=1)

daily_sorted = daily.sort_index()
bar_colors = [FLAG_COLORS.get(labels.get(d, "normal"), "#888") for d in daily_sorted.index]
fig.add_trace(go.Bar(
    x=[pd.Timestamp(d) for d in daily_sorted.index], y=daily_sorted["r_day"],
    marker_color=bar_colors, name="r_day (daily)", showlegend=False,
), row=2, col=1)
fig.add_hline(y=threshold, line=dict(color="black", dash="dash"), row=2, col=1,
              annotation_text=f"conformal threshold ({threshold:.4f})")

fig.update_layout(height=680, legend=dict(orientation="h", y=1.08), margin=dict(t=60, b=10))
fig.update_yaxes(title_text="kW", row=1, col=1)
fig.update_yaxes(title_text="r_day (capacity-normalised)", row=2, col=1)
st.plotly_chart(fig, width='stretch')

st.subheader("Legend -- flag colors")
cols = st.columns(len(FLAG_LABELS))
for c, (k, v) in zip(cols, FLAG_LABELS.items()):
    c.markdown(
        f"<span style='background-color:{FLAG_COLORS[k]};color:white;padding:2px 8px;"
        f"border-radius:4px;font-size:0.8rem'>{v}</span>", unsafe_allow_html=True,
    )

render_caveats([Q50_CAVEAT], title="Read before interpreting the central-estimate line")
