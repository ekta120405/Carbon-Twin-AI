"""
dashboard/app.py

Layer 4.6 -- CarbonTwinAI audit dashboard entry point.

Landing page + shared sidebar (plant / data-source selection, persisted via
st.session_state across all five pages in dashboard/pages/). The actual five
views (Overview v2 SS7.6, docs/LAYER4_ROADMAP.md SS4.6) live as separate
Streamlit multipage scripts:
    1_Project_Risk_Overview.py    -- 3.6's score/band
    2_Expected_vs_Claimed.py      -- time series, twin band, flags overlaid
    3_Flag_Detail.py              -- 4.1 + 4.2 explanation panels, 4.3 carbon
    4_Carbon_Exposure.py          -- 4.3 project/period carbon+financial summary
    5_Evaluation.py               -- placeholder pending Task 7's full benchmark

Run with:  streamlit run dashboard/app.py   (from the repo root, .venv active)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from dashboard.lib.pipeline import get_scored_context, load_plant_base
from dashboard.lib.ui import sidebar_controls, render_caveats, CARRYOVER_CAVEATS, band_color

st.set_page_config(page_title="CarbonTwinAI", page_icon="🌞", layout="wide")

plant, source_key = sidebar_controls()

st.title("CarbonTwinAI -- Audit Dashboard")
st.caption(
    "An audit-grade digital twin for continuous verification of renewable "
    "energy carbon credit claims. Layers 1-3 detect and quantify energy-"
    "domain fraud; this Layer 4 dashboard explains, prices, and presents "
    "what Layer 3 already found -- it does not re-detect anything."
)

with st.spinner("Loading plant model and scoring selected data source..."):
    base = load_plant_base(plant)
    ctx = get_scored_context(plant, source_key)

if ctx["warnings"]:
    for w in ctx["warnings"]:
        st.warning(w)

col1, col2, col3, col4 = st.columns(4)
risk = ctx["risk_result"]
with col1:
    st.metric("Risk score", f"{risk.score:.1f} / 100")
with col2:
    st.markdown(
        f"<div style='padding:0.5rem 0'><span style='background-color:{band_color(risk.band)};"
        f"color:white;padding:0.3rem 0.8rem;border-radius:0.4rem;font-weight:600'>"
        f"{risk.band}</span></div>",
        unsafe_allow_html=True,
    )
    st.caption("Risk band")
with col3:
    n_flagged = int(ctx["labels"].str.startswith("flagged").sum())
    st.metric("Flagged days", f"{n_flagged} / {len(ctx['labels'])}")
with col4:
    st.metric("Value at risk", f"${ctx['carbon_agg']['total_value_at_risk_usd']:,.2f}")

st.divider()

st.markdown(
    """
### Navigate the five views (sidebar, top-left)

1. **Project Risk Overview** -- 3.6's aggregate 0-100 risk score and its five components
2. **Expected vs. Claimed** -- 15-minute time series, twin uncertainty band, flags overlaid
3. **Flag Detail** -- per-flag Twin-SHAP + Flag-SHAP explanations and carbon exposure
4. **Carbon Exposure** -- project/period tCO2e and $ value at risk, with mandatory caveats
5. **Evaluation** -- informal sanity check against injected ground truth (placeholder pending the full Task 7 benchmark)

All five views share the plant and data-source selection in the sidebar.
"""
)

render_caveats(CARRYOVER_CAVEATS, title="Carryover caveats from Layers 2/3 (read before interpreting any number)")
