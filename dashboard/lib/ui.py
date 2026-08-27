"""
dashboard/lib/ui.py

Shared sidebar controls, caveat text, and small formatting helpers reused
across all five dashboard pages. Caveat strings mirror docs/LAYER4_MODULES_
1_TO_5_HANDOFF.md SS6 ("carryover caveats that surface through the Layer 4
UI") -- these MUST be visible on the relevant views, not just in a methods
appendix, per that document's own instruction to whoever builds 4.6.
"""
from __future__ import annotations

import streamlit as st

from dashboard.lib.pipeline import PLANTS, source_options, FLAG_COLORS, FLAG_LABELS, BAND_COLORS

ACHIEVABLE_ALPHA_CAVEAT = (
    "The conformal calibration set is only 5 days per plant -- below the "
    "sample-size floor for the nominal 5% false-positive rate. The "
    "**actually achievable** false-positive rate is ~16.7% (1/(n_calib+1)), "
    "not 5%. Treat flags as calibrated suspicion at that rate, not a 5% "
    "guarantee."
)

Q50_CAVEAT = (
    "The twin's central estimate (`expected_q50`) is a monotone-constrained "
    "**conditional mean**, not a true median -- LightGBM's monotone-"
    "constraint mechanism rejects any piecewise-linear-gradient objective "
    "(pinball/MAE), so q50 is trained with an L2 objective instead. Read "
    "'expected_q50' as 'the twin's central estimate', never 'the median "
    "forecast'."
)

PLANT2_TWIN_CAVEAT = (
    "**plant_2's twin underperforms the physics-only anchor** in clear and "
    "partly-cloudy regimes (confirmed, unresolved overfitting -- see "
    "docs/LAYER2_HANDOFF.md SS4). A 'the twin beats physics-only' claim is "
    "false for plant_2 specifically."
)

SMALL_N_CAVEAT = (
    "This dataset spans only 34 real days per plant. Any regime-level or "
    "small-window statistic shown here should be read alongside its sample "
    "size (n) -- small-n statistics in this project have previously shown "
    "wildly unstable values (e.g. an R^2 of -61.6 on n=7)."
)

STUCK_SENSOR_UI_CAVEAT = (
    "The stuck-sensor guardrail also fires on legitimate grid-ordered "
    "curtailment (N3) when the cap binds for >=8 consecutive slots. A fix "
    "was attempted and reverted after it cut real detection coverage "
    "roughly in half. **Treat this flag type as suspected, not confirmed, "
    "fraud.**"
)

CARRYOVER_CAVEATS = [ACHIEVABLE_ALPHA_CAVEAT, Q50_CAVEAT, PLANT2_TWIN_CAVEAT, SMALL_N_CAVEAT]


def sidebar_controls() -> tuple[str, str]:
    """Renders the plant + data-source selectors. Uses session_state keys so
    the selection persists as the user navigates between pages."""
    st.sidebar.title("CarbonTwinAI")
    st.sidebar.caption("Audit-grade digital twin -- Layer 4 explainability & carbon dashboard")

    plant = st.sidebar.selectbox(
        "Plant", PLANTS, key="plant",
        format_func=lambda p: p.replace("_", " ").title(),
    )
    opts = source_options(plant)
    keys = [k for k, _ in opts]
    labels = {k: v for k, v in opts}
    if st.session_state.get("source_key") not in keys:
        st.session_state["source_key"] = keys[0]
    source_key = st.sidebar.selectbox(
        "Data source", keys, key="source_key", format_func=lambda k: labels[k],
    )
    st.sidebar.divider()
    st.sidebar.caption(
        "Injected replicates are synthetic anomaly injections from the "
        "frozen injector (injector-v1.0), used for bounded sanity checks -- "
        "never for tuning detector logic. 'Real' sources are the plant's "
        "actual claimed generation."
    )
    return plant, source_key


def render_caveats(caveats: list[str], title: str = "Known caveats", expanded: bool = False) -> None:
    if not caveats:
        return
    with st.expander(f"{title} ({len(caveats)})", expanded=expanded):
        for c in caveats:
            st.markdown(f"- {c}")


def band_color(band: str) -> str:
    return BAND_COLORS.get(band, "#7f7f7f")


def flag_color(flag_type: str) -> str:
    return FLAG_COLORS.get(flag_type, "#7f7f7f")


def flag_label(flag_type: str) -> str:
    return FLAG_LABELS.get(flag_type, flag_type)


def fmt_kwh(x: float) -> str:
    return f"{x:,.1f} kWh"


def fmt_usd(x: float) -> str:
    return f"${x:,.2f}"


def fmt_tco2e(x: float) -> str:
    return f"{x:,.3f} tCO2e"
