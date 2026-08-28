"""
dashboard/lib/ui.py

Plain-language labels, colors, and small formatting helpers for the
audience-facing dashboard. Every string in here is meant to be read by
someone with no ML/statistics background -- no internal module names,
metric jargon, or implementation details belong in this file. Technical
detail lives in the code comments and docs/, not on screen.
"""
from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Plain-language flag categories (mirrors pipeline.FLAG_COLORS' keys exactly)
# ---------------------------------------------------------------------------

FLAG_COLORS = {
    "flagged_overclaim": "#d62728",
    "flagged_night_injection": "#ff7f0e",
    "flagged_stuck_sensor": "#9467bd",
    "fault_underclaim": "#1f77b4",
    "normal": "#7f7f7f",
}

FLAG_LABELS_SIMPLE = {
    "flagged_overclaim": "Overclaimed energy",
    "flagged_night_injection": "Reported power at night",
    "flagged_stuck_sensor": "Frozen sensor reading",
    "fault_underclaim": "Under-claimed (equipment issue, not fraud)",
    "normal": "Normal",
}

BAND_COLORS = {"Low": "#2ca02c", "Elevated": "#bcbd22", "High": "#ff7f0e", "Critical": "#d62728"}

VERDICT_TEXT = {
    "Low": ("Looks trustworthy", "This period's reported generation closely matches independent weather data."),
    "Elevated": ("A few irregularities found", "Some days don't fully match expectations -- worth a closer look."),
    "High": ("Multiple irregularities found", "Several days show meaningful gaps between claimed and expected generation."),
    "Critical": ("Serious irregularities found", "This period shows strong, repeated signs of over-reporting."),
}

VERDICT_ICON = {"Low": "✅", "Elevated": "\U0001F7E1", "High": "\U0001F7E0", "Critical": "\U0001F534"}


def flag_label(flag_type: str) -> str:
    return FLAG_LABELS_SIMPLE.get(flag_type, flag_type)


def flag_color(flag_type: str) -> str:
    return FLAG_COLORS.get(flag_type, "#7f7f7f")


def band_color(band: str) -> str:
    return BAND_COLORS.get(band, "#7f7f7f")


def source_label(source_key: str) -> str:
    if source_key == "real_full":
        return "Recent activity (real data)"
    if source_key == "real_clean":
        return "Verified clean period (real data)"
    if source_key.startswith("injected_"):
        n = int(source_key.split("_")[1]) + 1
        return f"Test case {n}"
    return source_key


def sidebar_controls() -> tuple[str, str]:
    """Renders the plant + period selectors once, at the top of the single-
    page app. Because this app has no multi-page navigation, whatever the
    user picks here simply persists in st.session_state for the rest of the
    session -- there is no separate "page" for it to reset on."""
    from dashboard.lib.pipeline import PLANTS, source_options

    st.sidebar.title("CarbonTwinAI")
    st.sidebar.caption("Solar generation audit")

    plant = st.sidebar.selectbox(
        "Plant", PLANTS, key="plant", format_func=lambda p: p.replace("_", " ").title(),
    )
    opts = source_options(plant)
    keys = [k for k, _ in opts]
    if st.session_state.get("source_key") not in keys:
        st.session_state["source_key"] = keys[0]
    source_key = st.sidebar.selectbox(
        "Period", keys, key="source_key", format_func=source_label,
    )
    st.sidebar.divider()
    st.sidebar.caption(
        "Test cases are simulated scenarios (some with fraud built in, some "
        "with real-but-legitimate oddities) used to check the system works "
        "correctly. Real periods are the plant's actual reported data."
    )
    return plant, source_key


def note(text: str, title: str = "What does this mean?") -> None:
    """A short, collapsed-by-default explanation -- never a wall of text."""
    with st.expander(title):
        st.markdown(text)


def fmt_kwh(x: float) -> str:
    return f"{x:,.0f} kWh"


def fmt_usd(x: float) -> str:
    return f"${x:,.2f}"


def fmt_tco2e(x: float) -> str:
    return f"{x:,.2f} t"
