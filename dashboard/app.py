"""
dashboard/app.py

CarbonTwinAI -- audience-facing dashboard. Single page, five tabs. This is
a deliberate simplification for presenting to a non-technical audience:
- ONE Streamlit script (no dashboard/pages/ multipage navigation), so the
  plant/period selection in the sidebar persists automatically across every
  tab -- there is no "page" for it to reset on.
- No internal module/metric jargon on screen (no "Layer 3.6", "CUSUM",
  "conformal", "SHAP", "flag_type", "F1/AUC-PR"...) -- every label is
  written for someone seeing this for the first time. Full technical detail
  lives in docs/ and reports/tables/, not here.
- Carbon credits get their own dedicated, prominent tab, since that's the
  project's actual subject.

Run with:  streamlit run dashboard/app.py   (from the repo root, .venv active)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from dashboard.lib.pipeline import get_scored_context, load_plant_base
from dashboard.lib.ui import (
    sidebar_controls, note, flag_color, flag_label, band_color, FLAG_COLORS,
    VERDICT_TEXT, VERDICT_ICON, fmt_kwh, fmt_usd, fmt_tco2e,
)

# Plain-English names for the technical weather/evidence feature columns that
# can appear in a Flag Details explanation chart. Anything not listed here
# falls back to a readable version of the raw column name (underscores ->
# spaces, "twin_"/"evidence_" prefixes stripped) so a new feature never shows
# raw as e.g. "twin_ghi_lag1" without at least some cleanup.
READABLE_FACTORS = {
    "twin_ghi": "Sunlight intensity",
    "twin_clearsky_ghi": "Expected sunlight (clear sky)",
    "twin_kt": "Cloud cover",
    "twin_t2m": "Air temperature",
    "twin_rh2m": "Humidity",
    "twin_ws2m": "Wind speed",
    "twin_ps": "Air pressure",
    "twin_prectotcorr": "Rainfall",
    "twin_solar_zenith": "Sun angle",
    "twin_solar_elevation": "Sun height",
    "twin_solar_azimuth": "Sun direction",
    "twin_airmass": "Atmospheric thickness",
    "twin_cell_temp": "Panel temperature",
    "twin_hour_sin": "Time of day",
    "twin_hour_cos": "Time of day",
    "twin_doy_sin": "Season",
    "twin_doy_cos": "Season",
    "twin_ghi_lag1": "Recent sunlight trend",
    "twin_ghi_lag2": "Recent sunlight trend",
    "twin_kt_lag1": "Recent cloud trend",
    "evidence_irr_divergence_pct": "On-site vs. satellite sunlight mismatch",
    "evidence_temp_divergence": "On-site vs. satellite temperature mismatch",
    "evidence_onsite_irradiation": "On-site sunlight reading",
    "night_claimed_kwh": "Power claimed at night",
    "yield_jump_frac": "Sudden jump vs. recent days",
    # Day-level aggregated versions of the three evidence_* columns above
    # (src/layer4_xai/flag_shap.py's compute_day_level_evidence_features,
    # the actual column names Path A's suspicion-driver ranking uses) --
    # separate keys because they're literally different column names
    # (aggregated to one-row-per-date, "evidence_" prefix dropped, "_mean"
    # suffix added), not a duplicate entry.
    "irr_divergence_pct_mean": "On-site vs. satellite sunlight mismatch",
    "temp_divergence_mean": "On-site vs. satellite temperature mismatch",
    "onsite_irradiation_mean": "On-site sunlight reading",
}


def _readable_factor(raw: str) -> str:
    if raw in READABLE_FACTORS:
        return READABLE_FACTORS[raw]
    return raw.replace("twin_", "").replace("evidence_", "").replace("_", " ").strip().capitalize()


CAVEAT_BY_FLAG_TYPE = {
    "flagged_stuck_sensor": (
        "This kind of flag can occasionally also fire on a legitimate "
        "grid-ordered power cap, not just a frozen sensor — treat as "
        "suspected, not confirmed."
    ),
    "flagged_night_injection": (
        "The 'what we expected' panel shows the day's most unusual "
        "daytime moment, not the actual night-time reading that triggered "
        "this flag — see the 'why it's flagged' panel for that."
    ),
}


def _plain_caveat(flag_type: str) -> str:
    return CAVEAT_BY_FLAG_TYPE.get(flag_type, "This is a statistical flag, not a confirmed finding — worth a manual check.")


def _synthesize_physical_text(feats: list[tuple[str, float]]) -> str:
    """
    Turns the top 1-2 twin-SHAP drivers (already sorted by |contribution|,
    src/layer4_xai/twin_shap.py) into one plain-language sentence, so the
    chart isn't the only explanation on screen. Direction is well-defined
    here (positive kW = pushed the estimate up, negative = pushed it
    down — see LAYER4_MODULES_1_TO_5_HANDOFF.md SS1), so it's safe to state.
    """
    if not feats:
        return ""
    bits = [
        f"**{_readable_factor(name)}** ({'pushed the estimate up' if val >= 0 else 'pushed it down'})"
        for name, val in feats[:2]
    ]
    if len(bits) == 1:
        return f"Biggest factor today: {bits[0]}."
    return f"Biggest factors today: {bits[0]} and {bits[1]}."


def _synthesize_suspicion_text(drivers: list[tuple[str, float]]) -> str:
    """
    Turns the top 1-2 flag-SHAP drivers (already sorted by |z-score|,
    src/layer4_xai/flag_shap.py) into one plain-language sentence.
    Deliberately magnitude-only, no up/down claim: unlike the physical
    panel, what a positive vs. negative z-score MEANS varies per feature
    (e.g. an unusually high vs. low on-site/satellite mismatch can both be
    a red flag depending on the feature), so stating a direction here
    could misrepresent what the number actually means.
    """
    if not drivers:
        return ""
    names = [f"**{_readable_factor(name)}**" for name, _ in drivers[:2]]
    if len(names) == 1:
        return f"The clue that stood out most: {names[0]} — far outside what normal, honest days look like."
    return f"The clues that stood out most: {names[0]} and {names[1]} — both far outside what normal, honest days look like."

st.set_page_config(page_title="CarbonTwinAI", page_icon="🌞", layout="wide")

plant, source_key = sidebar_controls()

st.title("CarbonTwinAI")
st.caption("Checks a solar plant's reported generation against independent weather data, and prices any gap in carbon credits.")

with st.spinner("Loading..."):
    base = load_plant_base(plant)
    ctx = get_scored_context(plant, source_key)

for w in ctx["warnings"]:
    st.warning(w)

risk = ctx["risk_result"]
labels = ctx["labels"]
n_flagged = int(labels.str.startswith("flagged").sum())
n_total = len(labels)
carbon_agg = ctx["carbon_agg"]

tab_overview, tab_evidence, tab_carbon, tab_detail, tab_accuracy = st.tabs([
    "Overview", "Evidence", "Carbon Credits", "Flag Details", "Accuracy & Testing",
])

# ===========================================================================
# TAB 1 -- Overview
# ===========================================================================
with tab_overview:
    icon = VERDICT_ICON.get(risk.band, "")
    headline, sub = VERDICT_TEXT.get(risk.band, ("", ""))
    st.markdown(
        f"""<div style="background:{band_color(risk.band)}15; border:1px solid {band_color(risk.band)}55;
        border-radius:12px; padding:20px 24px; margin-bottom:8px;">
        <div style="font-size:1.4rem; font-weight:700;">{icon}&nbsp; {headline}</div>
        <div style="color:#666; margin-top:4px;">{sub}</div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.caption(f"Risk score: **{risk.score:.0f} / 100**")

    c1, c2, c3 = st.columns(3)
    c1.metric("Days reviewed", n_total)
    c2.metric("Days flagged", n_flagged)
    c3.metric("Carbon credits at risk", fmt_usd(carbon_agg["total_value_at_risk_usd"]))

    st.subheader("What was flagged, and how much")
    counts = labels.value_counts()
    if n_flagged > 0:
        order = [k for k in FLAG_COLORS if k in counts.index]
        fig = go.Figure(go.Bar(
            x=[flag_label(k) for k in order], y=[int(counts[k]) for k in order],
            marker_color=[flag_color(k) for k in order],
        ))
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Days")
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No flags in this period.")

    note(
        "The risk score combines five checks: how many days were flagged, "
        "how much energy is unaccounted for, whether there's a slow "
        "creeping pattern of over-reporting, whether this plant's numbers "
        "match its own normal behavior, and whether its on-site sensors "
        "agree with independent satellite weather data. A LOW score means "
        "the plant looks clean; a HIGH score means more irregularities were "
        "found.",
        title="How is the risk score calculated?",
    )

# ===========================================================================
# TAB 2 -- Evidence (time series)
# ===========================================================================
with tab_evidence:
    st.subheader("Claimed vs. expected generation")
    ts = ctx["ts_table"]
    daily = ctx["daily"]
    threshold = base["conformal_result"].threshold

    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.65, 0.35], vertical_spacing=0.10,
        subplot_titles=("Power over time", "Daily gap vs. the flag threshold"),
    )
    fig.add_trace(go.Scatter(x=ts.index, y=ts["expected_q90"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ts.index, y=ts["expected_q10"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(31,119,180,0.15)", name="Expected range"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ts.index, y=ts["expected_q50"], mode="lines", line=dict(color="#1f77b4", width=1.5), name="Expected"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ts.index, y=ts["ac_power"], mode="lines", line=dict(color="#d62728", width=1.5), name="Claimed"), row=1, col=1)

    for date, ft in labels.items():
        if not str(ft).startswith("flagged"):
            continue
        start = pd.Timestamp(date, tz=ts.index.tz)
        fig.add_vrect(x0=start, x1=start + pd.Timedelta(days=1), fillcolor=flag_color(ft), opacity=0.15, line_width=0, row=1, col=1)

    daily_sorted = daily.sort_index()
    bar_colors = [flag_color(labels.get(d, "normal")) for d in daily_sorted.index]
    fig.add_trace(go.Bar(x=[pd.Timestamp(d) for d in daily_sorted.index], y=daily_sorted["r_day"], marker_color=bar_colors, showlegend=False), row=2, col=1)
    fig.add_hline(y=threshold, line=dict(color="black", dash="dash"), row=2, col=1, annotation_text="Flag threshold")

    fig.update_layout(height=560, legend=dict(orientation="h", y=1.1), margin=dict(t=50, b=10))
    fig.update_yaxes(title_text="kW", row=1, col=1)
    fig.update_yaxes(title_text="Gap", row=2, col=1)
    st.plotly_chart(fig, width="stretch")

    note(
        "The red line is what the plant claimed. The blue line (with a "
        "shaded range) is what independent weather data says was possible. "
        "Shaded stripes mark flagged days. The bottom chart shows the "
        "day's actual gap against the line it needs to cross to get "
        "flagged.",
        title="How to read this chart",
    )

# ===========================================================================
# TAB 3 -- Carbon Credits (the project's actual subject -- kept front and center)
# ===========================================================================
with tab_carbon:
    st.subheader("Carbon credits at risk")
    st.caption("The estimated value of carbon credits tied to flagged, potentially over-reported energy.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Flagged days", carbon_agg["n_flagged_days"])
    c2.metric("CO₂e at risk", fmt_tco2e(carbon_agg["total_excess_tco2e"]))
    c3.metric("Value at risk", fmt_usd(carbon_agg["total_value_at_risk_usd"]))

    carbon_table = ctx["carbon_table"]
    if carbon_agg["by_flag_type"]:
        st.markdown("##### By category")
        rows = [{"Category": flag_label(ft), "CO2e (t)": v["excess_tco2e"], "Value ($)": v["value_at_risk_usd"]}
                for ft, v in carbon_agg["by_flag_type"].items()]
        by_type_df = pd.DataFrame(rows)
        fig = go.Figure(go.Bar(
            x=by_type_df["Category"], y=by_type_df["Value ($)"],
            marker_color=[flag_color(ft) for ft in carbon_agg["by_flag_type"].keys()],
            text=by_type_df["Value ($)"].round(2), textposition="outside",
        ))
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Value at risk ($)")
        st.plotly_chart(fig, width="stretch")

        st.markdown("##### By day")
        flagged_only = carbon_table[carbon_table["flag_type"].str.startswith("flagged")].copy()
        flagged_only.insert(0, "Category", flagged_only["flag_type"].map(flag_label))
        display_df = flagged_only[["Category", "severity_kwh", "excess_tco2e", "value_at_risk_usd"]].rename(columns={
            "severity_kwh": "Energy (kWh)", "excess_tco2e": "CO2e (t)", "value_at_risk_usd": "Value ($)",
        }).round(2)
        st.dataframe(display_df, width="stretch")
    else:
        st.info("No carbon credits at risk in this period — nothing was flagged.")

    note(
        "- These are estimates, not a market quote: the carbon price used is a "
        "representative reference value (voluntary carbon markets don't have "
        "one fixed price).\n"
        "- Figures say **'at risk'**, never **'confirmed fraud'** — a flag is "
        "a statistical red flag, not a legal finding.",
        title="About these numbers",
    )

# ===========================================================================
# TAB 4 -- Flag Details
# ===========================================================================
with tab_detail:
    st.subheader("Look at one flagged day")
    ranked = ctx["ranked"]
    bundles_by_date = ctx["bundles_by_date"]

    if len(ranked) == 0:
        st.info("No flagged days in this period — try a test case from the sidebar to see an example.")
    else:
        option_labels = {
            str(d): f"{d} — {flag_label(ranked.loc[d, 'flag_type'])} — {fmt_usd(ranked.loc[d, 'value_at_risk_usd'])}"
            for d in ranked.index
        }
        options = list(option_labels.keys())
        choice = st.selectbox("Pick a flagged day", options, format_func=lambda s: option_labels[s])
        sel_date = ranked.index[options.index(choice)]
        bundle = bundles_by_date.get(str(sel_date))

        if bundle is None:
            st.error("Details unavailable for this day.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Energy overclaimed", fmt_kwh(bundle["severity_kwh"]))
            c2.metric("CO2e at risk", fmt_tco2e(bundle["carbon_exposure"]["excess_tco2e"]))
            c3.metric("Value at risk", fmt_usd(bundle["carbon_exposure"]["value_at_risk_usd"]))

            left, right = st.columns(2)
            with left:
                st.markdown("##### What we expected")
                phys = bundle["physical_explanation"]
                feats = phys["top_features_kw"] if phys["mode"] == "shap" else []
                if feats:
                    names = [_readable_factor(f[0]) for f in feats][::-1]
                    vals = [f[1] for f in feats][::-1]
                    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in vals]
                    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=colors))
                    fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Effect on expected power (kW)")
                    st.plotly_chart(fig, width="stretch")
                    st.info(_synthesize_physical_text(feats))
                else:
                    st.caption("Weather conditions were too overcast for a detailed breakdown; a simpler physics-based estimate was used instead.")

            with right:
                st.markdown("##### Why it's flagged")
                susp = bundle["suspicion_explanation"]
                if susp["mode"] == "statistical":
                    drivers = susp["top_drivers"]
                    names = [_readable_factor(d[0]) for d in drivers][::-1]
                    zscores = [d[1] for d in drivers][::-1]
                    colors = ["#d62728" if abs(z) >= 2 else "#ff7f0e" if abs(z) >= 1 else "#7f7f7f" for z in zscores]
                    fig2 = go.Figure(go.Bar(x=zscores, y=names, orientation="h", marker_color=colors))
                    fig2.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="How unusual, vs. normal clean days")
                    st.plotly_chart(fig2, width="stretch")
                    st.info(_synthesize_suspicion_text(drivers))
                else:
                    st.warning(susp["text"])

            if bundle["known_caveats"]:
                st.caption("⚠ " + _plain_caveat(bundle["flag_type"]))

# ===========================================================================
# TAB 5 -- Accuracy & Testing (static, pooled across all test cases)
# ===========================================================================
with tab_accuracy:
    st.subheader("How well does this work?")
    st.caption("Tested against 30 simulated scenarios per plant, each with a known answer sealed before testing.")

    bench_path = ROOT / "reports" / "tables" / "task7_benchmark_full.json"
    mdo_path = ROOT / "reports" / "tables" / "mdo_curve.json"

    if not bench_path.exists():
        st.error("Test results not found. Run `python -m src.evaluation.benchmark` first.")
    else:
        with open(bench_path) as f:
            bench = json.load(f)
        overall = bench["pooled"]["production_flag_only"]["overall"]
        specificity = bench["pooled"]["production_flag_only"]["hard_negative_overall"]["specificity"]

        c1, c2 = st.columns(2)
        c1.metric("Fraud caught", f"{overall['recall']*100:.0f}%", help="Of all simulated fraud cases, the share correctly flagged.")
        c2.metric("Legitimate cases correctly left alone", f"{specificity*100:.0f}%", help="Of all real-but-unusual, non-fraud cases (equipment issues, cloud cover, grid limits...), the share correctly NOT flagged.")

        note(
            "We built 30 test scenarios per plant with the answers sealed "
            "before any tuning happened — a mix of simulated fraud and "
            "real but legitimate oddities (equipment outages, cloudy days, "
            "grid limits) that must NOT be mistaken for fraud. These two "
            "numbers are how often the system got each kind right.",
            title="How was this tested?",
        )

    if mdo_path.exists():
        with open(mdo_path) as f:
            mdo = json.load(f)
        st.markdown("##### How small a fraud can we catch?")
        summary_df = pd.DataFrame(mdo["summary"])
        label_map = {"15min": "A single moment", "hourly": "One hour", "daily": "A full day"}
        fig = go.Figure()
        for window in mdo["windows_computed"]:
            sub = summary_df[summary_df["window"] == window].sort_values("magnitude")
            fig.add_trace(go.Scatter(x=sub["magnitude"] * 100, y=sub["detection_rate"], mode="lines+markers", name=label_map.get(window, window)))
        fig.update_layout(height=340, xaxis_title="Size of the over-claim (%)", yaxis_title="Chance of catching it", yaxis_range=[-0.05, 1.05], margin=dict(t=20, b=10))
        st.plotly_chart(fig, width="stretch")
        note(
            "The shorter a fraudulent spike lasts, the harder it is to "
            "catch — even a large but brief over-claim can hide inside a "
            "normal day's total. Sustained, full-day over-claiming is what "
            "this system is built to catch reliably.",
            title="What does this chart show?",
        )
