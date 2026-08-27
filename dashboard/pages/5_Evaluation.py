"""
View 5 -- Evaluation / benchmark. FULL REPLACEMENT (Task 7.6) of the
single-replicate ad hoc placeholder that shipped with 4.6, per
docs/LAYER4_DASHBOARD_HANDOFF.md SS3/SS5 and TASK7_FULL_ROADMAP_FOR_
CLAUDE_CODE.md SS7.6's explicit instruction: "Do not merge the
placeholder's old numbers with the real ones -- full replacement."

Shows Task 7's five outputs, all loaded directly from their committed
reports/tables/*.json files -- this page performs NO new computation,
matching every other dashboard view's "consume, don't recompute"
discipline (LAYER4_DASHBOARD_HANDOFF.md SS1). NOT tied to the sidebar's
plant/data-source selection: these are pooled, full-30-replicate x
2-plant results, a different scope than the other four views' single-
(plant, source) drill-down -- the sidebar stays visible for navigation
consistency but doesn't filter anything on this page.

If any reports/tables/*.json file is missing (e.g. a fresh checkout where
Task 7's modules haven't been run), that section shows an explicit error
naming the script to run, never a silent placeholder pretending to be
real data.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.lib.ui import sidebar_controls

st.set_page_config(page_title="Evaluation -- CarbonTwinAI", page_icon="🌞", layout="wide")

sidebar_controls()
st.title("5. Evaluation / Benchmark")
st.caption(
    "Task 7's full evaluation -- 30 injected replicates x 2 plants, both "
    "held-out from every Layer 3 calibration. Not filtered by the sidebar "
    "selection above; these are pooled, full-scale results."
)

TABLES_DIR = ROOT / "reports" / "tables"


def _load(name: str):
    path = TABLES_DIR / name
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


benchmark = _load("task7_benchmark_full.json")
mdo = _load("mdo_curve.json")
lopo = _load("leave_one_plant_out.json")
ablations = _load("ablations.json")
stuck_fix = _load("stuck_sensor_fix_reattempt.json")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "7.1 Full Benchmark", "7.2 MDO Curve", "7.3 Leave-One-Plant-Out",
    "7.4 Ablations", "7.5 Stuck-Sensor Fix",
])

# ---------------------------------------------------------------------------
# 7.1 Full benchmark
# ---------------------------------------------------------------------------
with tab1:
    if benchmark is None:
        st.error("reports/tables/task7_benchmark_full.json not found -- run `python -m src.evaluation.benchmark`.")
    else:
        st.info(benchmark["methodology_note"])
        definition = st.radio(
            "Detection definition", ["production_flag_only", "including_cusum"],
            horizontal=True, key="bench_def",
            help="production_flag_only = what the dashboard/evidence bundles actually use. "
                 "including_cusum also counts CUSUM's own flag (Overview v2's stated A6 mechanism, "
                 "architecturally separate from classify_days()).",
        )
        pooled = benchmark["pooled"][definition]
        overall = pooled["overall"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precision", f"{overall['precision']:.3f}")
        c2.metric("Recall", f"{overall['recall']:.3f}")
        c3.metric("F1", f"{overall['f1']:.3f}")
        c4.metric("AUC-PR", f"{overall['auc_pr']:.3f}")
        st.caption(f"n={overall['n']} day-instances (n_positive={overall['n_positive']}, n_negative={overall['n_negative']}) "
                   f"-- TP={overall['tp']} FP={overall['fp']} FN={overall['fn']} TN={overall['tn']}")

        hn_overall = pooled["hard_negative_overall"]
        st.metric("Hard-negative specificity (pooled)", f"{hn_overall['specificity']:.3f}",
                   help=f"n={hn_overall['n']}, correctly unflagged={hn_overall['n_correctly_unflagged']}")

        st.subheader("Per anomaly type (recall) -- positives")
        per_type = pooled["per_type"]
        rows = [{"type": t, **m} for t, m in per_type.items()]
        df_pt = pd.DataFrame(rows).sort_values("recall", ascending=False)
        fig = go.Figure(go.Bar(x=df_pt["type"], y=df_pt["recall"], text=df_pt["n"].map(lambda n: f"n={n}"), textposition="outside"))
        fig.update_layout(height=350, yaxis_title="Recall", yaxis_range=[0, 1.15], margin=dict(t=10, b=10))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(df_pt.set_index("type"), width="stretch")
        st.caption(f"Shared false-positive pool (clean+hard-negative days flagged): "
                   f"{pooled['shared_fp_pool']['shared_fp_pool_flagged']} / {pooled['shared_fp_pool']['shared_fp_pool_total']} -- "
                   f"reused across every type's precision (false positives have no injected type of their own).")

        st.subheader("Per hard-negative type (specificity)")
        hn_per_type = pooled["hard_negative_per_type"]
        rows2 = [{"type": t, **m} for t, m in hn_per_type.items()]
        df_hn = pd.DataFrame(rows2).sort_values("specificity")
        fig2 = go.Figure(go.Bar(x=df_hn["type"], y=df_hn["specificity"], text=df_hn["n"].map(lambda n: f"n={n}"), textposition="outside"))
        fig2.update_layout(height=350, yaxis_title="Specificity", yaxis_range=[0, 1.15], margin=dict(t=10, b=10))
        st.plotly_chart(fig2, width="stretch")
        st.dataframe(df_hn.set_index("type"), width="stretch")
        if "N3_curtailment" in hn_per_type:
            st.warning(
                f"N3_curtailment specificity is {hn_per_type['N3_curtailment']['specificity']:.3f} -- "
                f"the documented stuck-sensor/N3 collision (see tab 7.5). Confirmed at full scale, "
                f"not just the earlier n=2 bounded check."
            )

# ---------------------------------------------------------------------------
# 7.2 MDO curve
# ---------------------------------------------------------------------------
with tab2:
    if mdo is None:
        st.error("reports/tables/mdo_curve.json not found -- run `python -m src.evaluation.mdo_curve`.")
    else:
        st.warning(mdo["weekly_monthly_infeasibility_note"])
        if mdo.get("excluded_baseline_false_positives"):
            st.info(f"Excluded pre-existing baseline false positives (already flagged at zero "
                    f"injection, so injecting a magnitude tests nothing): {mdo['excluded_baseline_false_positives']}")

        summary_df = pd.DataFrame(mdo["summary"])
        n_per_cell = int(summary_df["n"].iloc[0]) if len(summary_df) else 0

        fig = go.Figure()
        offsets = {"15min": -0.015, "hourly": 0.015, "daily": 0.0}
        for window in mdo["windows_computed"]:
            sub = summary_df[summary_df["window"] == window].sort_values("magnitude")
            fig.add_trace(go.Scatter(
                x=sub["magnitude"] * 100, y=sub["detection_rate"] + offsets.get(window, 0),
                mode="lines+markers", name=window,
            ))
        fig.update_layout(
            height=420, xaxis_title="Injected overstatement magnitude (% of claimed power)",
            yaxis_title="Detection rate", yaxis_range=[-0.08, 1.08],
            title=f"Minimum Detectable Overstatement (n={n_per_cell} held-out clean days/cell, pooled)",
            margin=dict(t=40, b=10),
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "'Window' = how much of a day's timesteps the injected overstatement touches, not a "
            "separate statistical test -- this project has exactly one decision point, the daily "
            "conformal test. A narrow window diluted into the same daily aggregate is mechanically "
            "harder to detect; that dilution effect IS the trade-off being characterized here."
        )
        st.dataframe(summary_df.pivot(index="magnitude", columns="window", values="detection_rate"), width="stretch")

# ---------------------------------------------------------------------------
# 7.3 Leave-one-plant-out
# ---------------------------------------------------------------------------
with tab3:
    if lopo is None:
        st.error("reports/tables/leave_one_plant_out.json not found -- run `python -m src.evaluation.leave_one_plant_out`.")
    else:
        st.info(lopo["methodology_note"])
        comparison = lopo["comparison_vs_own_twin"]
        for plant, comp in comparison.items():
            st.subheader(plant.replace("_", " ").title())
            rows = []
            for arm in ("own_twin", "cross_twin"):
                m = comp[arm]
                rows.append({"twin": arm, **m,
                             "hard_negative_specificity": comp[f"{arm}_hard_negative_specificity"]})
            df_arm = pd.DataFrame(rows).set_index("twin")
            c1, c2 = st.columns([1, 1])
            with c1:
                st.dataframe(df_arm, width="stretch")
            with c2:
                fig = go.Figure()
                for metric in ("precision", "recall", "f1", "hard_negative_specificity"):
                    fig.add_trace(go.Bar(name=metric, x=list(df_arm.index), y=df_arm[metric]))
                fig.update_layout(barmode="group", height=280, margin=dict(t=10, b=10), yaxis_range=[0, 1.05])
                st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# 7.4 Ablations
# ---------------------------------------------------------------------------
with tab4:
    if ablations is None:
        st.error("reports/tables/ablations.json not found -- run `python -m src.evaluation.ablations`.")
    else:
        st.markdown("#### (a) Satellite vs. on-site weather input")
        st.caption("On-site input is used ONLY for this accuracy comparison -- never deployed. "
                   "See src/evaluation/ablations.py's R1-invariant note.")
        a = ablations["a_satellite_vs_onsite"]
        for plant, m in a.items():
            sat_r2 = m["satellite_input_pigbqr"]["overall"]["r2"]
            onsite_r2 = m["onsite_input_pigbqr"]["overall"]["r2"]
            st.write(f"**{plant}**: satellite R²={sat_r2:.3f}  |  on-site R²={onsite_r2:.3f}  "
                     f"(on-site {'higher' if onsite_r2 > sat_r2 else 'lower'} -- expected, on-site is "
                     f"co-located, satellite is a ~110km grid average; this is the accuracy cost paid "
                     f"for tamper-resistance)")

        st.divider()
        st.markdown("#### (b) With / without physics anchor")
        st.caption("Pulled from Layer 2's committed twin_metrics.json -- not recomputed here.")
        b = ablations["b_with_without_physics_anchor"]
        for plant, m in b.items():
            st.write(f"**{plant}**: PI-GBQR R²={m['with_physics_anchor_pigbqr']['overall']['r2']:.3f}  |  "
                     f"physics-only R²={m['physics_anchor_only']['overall']['r2']:.3f}")

        st.divider()
        st.markdown("#### (c) Conformal vs. fixed 2σ threshold")
        c = ablations["c_conformal_vs_fixed_2sigma"]["pooled"]
        fig_c = go.Figure()
        for arm in ("conformal", "fixed_2sigma"):
            fig_c.add_trace(go.Bar(name=arm, x=["precision", "recall", "f1", "auc_pr"],
                                     y=[c[arm]["precision"], c[arm]["recall"], c[arm]["f1"], c[arm]["auc_pr"]]))
        fig_c.update_layout(barmode="group", height=300, margin=dict(t=10, b=10), yaxis_range=[0, 1.0])
        st.plotly_chart(fig_c, width="stretch")

        st.divider()
        st.markdown("#### (d) One-sided vs. two-sided")
        st.caption("Isolates the conformal magnitude test only (guardrails excluded) -- see roadmap SS discussion.")
        d = ablations["d_one_sided_vs_two_sided"]["pooled"]
        fig_d = go.Figure()
        for arm in ("one_sided", "two_sided"):
            fig_d.add_trace(go.Bar(name=arm, x=["precision", "recall", "f1"],
                                     y=[d[arm]["precision"], d[arm]["recall"], d[arm]["f1"]]))
        fig_d.update_layout(barmode="group", height=300, margin=dict(t=10, b=10), yaxis_range=[0, 1.0])
        st.plotly_chart(fig_d, width="stretch")
        st.write(f"Hard-negative specificity: one-sided={d['one_sided_hard_negative_specificity']:.3f}  |  "
                 f"two-sided={d['two_sided_hard_negative_specificity']:.3f}")

        st.divider()
        st.markdown("#### (e) Online vs. batch twin")
        st.info(ablations["e_online_vs_batch"])

        st.divider()
        st.markdown("#### (f) Conformal vs. autoencoder vs. ensemble")
        f = ablations["f_conformal_vs_autoencoder_vs_ensemble"]["pooled"]
        fig_f = go.Figure()
        for arm in ("conformal_production", "autoencoder_only", "ensemble"):
            fig_f.add_trace(go.Bar(name=arm, x=["precision", "recall", "f1"],
                                     y=[f[arm]["precision"], f[arm]["recall"], f[arm]["f1"]]))
        fig_f.update_layout(barmode="group", height=300, margin=dict(t=10, b=10), yaxis_range=[0, 1.0])
        st.plotly_chart(fig_f, width="stretch")
        st.write(f"Hard-negative specificity: conformal={f['conformal_production_hard_negative_specificity']:.3f}  |  "
                 f"autoencoder={f['autoencoder_only_hard_negative_specificity']:.3f}  |  "
                 f"ensemble={f['ensemble_hard_negative_specificity']:.3f}")
        st.caption("Autoencoder alone is far more sensitive but far less specific -- consistent with "
                   "why conformal is the primary detector and the autoencoder stays a secondary signal.")

        st.divider()
        st.markdown("#### (g) Leave-one-plant-out")
        st.caption(ablations["g_leave_one_plant_out"]["note"] + " See tab 7.3 above.")

# ---------------------------------------------------------------------------
# 7.5 Stuck-sensor fix re-attempt
# ---------------------------------------------------------------------------
with tab5:
    if stuck_fix is None:
        st.error("reports/tables/stuck_sensor_fix_reattempt.json not found -- run `python -m src.evaluation.stuck_sensor_fix`.")
    else:
        decision = stuck_fix["decision"]
        if decision.startswith("REVERTED"):
            st.error(f"**{decision}**")
        else:
            st.success(f"**{decision}**")
        st.write(stuck_fix["decision_text"])

        s = stuck_fix["summary"]
        rows = [
            {"metric": "A8 recall (old rule)", **s["a8_recall_old"]},
            {"metric": "A8 recall (ceiling-discriminated rule)", **s["a8_recall_new"]},
            {"metric": "N3 specificity (old rule)", **s["n3_specificity_old"]},
            {"metric": "N3 specificity (ceiling-discriminated rule)", **s["n3_specificity_new"]},
            {"metric": "Other hard-negative specificity (old rule)", **s["other_hard_negative_specificity_old"]},
            {"metric": "Other hard-negative specificity (ceiling-discriminated rule)", **s["other_hard_negative_specificity_new"]},
        ]
        st.dataframe(pd.DataFrame(rows).set_index("metric"), width="stretch")
        st.caption(
            "Built and evaluated as an isolated comparison in src/evaluation/stuck_sensor_fix.py -- "
            "src/layer3_anomaly/residual.py was NOT modified regardless of outcome."
        )
