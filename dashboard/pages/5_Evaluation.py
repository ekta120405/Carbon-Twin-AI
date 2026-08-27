"""
View 5 -- evaluation/benchmark view. Per docs/LAYER4_ROADMAP.md SS4.6 and
SS6, the full quantitative evaluation (precision/recall/F1/AUC-PR, hard-
negative specificity, the MDO curve -- Overview v2 SS9) is Task 7 scope,
NOT part of this build. This page is explicitly a placeholder for that,
plus one small bonus: an informal, bounded comparison of this session's
detector output against the frozen injector's ground truth for whichever
SINGLE injected replicate is currently selected in the sidebar. This is
NOT the formal benchmark (n=1 replicate, not the full 30-replicate sweep;
no precision/recall/AUC-PR computed) -- it exists only so a demo audience
sees *some* evidence the detector works, without overstating what's been
validated. See docs/LAYER3_HANDOFF.md SS8 for the actual bounded
hard-negative specificity check this project already ran (2 examples/type,
28 total), which is a different, already-completed thing.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from dashboard.lib.pipeline import get_scored_context, load_plant_base, load_ground_truth
from dashboard.lib.ui import sidebar_controls, flag_label

st.set_page_config(page_title="Evaluation -- CarbonTwinAI", page_icon="🌞", layout="wide")

plant, source_key = sidebar_controls()
st.title("5. Evaluation / Benchmark")

st.warning(
    "**This view is a placeholder.** The full quantitative evaluation "
    "(precision / recall / F1 / AUC-PR against injected labels, specificity "
    "on the hard-negative set, and the Minimum Detectable Overstatement "
    "curve across 8 magnitudes x 5 aggregation windows -- Overview v2 SS9) "
    "is Task 7 scope and has not been run as part of this build. Building "
    "this view further is intentionally deferred, per "
    "docs/LAYER4_ROADMAP.md SS4.6, so it does not block the rest of the "
    "dashboard."
)

if not source_key.startswith("injected_"):
    st.info(
        "Select an **injected replicate** in the sidebar to see an informal "
        "day-by-day comparison against that replicate's frozen ground truth. "
        "'Real' data sources have no ground truth to compare against."
    )
    st.stop()

rep_id = int(source_key.split("_")[1])
gt = load_ground_truth(plant)
if gt.empty:
    st.error(f"No ground truth found for {plant} at data/ground_truth/{plant}_labels.parquet")
    st.stop()

gt_rep = gt[gt["replicate"] == rep_id].copy()
if gt_rep.empty:
    st.info(f"No ground truth rows for replicate {rep_id:03d}.")
    st.stop()

ctx = get_scored_context(plant, source_key)
labels = ctx["labels"]

gt_rep["date"] = pd.to_datetime(gt_rep["date"]).dt.date
gt_rep["detected_flag_type"] = gt_rep["date"].map(lambda d: labels.get(d, "normal"))
gt_rep["detected_as_fraud"] = gt_rep["detected_flag_type"].astype(str).str.startswith("flagged")

st.caption(
    f"{plant}, replicate {rep_id:03d} -- {len(gt_rep)} labeled day(s) from the "
    "frozen injector. `kind` = clean / positive (injected anomaly) / "
    "hard_negative (legitimate fault that should NOT be flagged as fraud)."
)

def _row_style(row):
    if row["kind"] == "positive":
        ok = row["detected_as_fraud"]
        return ["background-color: #eafaea" if ok else "background-color: #fbdada"] * len(row)
    if row["kind"] == "hard_negative":
        ok = not row["detected_as_fraud"]
        return ["background-color: #eafaea" if ok else "background-color: #fbdada"] * len(row)
    return [""] * len(row)

display_cols = ["date", "kind", "type", "detected_flag_type", "detected_as_fraud"]
display = gt_rep[display_cols].copy()
display["detected_flag_type"] = display["detected_flag_type"].map(lambda f: flag_label(f))
st.dataframe(display.style.apply(_row_style, axis=1), width='stretch', hide_index=True)

n_positive = int((gt_rep["kind"] == "positive").sum())
n_positive_caught = int(((gt_rep["kind"] == "positive") & gt_rep["detected_as_fraud"]).sum())
n_hard_neg = int((gt_rep["kind"] == "hard_negative").sum())
n_hard_neg_clean = int(((gt_rep["kind"] == "hard_negative") & ~gt_rep["detected_as_fraud"]).sum())

c1, c2 = st.columns(2)
c1.metric("Injected anomaly days caught", f"{n_positive_caught} / {n_positive}" if n_positive else "n/a")
c2.metric("Hard negatives correctly left unflagged", f"{n_hard_neg_clean} / {n_hard_neg}" if n_hard_neg else "n/a")

st.caption(
    "Informal, single-replicate sanity check only -- not a statistically "
    "meaningful recall/specificity estimate. The project's actual bounded "
    "hard-negative check (docs/LAYER3_HANDOFF.md SS8) used 28 examples "
    "across both plants and found specificity 0.929 (26/28), with both "
    "false positives being the documented stuck-sensor/N3-curtailment "
    "overlap -- that check is not reproduced live here."
)
