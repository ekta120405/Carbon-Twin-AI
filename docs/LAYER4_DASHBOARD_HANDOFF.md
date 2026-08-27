# Layer 4.6 (Dashboard) — Handoff Notes

Status as of this commit: **built, wired to real Layers 2-4 code (no
reimplemented logic), and verified — both plants, five data sources each,
all five views, zero runtime exceptions.** This is the last module in
Layer 4 (docs/LAYER4_ROADMAP.md); 4.1-4.5 are documented in
`docs/LAYER4_MODULES_1_TO_5_HANDOFF.md`.

This document exists for the same reason LAYER2/3/4(1-5)_HANDOFF.md exist:
several real bugs and non-obvious decisions were found while building this
that aren't in the roadmap. Whoever touches the dashboard next needs these,
or a change will silently reintroduce something already fixed, or silently
violate a caveat-visibility rule the other handoff docs require.

---

## 0. File locations

```
dashboard/app.py                                entry point + landing page
dashboard/lib/pipeline.py                        ALL Layer 2-4 wiring + caching lives here
dashboard/lib/ui.py                              sidebar controls, caveat text, formatters
dashboard/pages/1_Project_Risk_Overview.py        view 1 (3.6's score/band)
dashboard/pages/2_Expected_vs_Claimed.py          view 2 (time series + flags)
dashboard/pages/3_Flag_Detail.py                  view 3 (4.1/4.2 panels, 4.5 ranking, 4.3 carbon)
dashboard/pages/4_Carbon_Exposure.py              view 4 (4.3 project/period summary)
dashboard/pages/5_Evaluation.py                   view 5 (Task 7's real full-scale results, see SS3 below)
```

Run with `streamlit run dashboard/app.py` from the repo root with `.venv`
active. Streamlit's multipage convention auto-discovers `dashboard/pages/*.py`
as sidebar nav entries — the numeric filename prefixes control ordering.

**No new detection/explanation/carbon logic exists anywhere in `dashboard/`.**
Every number shown is produced by calling the real `src/layer{2,3,4}_*`
functions in the exact sequence already verified by `score_with_fixes.py`
(project risk score) and `verify_evidence_module.py` (evidence bundles) —
`dashboard/lib/pipeline.py` is a caching/wiring layer, not a reimplementation.
If a number on the dashboard looks wrong, the bug is almost certainly in how
`pipeline.py` calls the underlying module, not in the module itself — check
there first, then re-run `verify_evidence_module.py` / `score_with_fixes.py`
directly to isolate which side the bug is on.

---

## 1. Caching architecture — read before changing `pipeline.py`

Two cache layers, both `st.cache_resource` (NOT `st.cache_data`):

```python
@st.cache_resource
def load_plant_base(plant: str) -> dict:      # per-PLANT fixed quantities
@st.cache_resource
def get_scored_context(plant: str, source_key: str) -> dict   # per-(plant, source)
```

**Why `cache_resource`, not `cache_data`:** `load_plant_base`'s return dict
holds the loaded `PIGBQRTwin` (LightGBM boosters wrapped in a
`_ReadOnlyWrapper`, see LAYER2_HANDOFF.md §1) — `cache_data` pickles its
return value to build the cache key comparison, and boosters/wrappers are
not reliably picklable. `cache_resource` stores the object by identity
instead, which is what both twin objects and the DataFrames/dataclasses in
`get_scored_context`'s result need. **If you ever see a pickling error or a
hashing `UnhashableParamError` after editing this file, this is almost
certainly why — do not "fix" it by switching to `cache_data`.**

**`get_scored_context` does NOT accept `base` as a parameter.** It looks
correct to pass `load_plant_base(plant)`'s result in as an argument to avoid
recomputing it — this was tried and reverted. `st.cache_resource` hashes
its *arguments* to form the cache key, and a dict containing a twin object
is not hashable. Instead, `get_scored_context` calls `load_plant_base(plant)`
internally on every invocation; because that call is itself cached, this
costs a cheap cache lookup, not a real recompute. **Keep this pattern if you
add new cached functions that need the twin.**

### What's fixed per plant vs. recomputed per source

This split is load-bearing, not incidental — mirrors `score_with_fixes.py`'s
own structure exactly:

| Quantity | Computed in | Frozen across all sources? |
|---|---|---|
| Twin, conformal threshold, achievable alpha | `load_plant_base` | Yes — calibrated ONLY on real `calib` days |
| CUSUM reference mean/std | `load_plant_base` | Yes — pooled real `calib`+`test_clean` |
| Clean calibration-ratio baseline | `load_plant_base` | Yes |
| Divergence normalization cap | `load_plant_base` | Yes — 95th pct of real `calib`+`test_clean` |
| CUSUM's `h` threshold | `get_scored_context` | **No — recalibrated per source**, see below |
| Everything else (flags, severity, carbon, SHAP, ranking, bundles) | `get_scored_context` | No — computed fresh per source |

**CUSUM's `h` is the one deliberate exception**, recalibrated on every call
to `get_scored_context` via `calibrate_h_from_clean_data(..., window_len=len(daily))`.
This is not an inconsistency — `h` is only statistically valid for the
specific window length it was calibrated for (docs/LAYER3_HANDOFF.md §7 /
`cusum.py`'s own module docstring: a 9-day-calibrated `h` gives a
false-positive-inflating result if applied to a 34-day sequence). Do not
"simplify" this by calibrating `h` once in `load_plant_base` — it would
silently reintroduce the exact CUSUM window-length bug LAYER3_HANDOFF.md §7
already found and fixed once (a genuinely clean plant_2 scored 30.7/
"Elevated" instead of 20.2/"Low" before this fix).

---

## 2. Data sources

```python
source_options(plant) -> [
    ("real_full",  "Real data -- full history (34 days)"),
    ("real_clean", "Real data -- clean block (calib + test_clean, 10 days)"),
    ("injected_000", "Injected replicate 000"), ... ("injected_029", ...),
]
```

- `real_full`: `data/processed/{plant}.parquet` unfiltered — includes
  `train`/`calib`/`test_clean`/`test_injected` splits, all REAL (unmodified)
  claimed generation. `test_injected` here is a split-name collision, not
  synthetic data — Layer 1's split protocol names days 31-34
  `test_injected_days` in `config.yaml`, but the *real* generation for those
  days is what's in `{plant}.parquet`; the *synthetic* injected replicates
  are a completely separate set of files (below).
- `real_clean`: the same file, filtered to the 10 `calib`+`test_clean` dates
  — this is the exact block LAYER3_HANDOFF.md §7's acceptance test scores
  (plant_1=12.1/Low, plant_2=20.2/Low — reproduced by this dashboard, see §5).
- `injected_NNN`: `data/processed/injected/{plant}/replicate_NNN.parquet` —
  a 9-day window (5 real `test_clean` days + 4 re-injected `test_clean` days
  with synthetic anomalies from the frozen injector, `injector-v1.0`). 30
  replicates exist per plant. **Read-only, per docs/LAYER4_ROADMAP.md §2.8
  — the dashboard never writes to `data/processed/injected/`.**

---

## 3. View-by-view notes

### View 1 — Project Risk Overview
Pure passthrough of `compute_risk_score()`'s `RiskScoreResult` — the gauge,
band color, and 5-component bar chart are all derived directly from
`risk.score`, `risk.band`, `risk.components`. The weights shown in the
component chart (`0.30/0.30/0.20/0.12/0.08`) are hardcoded in this page to
match `aggregate.py`'s documented default — **if `compute_risk_score`'s
default `weights` dict is ever changed, update this page's copy too**, or
the displayed "contribution to score" bars will silently misrepresent the
real weighting. (Not read from `compute_risk_score` at runtime because the
function returns only the normalized components, not the weights used.)

### View 2 — Expected vs. Claimed
**Deliberately honest chart labeling, don't relax this in a future edit:**
the shaded band is the twin's own q10-q90 quantile output, explicitly
captioned as NOT a per-timestep conformal interval. Layer 3's actual
conformal guarantee applies to the *daily* `r_day` aggregate, not to
per-15-minute predictions (LAYER3_HANDOFF.md §4) — the lower subplot (daily
`r_day` bars vs. the conformal threshold line) is where the real
statistical claim lives. If someone asks to "add the conformal band to the
top chart," push back or make very clear in the label that it would be a
visual approximation, not the calibrated quantity.

### View 3 — Flag Detail
Consumes `build_evidence_bundles_for_plant()`'s output directly (same
function `verify_evidence_module.py` uses) — computed fresh per
`(plant, source_key)` inside `get_scored_context`, **not read from the
committed sample files** in `reports/evidence_bundles/` (those are just
`.gitignore`'d demo artifacts from Task-level verification runs, unrelated
to what the dashboard shows live).

Branches on `physical_explanation["mode"]` (`"shap"` vs.
`"physics_anchor_fallback"`) and `suspicion_explanation["mode"]`
(`"statistical"` vs. `"guardrail"`) exactly as `LAYER4_MODULES_1_TO_5_HANDOFF.md`
§1/§2 require — **do not render a SHAP chart when mode is
`physics_anchor_fallback`** (there is no learned correction to show; only
`note` should render), and do not force a z-score chart onto a guardrail
explanation (nothing statistical to show; only `text` + `trigger_values`).

**Real bug found and fixed during this build:** `st.dataframe()` on
`suspicion_explanation["trigger_values"]` (a dict mixing floats, ints, and a
timestamp string) crashed pandas→Arrow conversion
(`pyarrow.lib.ArrowInvalid: Could not convert '2020-06-02 00:30:00+00:00'
with type str: tried to convert to double`). Streamlit auto-recovers from
this internally (doesn't crash the page) but logs a noisy traceback. Fixed
by using `st.json(trigger_values)` instead of `st.dataframe`. **If you add
any new heterogeneous dict for display, use `st.json`, not `st.dataframe`.**

### View 4 — Carbon Exposure
`REQUIRED_CAVEATS` from `src/layer4_xai/carbon.py` is imported and rendered
expanded-by-default at the top of the page, per that module's own docstring
requirement that these appear "not just computed silently." Do not move
this below the fold or make it collapsed-by-default.

### View 5 — Evaluation
**UPDATED (Task 7.6) — no longer a placeholder.** The original single-
replicate ad hoc comparison was fully replaced (not merged) with Task 7's
real full-scale results, per `TASK7_FULL_ROADMAP_FOR_CLAUDE_CODE.md` §7.6's
explicit instruction. Five tabs, each loading directly from a committed
`reports/tables/*.json` file — this page performs no new computation:
1. **7.1 Full Benchmark** — precision/recall/F1/AUC-PR, per-anomaly-type
   recall, per-hard-negative-type specificity, toggleable between the
   production `classify_days()` flag and the CUSUM-inclusive definition
   (see `src/evaluation/benchmark.py`'s methodology note for why both are
   shown, never just one — CUSUM is architecturally separate from
   `classify_days()` but is Overview v2's stated A6 mechanism).
2. **7.2 MDO Curve** — rebuilt live from `mdo_curve.json`'s raw numbers via
   Plotly, **never from `reports/figures/mdo_curve.png`** — that PNG is
   `.gitignore`'d (regenerable output, same convention as `reports/figures/`
   generally) and won't exist on a fresh checkout; depending on it would
   silently break the tab for anyone who hasn't locally re-run
   `src/evaluation/mdo_curve.py`.
3. **7.3 Leave-One-Plant-Out** — own-twin vs. cross-twin comparison per plant.
4. **7.4 Ablations** — all six built ablations (a, b, c, d, f, g) plus (e)'s
   scoped-out note, rendered from `ablations.json`.
5. **7.5 Stuck-Sensor Fix** — the physics-ceiling discriminator re-attempt's
   decision and before/after numbers, from `stuck_sensor_fix_reattempt.json`.

Not tied to the sidebar's plant/data-source selection (the sidebar stays
visible for navigation consistency only) — these are pooled, full-scale
results across all 30 replicates × both plants, a different scope than the
other four views' single-instance drill-down. Missing table files raise an
explicit `st.error` naming the script to run, never a silent placeholder.

---

## 4. Verified against the real repo, not assumed

Before wiring the dashboard, the pipeline was run standalone (bypassing
Streamlit's cache decorators via `.__wrapped__`) across both plants and
multiple sources. Output matched the other handoff docs' own numbers
exactly, which is strong evidence the wiring is correct, not just
internally self-consistent:

- `plant_1` / `real_clean` → risk score **12.1**, band **Low**
- `plant_2` / `real_clean` → risk score **20.2**, band **Low**
  (both match LAYER3_HANDOFF.md §7's acceptance test exactly)
- `plant_1` / `injected_010` → `flagged_overclaim` x3, `flagged_stuck_sensor` x1
  (matches LAYER4_MODULES_1_TO_5_HANDOFF.md §0's stated composition exactly)
- `plant_2` / `injected_000` → `flagged_night_injection` x2, `flagged_overclaim` x1
  (matches the same document exactly)

Then every page was executed via `streamlit.testing.v1.AppTest` (not just
`streamlit run` + a manual click-through) across 5 data sources x 6 scripts
= 30 combinations, including no-flag edge cases (`real_clean`, `real_full`
on both plants) and a flag-rich case exercising all three flag-type paths
(`plant_1/injected_010`) — zero exceptions after the two bugs above were
fixed. `AppTest` actually executes each page's Python body server-side and
captures exceptions; a plain `streamlit run` + `curl` only proves the server
process started, not that the script logic ran without error — use
`AppTest` for any future regression check on this dashboard, not just an
HTTP status code.

---

## 5. Known limitations, stated plainly (not silently absorbed)

- **Evidence bundles are recomputed on every cache miss**, not persisted.
  For a 9-day injected replicate this is fast (SHAP/TreeExplainer is
  millisecond-scale per instance, per LAYER4_ROADMAP.md §4.1), but for
  `real_full` (34 days) with many flagged days this could get slower if the
  dataset ever grows. If dashboard load time becomes a problem, the fix is
  to persist `get_scored_context`'s result (e.g. via `reports/evidence_bundles/`
  keyed by `(plant, source_key)`), not to skip caveat rendering to save time.
- **No authentication / access control** — this is a local demo/audit tool,
  not a deployed service. Don't reuse this app.py as-is behind a public URL
  without adding one.
- View 5 was a placeholder by design at initial 4.6 build time (matching
  `LAYER4_ROADMAP.md` §4.6's own instruction not to block on it) — **now
  filled in with Task 7's real results, see §3 above.**

---

## 6. If you extend this later

- New cached functions that need the twin object: follow the
  `get_scored_context` pattern (call the cached per-plant loader internally,
  don't accept the twin/base dict as a function argument) — see §1.
- New heterogeneous-dict displays: use `st.json`, not `st.dataframe` — see §3.
- New caveat text: add it to `dashboard/lib/ui.py`'s constants and render
  via `render_caveats()`, don't inline caveat strings in page files — keeps
  wording consistent with the source-of-truth modules (`carbon.py`'s
  `REQUIRED_CAVEATS`, `flag_shap.py`'s `STUCK_SENSOR_CAVEAT`, etc.) and
  avoids the exact kind of drift LAYER4_ROADMAP.md's own opening section
  warns this project has hit three times already (chat-summary vs.
  committed-code drift).
- Before trusting any new number on the dashboard, do what every other
  module in this project has done: hand-verify it against a direct call to
  the underlying `src/layer{2,3,4}_*` function on the same
  `(plant, source_key)`, the same way `verify_evidence_module.py` cross-checks
  its bundles against an independently-called `compute_daily_flag_severity_kwh()`.