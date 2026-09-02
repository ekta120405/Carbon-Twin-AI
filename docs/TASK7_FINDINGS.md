# Task 7 — Consolidated Findings (for paper reference)

Pure results reference — every number here is pulled directly from the
committed `reports/tables/*.json` outputs of Task 7's five evaluation
modules. No new analysis happens in this document; it exists so numbers
can be pulled straight into the paper (built separately) without digging
through JSON. For methodology, gotchas, and how each number was produced,
see `docs/TASK7_HANDOFF.md`.

**These numbers reflect the CURRENT production system, including a
post-Task-7 improvement (secondary-evidence escalation) — not the
original Task 7 baseline.** See **`docs/ESCALATION_IMPROVEMENT_HANDOFF.md`**
for the full story: what it does, a real bug it caught and fixed (a day
could be flagged that was literally the calib day defining the threshold
itself), and a methodology mistake made and caught along the way (an
initial "free" result did not survive honest split-validation, and a
second-round result that DID pass split-validation still regressed real
clean data and was tightened further). The original, pre-escalation
Task 7 baseline was: recall 58.5%, precision 80.1%, F1 67.7%, specificity
92.0%.

All numbers: 30 injected replicates × 2 plants, both plants' data pooled
unless stated otherwise. Source files named per section.

---

## 1. Full benchmark (7.1) — `reports/tables/task7_benchmark_full.json`

**Pooled, production flag definition** (`classify_days()`'s actual output —
what the dashboard/evidence bundles use):

| Metric | Value |
|---|---|
| n (day-instances) | 540 (193 positive, 347 negative) |
| Precision | 0.817 |
| Recall | 0.648 |
| F1 | 0.723 |
| AUC-PR | 0.766 |
| Hard-negative specificity | **0.920** (161/175 — IDENTICAL to the pre-escalation baseline, same 28 false positives) |

Sanity check: specificity matches the earlier bounded check's 0.929 (n=28)
closely — consistency across a 6× larger sample, and unaffected by the
escalation change (see `docs/ESCALATION_IMPROVEMENT_HANDOFF.md`).

**Pooled, including CUSUM's own binary flag on top** (architecturally
separate from `classify_days()`, but Overview v2's stated A6 mechanism):
precision 0.740, recall 0.663, F1 0.699, specificity 0.880 (154/175) —
worse on every metric than the production-flag-only row above (F1
0.699 vs. 0.723). Blanket-including CUSUM's flag no longer helps once
escalation is already recovering CUSUM's useful signal in the smarter,
borderline-gated way — this is consistent with the general finding
throughout this project that blindly OR-ing in a secondary detector costs
more than it gains (ablations (c)/(f)).

**Per-anomaly-type recall (production flag), pooled:**

| Type | n | Recall |
|---|---|---|
| A3_night_injection | 16 | **1.000** |
| A8_flatline | 15 | **1.000** |
| A4_day_swap | 31 | 0.742 |
| A7_capacity_violation | 27 | 0.741 |
| A1_constant_overclaim | 29 | 0.759 |
| A5_weather_spoofing | 30 | 0.700 |
| A2_peak_hour_shaving | 27 | 0.259 |
| A6_slow_drift | 18 | 0.056 |

**Per-hard-negative-type specificity, pooled:**

| Type | n | Specificity |
|---|---|---|
| N1_inverter_outage | 28 | 1.000 |
| N7_cloud_enhancement | 29 | 0.966 |
| N6_extreme_heat | 26 | 0.962 |
| N5_heavy_cloud_day | 23 | 0.957 |
| N2_soiling_loss | 22 | 0.909 |
| N4_sensor_dropout | 35 | 0.886 |
| **N3_curtailment** | 12 | **0.583** ← weak point, see §5 |

Every per-type specificity above is identical to the pre-escalation
baseline — the escalation logic never touches hard-negative days at this
calibration (`band_width=0.8, cusum_frac=0.5`).

---

## 2. Minimum Detectable Overstatement curve (7.2) — `reports/tables/mdo_curve.json`

n = 9 held-out clean days/cell, pooled (1 day excluded: `plant_2/2020-06-08`
was already a false positive at zero injection — see §5 of the handoff).

| Magnitude | 15min window | hourly window | daily window |
|---|---|---|---|
| 1% | 0.0 | 0.0 | 0.0 |
| 2% | 0.0 | 0.0 | 0.0 |
| 3% | 0.0 | 0.0 | 0.0 |
| 5% | 0.0 | 0.0 | 0.0 |
| 8% | 0.0 | 0.0 | 0.0 |
| 12% | 0.0 | 0.0 | **0.111** |
| 20% | 0.0 | 0.0 | **0.444** |
| 30% | 0.0 | 0.0 | **0.778** |

**Headline finding:** narrow-window (single 15-min slot or single hour)
overstatement is undetected at every tested magnitude up to 30% — this
project's detection mechanism operates at the daily level only, and a
narrow-window anomaly dilutes to a negligible fraction of the day's total
energy. weekly/monthly windows not computed — infeasible from the frozen
injector's 9-day eligible pool (see handoff for exact wording to reuse).

---

## 3. Leave-one-plant-out generalization (7.3) — `reports/tables/leave_one_plant_out.json`

| Plant (scored) | Twin used | Precision | Recall | F1 | AUC-PR | Hard-neg. specificity |
|---|---|---|---|---|---|---|
| plant_1 | own (plant_1) | 0.922 | 0.621 | 0.742 | 0.752 | 0.943 |
| plant_1 | **cross (plant_2's twin)** | 0.371 | **1.000** | 0.541 | 0.749 | **0.159** |
| plant_2 | own (plant_2) | 0.742 | 0.673 | 0.706 | 0.779 | 0.897 |
| plant_2 | cross (plant_1's twin) | 0.676 | 0.490 | 0.568 | 0.658 | 0.897 |

**Headline finding:** cross-plant generalization is asymmetric. plant_2's
twin transplanted onto plant_1 causes severe over-flagging (recall→1.0 but
specificity collapses to 0.16) — consistent with plant_2's twin being the
weaker, more overfit one (`LAYER2_HANDOFF.md` §4: plant_2 underperforms
physics-only in clear/partly-cloudy regimes). The escalation logic (see
top of this document) amplifies this specific weakness slightly further
(specificity was 0.307 pre-escalation, now 0.159) — a badly-miscalibrated
twin produces more borderline days for escalation to act on, so this is
an expected, not surprising, interaction. plant_1's twin transplanted
onto plant_2 degrades more gracefully and is essentially unaffected by
escalation (numbers match the pre-escalation baseline closely) — the one
hard-negative flagged, `2020-06-08`, is caught by the twin-independent
night-injection guardrail in both cases, verified by joining both runs'
rows and confirming every flagged instance is that same guardrail-driven
day, not a coincidence.

**Note on plant_1's AUC-PR staying roughly flat (0.752 own-twin vs. 0.749
cross-twin) despite precision/specificity collapsing:** not a
contradiction. AUC-PR is threshold-independent (integrates precision
across every cutoff of the continuous `severity_kwh` ranking);
precision/specificity are read at the one operating threshold
`classify_days()`+escalation actually uses. The underlying ranking
retains real power even though the specific classification threshold is
badly miscalibrated for this twin/plant pair — see
`docs/TASK7_HANDOFF.md` §3 for the original (pre-escalation) version of
this same investigation, which found the identical pattern.

---

## 4. Ablation study (7.4) — `reports/tables/ablations.json`

### (a) Satellite vs. on-site weather input (accuracy comparison only — never deployed)

| Plant | Satellite R² | On-site R² |
|---|---|---|
| plant_1 | 0.866 | 0.892 |
| plant_2 | 0.681 | 0.739 |

On-site input is more accurate (physically co-located vs. satellite's
~110km grid average) — quantifies the accuracy cost paid for tamper-
resistance, the architectural reason satellite-only is used in production.
The satellite R² column here and (b)'s PI-GBQR R² column are the SAME
pre-existing number, read from the same file (`{plant}_twin_metrics.json`,
untouched by Task 7 — confirmed via `git log`), not two independent
computations; only the on-site column is freshly trained in this ablation.

### (b) With/without physics anchor (pulled from Layer 2, not recomputed)

| Plant | PI-GBQR R² | Physics-only R² |
|---|---|---|
| plant_1 | 0.866 | 0.835 |
| plant_2 | 0.681 | **0.751** (twin underperforms physics-only — matches `LAYER2_HANDOFF.md` §4) |

plant_2's pooled 0.681 is legitimately higher than its own regime-specific
test R² elsewhere in the project (clear 0.307, partly-cloudy 0.140) — R²
does not decompose as an n-weighted average across regimes; the overcast
regime (n=239/480, R²=0.749, near-zero variance) contributes little
absolute error while the harder clear/partly-cloudy regimes hold most of
the real variance. Not a computation inconsistency.

### (c) Conformal vs. fixed 2σ threshold, pooled

| | Precision | Recall | F1 | AUC-PR | Hard-neg. specificity |
|---|---|---|---|---|---|
| Conformal | 0.817 | 0.648 | 0.723 | 0.766 | 0.920 |
| Fixed 2σ | 0.780 | 0.513 | 0.619 | 0.677 | 0.920 |

Conformal wins clearly on recall/F1/AUC-PR. The identical specificity
(0.920, both) was investigated, not assumed: every hard-negative false
positive under BOTH thresholds is guardrail-driven (`flagged_stuck_sensor`
/ `flagged_night_injection`), with identical `r_day` values in every case
— zero hard-negative false positives in this sample come from the
magnitude test at all, on either plant, so this specific slice literally
cannot distinguish the two thresholds. See `docs/TASK7_HANDOFF.md` §2.5.
Both rows now include the production secondary-evidence escalation
(`docs/ESCALATION_IMPROVEMENT_HANDOFF.md`), applied relative to each arm's own primary
threshold — this stays a fair, like-for-like comparison of the threshold
choice itself, just with the same enhancement layered onto both.

### (d) One-sided vs. two-sided, pooled (conformal magnitude test isolated, guardrails excluded)

| | Precision | Recall | F1 | Hard-neg. specificity |
|---|---|---|---|---|
| One-sided | 0.812 | 0.425 | 0.558 | 0.971 |
| Two-sided | 0.776 | 0.430 | 0.553 | 0.943 |

Two-sided's specificity cost is concentrated on `N4_sensor_dropout`
(plant_1: 1.0 → 0.714, 4/14 new false positives) — real but narrower than
a uniform drop across all under-generation hard negatives.

Ablation (d) is built directly from raw `r_day` comparisons, bypassing
`classify_days()`/escalation entirely by design (isolating the conformal
magnitude test alone) — its numbers are therefore UNCHANGED by the
production escalation feature (`docs/ESCALATION_IMPROVEMENT_HANDOFF.md`). Vs. §1's
current production numbers (precision 0.817, recall 0.648, specificity
0.920): isolating the conformal test alone (removing both guardrails AND
escalation) gives precision 0.812 (about the same), recall 0.425
(meaningfully *lower* — neither the guardrails' A3/A8 perfect-recall
detections nor escalation's gains are credited here), and specificity
0.971 (*higher* — the guardrail-driven hard-negative false positives are
also removed). Removing the guardrails and escalation costs recall
specifically, not a blanket "everything is lower."

### (e) Online vs. batch twin — NOT built

Reported per the pre-existing scope decision (`LAYER2_HANDOFF.md` §8): the
dataset's 34-day span gives a 20-day rolling window only 14 possible
advance positions — too short a horizon for meaningful drift, independent
of implementation effort.

### (f) Conformal vs. autoencoder vs. ensemble, pooled

| | Precision | Recall | F1 | Hard-neg. specificity |
|---|---|---|---|---|
| Conformal (production) | 0.817 | 0.648 | 0.723 | 0.920 |
| Autoencoder only (naive "any timestep" rule) | 0.351 | 0.731 | 0.474 | 0.246 |
| Ensemble (naive OR) | 0.389 | 0.938 | 0.550 | 0.194 |

This ablation deliberately still uses the ORIGINAL, naive "any single
timestep" autoencoder rule (ablations.py's `score_replicate_autoencoder`,
unchanged since it's a documented baseline comparison), not the
run-length-aware version that ships in production escalation
(`docs/ESCALATION_IMPROVEMENT_HANDOFF.md`) — the point of this ablation is showing
why the naive version was rejected, which is exactly why the production
system does NOT blindly OR the autoencoder in. Conformal (production)'s
row already includes the smarter, run-length-aware, borderline-gated use
of the autoencoder (plus CUSUM) via escalation. Autoencoder alone (naive
rule) is far too trigger-happy for standalone deployment (specificity
0.25) — explains why conformal remains the primary detector.

**Threshold definition used in THIS ablation (stated explicitly, since
it's doing real work in the 0.246 result):** per-timestep reconstruction
error (never day-mean/day-max), threshold = 95th percentile over the
pooled, held-out `calib`+`test_clean` error (model trained on `train`
only), day flagged if **any** single daylight timestep exceeds it. This
"any one timestep" rule is a much more permissive trigger than the
daily-*aggregate* conformal test and is the most likely driver of the low
specificity. **This WAS subsequently tried and confirmed**: requiring a
run of consecutive elevated timesteps (calibrated from clean data, not
guessed) is exactly what production escalation uses instead — see
`docs/ESCALATION_IMPROVEMENT_HANDOFF.md` for the run-length-aware version and why it
performs very differently from this naive baseline. See §3 above for the
original per-plant/cross-twin AUC-PR investigation this threshold
discussion is unrelated to.

### (g) Leave-one-plant-out — see §3 above, not duplicated

---

## 5. Stuck-sensor / N3-curtailment fix, re-attempted (7.5) — `reports/tables/stuck_sensor_fix_reattempt.json`

**Decision: REVERTED.** `src/layer3_anomaly/residual.py` unchanged.

| | Old rule | Ceiling-discriminated rule |
|---|---|---|
| A8 recall (n=15) | 1.000 | 0.867 |
| N3 specificity (n=12) | 0.583 | **0.583 (unchanged)** |
| Other hard-negative specificity (n=163) | 1.000 | 1.000 |

The physics-ceiling discriminator (does `physics_ceiling_fraud_kw` ever
drop below the frozen value during a stuck-sensor run?) gave **zero** N3
specificity improvement while costing 2 real A8 detections. Second
distinct fix attempt for this collision, both rejected for different
reasons (first: large regression for a real gain; second: zero gain at
all). The N3/stuck-sensor collision remains a documented, live limitation.

---

## 6. One-paragraph summary, if only one is needed

Across 30 injected replicates × 2 plants (540 day-instances, 193
positive), the production detector — including the post-Task-7 secondary-
evidence escalation feature (`docs/ESCALATION_IMPROVEMENT_HANDOFF.md`) — achieves
precision 0.82 / recall 0.65 / F1 0.72 / AUC-PR 0.77, with hard-negative
specificity 0.92 (unchanged from the pre-escalation baseline: literally
the same 28 false positives) — closely matching the earlier bounded check
(0.929, n=28). The guardrail mechanisms (A3 night-injection, A8 stuck-
sensor) achieve perfect recall by design; escalation meaningfully lifted
recall on constant-overclaim, day-swap, weather-spoofing, capacity-
violation, and peak-hour-shaving (A2: 14.8%→25.9%) at zero pooled
specificity cost — slow-drift (A6) remains the weakest category (5.6%,
essentially untouched by escalation at its final, conservative
calibration), and the one specificity weak point is the documented
N3-curtailment/stuck-sensor collision (re-attempted at full scale,
confirmed unfixable by the physics-ceiling idea, reverted). The MDO curve
shows detection is essentially daily-level-only: narrow-window anomalies
(a single hour or 15-minute slot) are undetected up to 30% magnitude —
unaffected by escalation, which only ever acts on already-borderline
full-day aggregates. Conformal calibration clearly outperforms both a
naive fixed-2σ threshold and a standalone/naively-ensembled autoencoder.
Cross-plant generalization is asymmetric — plant_2's twin degrades badly
when applied to plant_1 (and escalation mildly amplifies that specific
weakness), while the reverse degrades more gracefully.
