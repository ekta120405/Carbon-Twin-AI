# Task 7 — Consolidated Findings (for paper reference)

Pure results reference — every number here is pulled directly from the
committed `reports/tables/*.json` outputs of Task 7's five evaluation
modules. No new analysis happens in this document; it exists so numbers
can be pulled straight into the paper (built separately) without digging
through JSON. For methodology, gotchas, and how each number was produced,
see `docs/TASK7_HANDOFF.md`.

All numbers: 30 injected replicates × 2 plants, both plants' data pooled
unless stated otherwise. Source files named per section.

---

## 1. Full benchmark (7.1) — `reports/tables/task7_benchmark_full.json`

**Pooled, production flag definition** (`classify_days()`'s actual output —
what the dashboard/evidence bundles use):

| Metric | Value |
|---|---|
| n (day-instances) | 540 (193 positive, 347 negative) |
| Precision | 0.801 |
| Recall | 0.585 |
| F1 | 0.677 |
| AUC-PR | 0.725 |
| Hard-negative specificity | **0.920** (161/175) |

Sanity check: matches the earlier bounded check's 0.929 (n=28) closely —
consistency across a 6× larger sample.

**Pooled, including CUSUM's own flag** (architecturally separate from
`classify_days()`, but Overview v2's stated A6 mechanism):
Precision 0.722, Recall 0.606, F1 0.659, specificity drops (302/347
negative correctly unflagged vs. 319/347 production-only) — CUSUM catches
4 more true positives at the cost of 17 more false positives.

**Per-anomaly-type recall (production flag), pooled:**

| Type | n | Recall |
|---|---|---|
| A3_night_injection | 16 | **1.000** |
| A8_flatline | 15 | **1.000** |
| A7_capacity_violation | 27 | 0.704 |
| A1_constant_overclaim | 29 | 0.690 |
| A5_weather_spoofing | 30 | 0.667 |
| A4_day_swap | 31 | 0.581 |
| A2_peak_hour_shaving | 27 | 0.148 |
| A6_slow_drift | 18 | 0.056 (0.111 including CUSUM) |

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
| plant_1 | own (plant_1) | 0.915 | 0.568 | 0.701 | 0.718 | 0.943 |
| plant_1 | **cross (plant_2's twin)** | 0.438 | **1.000** | 0.609 | 0.760 | **0.307** |
| plant_2 | own (plant_2) | 0.720 | 0.602 | 0.656 | 0.732 | 0.897 |
| plant_2 | cross (plant_1's twin) | 0.676 | 0.490 | 0.568 | 0.658 | 0.897 |

**Headline finding:** cross-plant generalization is asymmetric. plant_2's
twin transplanted onto plant_1 causes severe over-flagging (recall→1.0 but
specificity collapses to 0.31) — consistent with plant_2's twin being the
weaker, more overfit one (`LAYER2_HANDOFF.md` §4: plant_2 underperforms
physics-only in clear/partly-cloudy regimes). plant_1's twin transplanted
onto plant_2 degrades more gracefully (recall drops modestly, specificity
unchanged — the one hard-negative flagged, `2020-06-08`, is caught by the
twin-independent night-injection guardrail in both cases, verified by
joining both runs' rows and confirming every flagged instance is that
same guardrail-driven day, not a coincidence).

**Note on plant_1's AUC-PR rising (0.718→0.760) despite precision/
specificity collapsing:** not a contradiction. AUC-PR is threshold-
independent (integrates precision across every cutoff of the continuous
`severity_kwh` ranking); precision/specificity are read at the one
operating threshold `classify_days()` actually uses. Verified by direct
recomputation and by checking mean `severity_kwh` is still higher for
positive than negative days under the cross-twin (77,409 vs. 44,567,
narrower than own-twin's 32,012 vs. 257 but still real separation) — the
underlying ranking retains real power even though the specific
classification threshold is badly miscalibrated for this twin/plant pair.

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
| Conformal | 0.801 | 0.585 | 0.677 | 0.725 | 0.920 |
| Fixed 2σ | 0.720 | 0.373 | 0.491 | 0.584 | 0.920 |

Conformal wins clearly on recall/F1/AUC-PR. The identical specificity
(0.920, both) was investigated, not assumed: every hard-negative false
positive under BOTH thresholds is guardrail-driven (`flagged_stuck_sensor`
/ `flagged_night_injection`), with identical `r_day` values in every case
— zero hard-negative false positives in this sample come from the
magnitude test at all, on either plant, so this specific slice literally
cannot distinguish the two thresholds. See `docs/TASK7_HANDOFF.md` §2.5.

### (d) One-sided vs. two-sided, pooled (conformal magnitude test isolated, guardrails excluded)

| | Precision | Recall | F1 | Hard-neg. specificity |
|---|---|---|---|---|
| One-sided | 0.812 | 0.425 | 0.558 | 0.971 |
| Two-sided | 0.776 | 0.430 | 0.553 | 0.943 |

Two-sided's specificity cost is concentrated on `N4_sensor_dropout`
(plant_1: 1.0 → 0.714, 4/14 new false positives) — real but narrower than
a uniform drop across all under-generation hard negatives.

Vs. §1's production numbers (precision 0.801, recall 0.585, specificity
0.920): isolating the conformal test alone (removing guardrails) gives
precision 0.812 (marginally *higher*), recall 0.425 (meaningfully
*lower* — the guardrails' A3/A8 perfect-recall detections aren't credited
here), and specificity 0.971 (*higher* — the guardrail-driven hard-negative
false positives are also removed). Removing the guardrails costs recall
specifically, not a blanket "everything is lower."

### (e) Online vs. batch twin — NOT built

Reported per the pre-existing scope decision (`LAYER2_HANDOFF.md` §8): the
dataset's 34-day span gives a 20-day rolling window only 14 possible
advance positions — too short a horizon for meaningful drift, independent
of implementation effort.

### (f) Conformal vs. autoencoder vs. ensemble, pooled

| | Precision | Recall | F1 | Hard-neg. specificity |
|---|---|---|---|---|
| Conformal (production) | 0.801 | 0.585 | 0.677 | 0.920 |
| Autoencoder only | 0.351 | 0.731 | 0.474 | 0.246 |
| Ensemble (OR) | 0.389 | 0.938 | 0.550 | 0.194 |

First-ever real autoencoder training run in this project (n/a before Task
7 — no persisted model or training driver existed). Autoencoder alone is
far too trigger-happy for standalone deployment (specificity 0.25) —
explains why conformal is the primary detector and autoencoder stays a
secondary/ensemble-only signal.

**Threshold definition used (stated explicitly, since it's doing real work
in the 0.246 result):** per-timestep reconstruction error (never day-mean/
day-max), threshold = 95th percentile over the pooled, held-out `calib`+
`test_clean` error (model trained on `train` only), day flagged if **any**
single daylight timestep exceeds it. This "any one timestep" rule is a
much more permissive trigger than the daily-*aggregate* conformal test and
is the most likely driver of the low specificity — a stricter rule (e.g.
requiring a run of consecutive elevated timesteps) was not tried. Read
this as a finding about this specific threshold choice, not a fully
threshold-independent statement that autoencoders can't work here — though
it's consistent either way with the architecture's framing of the
autoencoder as a secondary signal. See `docs/TASK7_HANDOFF.md` §3.

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
positive), the production detector achieves precision 0.80 / recall 0.59 /
F1 0.68 / AUC-PR 0.73, with hard-negative specificity 0.92 — closely
matching the earlier bounded check (0.929, n=28). The guardrail mechanisms
(A3 night-injection, A8 stuck-sensor) achieve perfect recall by design;
the conformal magnitude test's weakest points are peak-hour-shaving (A2)
and slow-drift (A6) detection, and its one specificity weak point is the
documented N3-curtailment/stuck-sensor collision (re-attempted at full
scale, confirmed unfixable by the physics-ceiling idea, reverted). The
MDO curve shows detection is essentially daily-level-only: narrow-window
anomalies (a single hour or 15-minute slot) are undetected up to 30%
magnitude. Conformal calibration clearly outperforms both a naive fixed-2σ
threshold and a standalone autoencoder. Cross-plant generalization is
asymmetric — plant_2's twin degrades badly when applied to plant_1, while
the reverse degrades more gracefully.
