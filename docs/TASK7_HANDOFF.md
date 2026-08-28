# Task 7 (Full Evaluation) — Handoff Notes

Status as of this commit: **7.0-7.6 built, run at full scale (30 replicates
× 2 plants), and verified — every module's output cross-checked against
either an already-known result or an independent hand-verification before
being trusted.** 7.7 (paper figure/table export) intentionally NOT built —
the paper is being written separately; see `docs/TASK7_FINDINGS.md` for the
consolidated numbers instead of a paper-specific export step.

This document exists for the same reason LAYER2/3/4(1-5)/4(dashboard)_
HANDOFF.md exist: several real bugs and non-obvious decisions were found
while building this that aren't in `TASK7_FULL_ROADMAP_FOR_CLAUDE_CODE.md`.
For the actual result numbers, see `docs/TASK7_FINDINGS.md` — this document
is about *how* they were produced and what to watch for if this code is
ever touched again.

---

## 0. File locations

```
src/evaluation/harness.py              7.0 — shared batch-scoring harness, everything else calls this
src/evaluation/benchmark.py            7.1 — full precision/recall/F1/AUC-PR/specificity benchmark
src/evaluation/mdo_curve.py            7.2 — Minimum Detectable Overstatement curve
src/evaluation/leave_one_plant_out.py  7.3 — cross-twin generalization
src/evaluation/ablations.py            7.4 — ablations a,b,c,d,f,g (e explicitly not built)
src/evaluation/stuck_sensor_fix.py     7.5 — physics-ceiling discriminator re-attempt (reverted)
dashboard/app.py                       7.6 — "Accuracy & Testing" tab (see NOTE below)

reports/tables/task7_benchmark_full.json       7.1 output
reports/tables/mdo_curve.json                  7.2 output (+ reports/figures/mdo_curve.png, gitignored)
reports/tables/leave_one_plant_out.json        7.3 output
reports/tables/ablations.json                  7.4 output
reports/tables/stuck_sensor_fix_reattempt.json 7.5 output

docs/TASK7_FINDINGS.md                 consolidated result numbers, paper-reference-ready
```

**NOTE on 7.6, dashboard integration:** originally built as a 5-tab
`dashboard/pages/5_Evaluation.py` covering all of 7.1-7.5. The dashboard
was later redesigned (see `docs/LAYER4_DASHBOARD_HANDOFF.md` §0.1) into a
single-file, audience-facing app with plain-language labels throughout.
In that redesign, only 7.1's headline numbers (recall/specificity,
relabeled "fraud caught" / "legitimate cases correctly left alone") and
7.2's MDO curve survived onto the visible "Accuracy & Testing" tab — 7.3
(leave-one-plant-out), 7.4 (ablations), and 7.5 (the stuck-sensor fix
re-attempt) were deliberately dropped from the on-screen dashboard as
too much implementation detail for that audience. **This data is not
lost** — all three still live in their `reports/tables/*.json` outputs
and in `docs/TASK7_FINDINGS.md`/`docs/TASK7_HANDOFF.md` (this document),
just not surfaced in the demo UI.

**Nothing in `src/evaluation/` reimplements Layer 2/3/4 detection logic.**
Every module calls the real `src/layer{2,3,4}_*` functions, reusing
`dashboard.lib.pipeline.load_plant_base()` directly for every per-plant
calibrated quantity (conformal threshold, CUSUM reference, calibration-
ratio baseline, divergence cap). If a Task 7 number ever looks wrong, the
bug is almost certainly in the wiring inside `src/evaluation/`, not in the
underlying Layer 2/3/4 module — check there first.

---

## 1. The 7.0 harness — what every other module reuses

`score_replicate(plant, replicate_id, twin=None, base=None)` runs one
injected replicate through Layer 2 (physics anchor, twin prediction) and
Layer 3.1-3.3 (conformal + guardrail classification) + 3.6's flag-type-
aware severity, returning one row per day. It does NOT build carbon
exposure, severity ranking, or evidence bundles/SHAP — a 30×2-replicate
benchmark needs only classification + severity, and skipping 4.x's
explanation overhead keeps a full run to a few seconds instead of minutes.

**Definition-of-done check, passed before anything else was trusted:**
`score_replicate('plant_1', 10)` and `score_replicate('plant_2', 0)`
reproduce the EXACT flag compositions documented in
`LAYER4_MODULES_1_TO_5_HANDOFF.md` §0 (3× overclaim + 1× stuck-sensor;
2× night-injection + 1× overclaim, respectively). If this ever stops
matching after an edit to `harness.py`, something broke — don't proceed to
trust any downstream module's output until this passes again.

**CUSUM's `h` is recalibrated per distinct window length, cached
in-process per `(id(base), window_len)`** — every injected replicate is a
9-day window, so this amounts to one real calibration per plant, not 30.
Never reuse an `h` calibrated for one window length against a different
one (LAYER3_HANDOFF.md §7's own documented bug, already fixed once for the
dashboard — don't reintroduce it here).

---

## 2. Real bugs found and fixed while building Task 7 (read before editing)

### 2.1 MDO curve: baseline false-positive contamination (7.2)

The raw MDO sweep initially showed a flat 0.1 detection rate across EVERY
tested magnitude for the narrow (15min/hourly) windows — the same "a
constant value across different scenarios is a red flag for dilution/
contamination, not a genuine flat signal" pattern this project has hit
three times before in Layer 3 (LAYER3_HANDOFF.md §0). Root cause:
`plant_2`'s real `test_clean` day `2020-06-08` is ALREADY a false positive
at ZERO injection (`r_day=0.248 > threshold=0.208` with no modification at
all) — the same documented plant_2 night-injection guardrail overlap from
LAYER3_HANDOFF.md §3. Injecting a magnitude into an already-flagged day
tests nothing about that magnitude.

**Fix:** `run_mdo_sweep()` computes each day's baseline (zero-injection)
r_day first and EXCLUDES any day already above threshold from the sweep
pool, reporting the exclusion explicitly in the output JSON
(`excluded_baseline_false_positives`) rather than silently dropping it.
n dropped from 10 to 9 held-out days/cell as a result — this is correct,
not a bug in the exclusion logic.

**If you ever re-run the injector or change the conformal threshold:**
re-check whether this exclusion still applies to the same day, or whether
a *different* day now needs excluding — don't assume `2020-06-08` is
permanently special, it's just the specific day this session's real data
happened to produce a baseline false positive on.

### 2.2 Leave-one-plant-out: cross-twin capacity de-normalization (7.3)

`PIGBQRTwin.predict()` computes
`preds[q] = physics_ac_kw.values + y_hat * self.ac_nameplate_kw` — using
the twin's OWN TRAINING-TIME capacity, not whichever plant is being
scored. Naively calling `plant_1_twin.predict(X_plant_2, ...)` would
silently rescale the learned correction by plant_1's capacity (27,986 kW)
instead of plant_2's (23,579 kW) — a real, if modest (~19% for this
project's two plants specifically), unit error, not a genuine test of
whether the twin's learned pattern generalizes.

**Fix:** `_twin_rescaled_for_plant()` shallow-copies the twin
(`copy.copy`, never mutating the cached original — it's shared via
`st.cache_resource` across every other module) and overrides
`ac_nameplate_kw` to the SCORED plant's own capacity before prediction.
`models_` and `feature_cols` are left untouched (confirmed both plants'
twins share identical `feature_cols` before relying on this).

**If this project ever adds a third plant, or the two plants' capacities
diverge much further (e.g. 10x, as the raw per-inverter DC magnitudes
already do per Layer 1 notes):** re-verify this override still produces
sane numbers — a 10x capacity mismatch left unfixed would produce a
correction 10x too large or too small, not just "somewhat off."

### 2.3 Ablation (a): the R1 leakage assertion would block on-site training (7.4)

`PIGBQRTwin.fit()` hard-asserts (`assert_twin_feature_safety`,
TC-L2-01) that its feature matrix contains ONLY `twin_*`-prefixed columns
and NEVER `evidence_*`/`ac_power` — Layer 2's core leakage guard. Ablation
(a) needs to train a variant twin on ON-SITE evidence columns specifically
(`evidence_onsite_irradiation` etc.), which the assertion would reject
outright.

**Resolution, NOT a weakening of the check:** the on-site columns are
RENAMED to `twin_`-prefixed aliases (`twin_onsite_irr` etc.) before being
handed to the real, unmodified `PIGBQRTwin.fit()`/`.predict()`. This
reuses the exact production training code path (same LightGBM config,
same monotone-constraint quantile setup including the q50-must-use-L2
gotcha from LAYER2_HANDOFF.md §2) instead of a second, independently
bug-prone reimplementation, while the assertion still passes honestly —
on-site evidence genuinely IS the intended input for this one isolated
model. **This ablation model is never saved, never used by
`score_replicate`/harness, never touched by anything outside
`train_onsite_variant()`'s caller.** If you ever see `evidence_*` data
reaching a `PIGBQRTwin` anywhere else in the codebase, that IS a real bug
— this is the only sanctioned exception, and it's confined to one function.

### 2.4 Benchmark: CUSUM is architecturally separate from `classify_days()` (7.1)

Discovered while sanity-checking A6_slow_drift's suspiciously low recall
(0.056 — looked like a possible bug at first). `conformal.py`'s
`classify_day()` never consults CUSUM's flag at all — CUSUM only feeds
Layer 3.6's aggregate risk score as one of five components. But Overview
v2 §3.5 explicitly frames CUSUM as the INTENDED detection mechanism for
A6 (persistent low-magnitude drift too small to trip any single-day
threshold). Reporting only the production `classify_days()` flag would
silently understate A6 recall by omitting the one mechanism designed to
catch it.

**Resolution:** every metrics function in `benchmark.py` (and the ablation
functions that build on it) takes a `detected_col` parameter and is
computed under BOTH `detected` (production) and `detected_with_cusum`
definitions, both always reported, never just one. Including CUSUM only
moves A6 recall from 1/18 to 2/18 at full scale (consistent with the
earlier bounded check's 1/4) — CUSUM is a real but weak contributor to A6
detection, not a fix for the underlying difficulty.

### 2.5 Two "suspiciously identical" specificity values, both investigated and both confirmed non-bugs

The standing rule (§5 below, LAYER3_HANDOFF.md §0) is that any suspicious
constant/identical value gets investigated before being trusted, never
assumed. Two such cases came up in Task 7, and both were traced to the
same root cause:

**7.3 (leave-one-plant-out), plant_2:** hard-negative specificity is
*exactly* 0.897 whether plant_2 is scored with its own twin or with
plant_1's. Traced by joining both runs' hard-negative rows on
`(replicate, date)`: `r_day` genuinely differs between the two twins (not
a caching bug — confirmed `own_twin is not cross_twin` and
`own_twin.models_ is not cross_twin.models_`), but every flagged
hard-negative day is `2020-06-08`, caught by the deterministic
night-injection guardrail — which depends only on `twin_clearsky_ghi` and
`ac_power`, never on the twin's prediction. Guardrail-driven flags are
twin-invariant by construction, so identical specificity here is
guaranteed, not coincidental.

**7.4 ablation (c), both plants:** conformal and fixed-2σ thresholds also
produce *exactly* identical hard-negative specificity (0.920 pooled; 0.943
plant_1, 0.897 plant_2). Investigated the same way: joined both runs' hard-
negative rows and printed every flagged one. **Every single hard-negative
false positive under BOTH thresholds is guardrail-driven**
(`flagged_stuck_sensor` for plant_1's 5 false positives, all real
`N3_curtailment` days at the documented collision; `flagged_night_injection`
for plant_2's 9, all `2020-06-08` across different replicates) — `r_day`
is identical between the two threshold runs for every one of these rows,
and in several plant_2 cases `r_day` (e.g. 0.248) doesn't even exceed the
*fixed-2σ* threshold (0.267) on its own, confirming the guardrail — not
either magnitude threshold — is what's actually flagging them. Zero
hard-negative false positives in this sample came from the magnitude test
at all, on either plant, so switching between conformal and fixed-2σ
literally cannot move hard-negative specificity here. (Precision/recall
still differ meaningfully between the two, §3 below — this identical value
is specific to the hard-negative/specificity slice, not the whole
comparison.)

Reproduce either check by joining `merge_with_ground_truth()`'s output for
the two runs being compared on `(replicate, date)` and filtering to
`is_hard_negative & detected` — this is the fastest way to confirm whether
an identical-looking number is guardrail-driven before trusting it.

---

## 3. Design decisions worth knowing (not bugs, but non-obvious)

- **"Window" in the MDO curve (7.2) is NOT a separate statistical test** —
  this project has exactly one decision point, the daily conformal test.
  "15min"/"hourly"/"daily" windows control how much of a day's timesteps
  get the injected magnitude applied before that single daily test runs.
  Don't let a future edit reinterpret this as "we need a 15-min-level
  conformal calibration" — that's a different, much larger project change.
- **Ablation (d)'s one-sided/two-sided comparison deliberately excludes
  the guardrails** (night-injection, stuck-sensor) to isolate exactly the
  conformal magnitude test's sidedness. Precisely what this costs/gains
  vs. 7.1's production numbers (pooled: precision 0.801, recall 0.585,
  specificity 0.920): one-sided-only gives precision 0.812 (marginally
  *higher*, not lower), recall 0.425 (meaningfully *lower* — the
  guardrails contribute A3/A8's perfect-recall detections, which this
  isolated comparison doesn't get credit for), and specificity 0.971
  (*higher* — removing the guardrails also removes their hard-negative
  false positives, e.g. the stuck-sensor/N3 collision). Removing
  guardrails costs recall specifically, not precision or specificity —
  don't describe this as a blanket "lower numbers," the table doesn't
  support that for two of the three metrics.
- **Ablation (f)'s "conformal" arm is the FULL production flag (including
  guardrails)**, not the isolated conformal-only definition from (d) — a
  deliberately different scope than (d), because (f) is comparing
  DEPLOYABLE detector configurations (what would you actually run: the
  real system, an autoencoder alone, or both together?), not isolating
  one mechanism's sidedness. Don't assume these two ablations' "conformal"
  numbers should match — they're answering different questions.
- **Per-type precision (7.1) uses ONE shared clean+hard-negative false-
  positive pool across every anomaly type**, not a per-type-specific FP
  count — because false positives have no injected type of their own. This
  is stated in `benchmark.py`'s own `METHODOLOGY_NOTE` and repeated in the
  output JSON; read `precision_i` as "precision vs. non-fraud days for
  this type specifically," not a fully independent statistic.
- **The autoencoder (7.4f) was trained for the first time ever in this
  project's history** — confirmed via `grep` before writing any code that
  no training driver or persisted model existed anywhere in the committed
  codebase, despite LAYER3_HANDOFF.md discussing its behavior in detail
  (that discussion was from an earlier, uncommitted throwaway verification
  run). If its numbers ever need reproducing, `train_plant_autoencoder()`
  in `ablations.py` is the only place this logic lives. **Exact threshold
  definition used (stated explicitly, since this choice is doing real work
  in the 0.246 specificity result):** `build_autoencoder_features()`
  produces PER-TIMESTEP reconstruction error (never day-mean/day-max —
  LAYER3_HANDOFF.md §5 already found day-mean/day-max hides localized
  anomalies entirely). The model trains on `train`-split features only;
  the threshold is the 95th percentile of reconstruction error over the
  pooled, held-out `calib`+`test_clean` features (`compute_reconstruction_
  error()` on `train_autoencoder()`'s fitted scaler/model) — the same p95-
  of-clean-reference convention as `aggregate.py`'s `calibrate_divergence_
  cap()`. A day is flagged if **ANY** daylight timestep's error exceeds
  that threshold (`score_replicate_autoencoder()`'s
  `.groupby(dates).any()`) — a single elevated 15-minute reading is enough
  to flag the whole day, which is a much more permissive trigger than the
  daily-*aggregate* conformal test. This "any single timestep" rule is the
  most likely driver of the low specificity (0.246) — a stricter rule
  (e.g. requiring N consecutive elevated timesteps, mirroring the stuck-
  sensor guardrail's own run-length logic) was NOT tried and could
  plausibly change this result; it just wasn't part of this task's scope.
  Don't read "autoencoder is a bad detector" as fully threshold-
  definition-independent — it's a finding about THIS threshold choice
  specifically, consistent with the architecture's framing of the
  autoencoder as a secondary signal either way.
- **Leave-one-plant-out's AUC-PR can rise even when precision/specificity
  collapse (7.3) — not a contradiction.** plant_1 scored with plant_2's
  twin: AUC-PR 0.760 (own-twin: 0.718) despite precision crashing from
  0.915 to 0.438 and specificity from 0.943 to 0.307. AUC-PR is
  threshold-independent (it integrates precision across every possible
  cutoff of the continuous `severity_kwh` ranking score); precision/
  specificity are read at the ONE operating threshold `classify_days()`
  actually uses. Verified this isn't a fluke of the metric, not just
  asserted: recomputing `average_precision_score` directly reproduces
  both values exactly, and mean `severity_kwh` is higher for genuinely
  positive days than negative days under BOTH twins (own: 32,012 vs. 257;
  cross: 77,409 vs. 44,567) — the cross-twin's underlying ranking still
  has real, if noisier, separating power even though its one specific
  classification threshold is badly miscalibrated for this plant. A
  badly-calibrated absolute threshold and a reasonably well-ordered
  continuous score can genuinely coexist.
- **Ablations (a) and (b) both report plant_2's PI-GBQR R²=0.6808 from the
  SAME pre-existing file** (`reports/tables/plant_2_twin_metrics.json`),
  **not two independent computations that happen to agree.** Confirmed via
  `git log` on that file: its only commit is Layer 2's original build
  (`68b54ad`), untouched by anything in Task 7 — ablation (b) explicitly
  reads it without recomputing (by design, per the roadmap), and ablation
  (a) reuses the same file as its "satellite" comparison baseline for the
  freshly-trained on-site variant. The two numbers matching is a sourcing
  fact, not a validation signal. Separately: 0.6808 (pooled/overall) is
  legitimately higher than either regime-specific test R² documented
  elsewhere (clear 0.307, partly-cloudy 0.140, both matching LAYER2_
  HANDOFF.md's own numbers) because R² does not decompose as a simple
  n-weighted average of per-regime R² values — it depends on residual sum
  of squares relative to the GLOBAL variance around the global mean, and
  the overcast regime (n=239 of 480, R²=0.749, near-zero-power near-zero-
  variance) contributes very little absolute error while the clear/
  partly-cloudy regimes (n=137, n=104) contain most of the real variance
  the model is fit against. This is a standard property of R², not a
  computation error.

---

## 4. What was explicitly NOT built, and why

- **7.7 (paper figure/table export)** — the paper is being written
  separately; `docs/TASK7_FINDINGS.md` holds the consolidated numbers
  instead of a paper-specific export script.
- **Ablation (e), online vs. batch twin** — per the roadmap's own §0.1
  scope decision, reported as text (`ablations.json`'s
  `e_online_vs_batch` key) rather than built, per the pre-existing
  `LAYER2_HANDOFF.md` §8 reasoning (34-day span gives a 20-day window only
  14 possible advance positions).
- **weekly/monthly MDO windows** — infeasible from the frozen injector's
  9-day eligible pool, reported as an explicit infeasibility finding
  (`mdo_curve.json`'s `weekly_monthly_infeasibility_note`), not silently
  dropped.
- **The stuck-sensor/N3-curtailment fix (7.5)** — re-attempted with a
  new, better-motivated idea (physics-ceiling discriminator) at full
  scale, found to give zero benefit, reverted. `residual.py` is
  unchanged. See `docs/LAYER3_HANDOFF.md` §3's updated entry for the full
  reasoning — this is now a twice-attempted, twice-rejected limitation,
  not a single inconclusive try.

---

## 5. Testing discipline followed (same standard as every prior layer)

- Every module's "Definition of done" check ran BEFORE trusting its
  output at full scale — 7.0 against known replicate compositions, 7.1
  against the earlier bounded specificity check, 7.3/7.4(c) by manually
  tracing why hard-negative specificity was suspiciously IDENTICAL across
  two different scoring variants in each case (both confirmed non-bugs: a
  twin-independent guardrail flag driving both, not a caching error — see
  §2.5 for both investigations in full).
- Every suspicious constant/flat result was investigated before being
  trusted, per LAYER3_HANDOFF.md §0's standing rule — this caught the MDO
  baseline-contamination bug (§2.1) and both identical-specificity cases
  in §2.5.
- `n` is reported alongside every rate/percentage in every output JSON,
  no exceptions, per the roadmap's own §1.8.
- No Layer 3 calibrated quantity (conformal threshold, CUSUM reference,
  calibration-ratio baseline, divergence cap) was ever re-tuned against
  injected/ground-truth data anywhere in Task 7 — confirmed by construction,
  since every module reuses `load_plant_base()`'s already-calibrated
  values rather than recomputing them.
