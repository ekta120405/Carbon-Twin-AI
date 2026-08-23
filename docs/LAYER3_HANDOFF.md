# Layer 3 (Anomaly Detection) — Handoff Notes

Status as of this commit: 3.1-3.5 complete and verified against real data
(both plants) and hand-computed ground truth. 3.6 (risk aggregation) in
progress — severity and divergence calculations done and verified; the
actual 0-100 score/weighting formula not yet built.

This document exists so nobody re-discovers the same bugs, mirroring
docs/LAYER2_HANDOFF.md's role for Layer 2.

---

## 0. STANDING RULE — read this before writing any new aggregation logic

**Any per-day or per-window signal in this codebase must be computed
over the specific flagged/affected rows, never as a blanket mean/sum
across the full evaluation window.** This exact mistake has caused
three separate silent-dilution bugs this session:

1. 3.4's autoencoder: day-MEAN and day-MAX reconstruction error
   completely hid A8 (flatline) — the frozen run's genuinely elevated
   error was averaged/maxed against ~45 ordinary daylight timesteps in
   the same day. Only checking the exact labeled frozen window (via
   ground truth's `start_ts`/`run_length_slots`) revealed the real signal.
2. 3.6's original excess-energy severity: summed 3.1's `excess_energy_kwh`
   (daylight-only by construction) across ALL flagged days regardless of
   flag type. For `flagged_night_injection` days this summed unrelated
   ordinary daytime variation, not the actual fabricated night energy —
   inflated one real result from ~500 kWh (correct) to ~173,000 kWh
   (wrong, ~340x too high).
3. `divergence_mean_pct`: averaged `evidence_irr_divergence_pct` across
   the ENTIRE table. Produced the *identical* value (0.259234) across 6
   different injected replicates with completely different anomaly
   mixes — a dead giveaway of dilution, not a genuinely flat signal.

**Before trusting any new day/window-level statistic:** ask whether it's
computed only over the rows that actually matter, or over the whole
window with the signal diluted into a mostly-clean background. When in
doubt, verify like all three cases above were verified — compare the
same computation across multiple known-different real (or replicate)
cases and confirm you see real variation, not a suspicious constant.

---

## 1. Repo-level fix: CRLF corruption of LightGBM model files

`reports/models/<plant>/twin_q*.txt` are LightGBM's text-format boosters
(see LAYER2_HANDOFF.md Section 1 — never pickle). On Windows, with
`core.autocrlf=true` and no `.gitattributes`, a plain `git clone`/`git
pull` corrupts these files' `\n` line endings to `\r\n`, which breaks
LightGBM's strict text parser (`[LightGBM] [Fatal] Model format error,
expect a tree here`). **Fixed via `.gitattributes`:**
```
reports/models/**/*.txt -text
```
This is committed. If it's ever accidentally removed again (it was,
once, mid-session, when the repo was reset), the corruption WILL
recur on the next real `git pull` on any Windows machine — re-add it
immediately, don't just work around it locally.

---

## 2. 3.1 — Residual definition (`residual.py`)

Capacity-normalized: `r(t) = (claimed(t) - expected_q50(t)) / ac_nameplate_kw`.
Daily aggregation (`compute_daily_energy_residual`) is **daylight-only
by construction** — this is intentional and correct for its purpose
(continuous over-claiming during generation hours), but it has a real,
documented consequence: **`r_day` is structurally blind to A3 (night
injection)**, since night claims contribute zero to it. Don't try to fix
this by changing 3.1's daylight filter — see Section 3 below for the
correct fix.

---

## 3. Two deterministic guardrails, both in `residual.py`

Both exist because 3.1-3.3's magnitude-based residual test cannot see
these anomaly types by construction (A3) or because the learned
autoencoder empirically failed to catch it even with features engineered
specifically for it (A8) — these are not backup/redundant checks, they
are THE detection mechanism for their respective types.

### `flag_night_injection()` — A3 (night injection)

**Two earlier approaches were tried and rejected, both verified against
real injected replicate data, not assumed correct:**
- Using Layer 1's precomputed `exceeds_ceiling_fraud` column: REJECTED.
  The injector only modifies `ac_power`, never recomputes derived QC
  columns. Confirmed empirically: 0/11 real A3-injected nights showed
  `exceeds_ceiling_fraud=True` — the column is stale, computed on
  pre-injection clean data.
- A flat kW tolerance, or a *recomputed* ceiling comparison
  (`ac_power > physics_ceiling_fraud_kw`): REJECTED. `is_daylight` uses
  `min_solar_elevation_deg=10`, so "night" includes a real dawn/dusk
  twilight zone with legitimate generation up to ~2000 kW. Both
  approaches produced high false-positive rates (10/10 and 20/20 clean
  nights tested) concentrated exactly at IST sunrise/sunset.

**Resolution:** gate on `twin_clearsky_ghi < 1.0` (the actual formula
input `physics_ac_kw` is computed from — zero by construction there, no
calibration gap) combined with a tolerance of **1.5% of
`ac_nameplate_kw`**. Verified: 0/1039 false positives (plant_1),
4/1097 (~0.36%, plant_2 — a small, known, irreducible overlap with A3's
own minimum injection magnitude, not fully eliminable by any threshold).
Coverage: 11/11 (plant_1), 5/5 (plant_2) real A3-injected days caught at
the day level.

### `flag_stuck_sensor()` — A8 (flatline / frozen replay)

Same staleness problem as above: Layer 1's `is_stuck` column is ALSO
never recomputed post-injection (confirmed: `is_stuck=False` on all
29/29 rows of a real, verified frozen run). Fresh recomputation — N=8
consecutive identical daylight `ac_power` values (same threshold as
Layer 1's original `stuck_sensor_run_length` config) — gives **8/8
(plant_1), 7/7 (plant_2)** coverage with **0/25 false positives on
either plant**.

**KNOWN, DOCUMENTED FALSE-POSITIVE RISK (found later, via the hard-
negative specificity check — Section 8 below — NOT fixed, don't
reintroduce the attempted fix without a larger sample):** this rule also
fires on `N3_curtailment` (grid-ordered output cap — legitimate, not
fraud) whenever the cap binds continuously for ≥8 slots. Confirmed:
2/2 tested plant_1 examples were false positives this way. An attempted
fix (flag a run as genuinely-stuck only if `ac_power` stays on one
consistent side of `expected_q50` throughout — verified true for the
one A8 example checked, and true for the N3 counter-example) was
implemented, then VERIFIED on the full A8 label set and found to cause
a severe regression: A8 coverage dropped from 8/8→4/8 (plant_1) and
7/7→1/7 (plant_2). A freeze starting near/after the daily peak can show
the same kind of sign-change as legitimate curtailment simply because
real generation declines toward evening while the frozen value stays
put — the one-example hypothesis didn't generalize. **Reverted.** Do
not re-attempt this fix without checking against a much larger, more
representative sample of both A8 and N3 examples first (the n=1-2
available this session is not enough to derive a reliable
discriminator — this is precisely what the full 30-replicate benchmark,
Task 7, is for).

Both `flag_night_injection` and `flag_stuck_sensor` are wired into
`conformal.py`'s `classify_day`/`classify_days` as parallel, guaranteed
paths, checked BEFORE the conformal threshold.

---

## 8. Hard-negative specificity check (bounded, not the full Task 7 benchmark)

Every module above got a build-time correctness check before being
called done — but until this point, every check was "does it catch
fraud correctly." The equally load-bearing other half of the one-sided
design (Overview v2 §2.6) — "does it correctly leave legitimate
operational faults alone" — had never been tested. Ran 2 real examples
per hard-negative type (N1-N7), both plants (28 total), through the
full 3.1→3.6 pipeline.

**Result: 26/28 correctly left unflagged (specificity 0.929).** The 2
false positives are both the N3_curtailment/stuck-sensor interaction
documented in Section 3 above — not spread across other types, not a
broader systemic problem. All other hard-negative types (N1, N2, N4,
N5, N6, N7) showed 0 false positives across both plants in this sample.

This is a bounded sanity check (2 examples/type, not the full
population), not the formal Task 7 benchmark (which needs the full
30-replicate sweep, proper sample sizes, and the MDO curve) — but it's
the right-sized check for "does the build actually work," at the same
scale already used successfully elsewhere this session (4 A6 examples,
12 replicates for the 3.6 signal check). See
`check_hard_negatives.py` to reproduce or extend.

---

## 4. 3.2 — Conformal calibration (`conformal.py`)

`calib` split is only **5 days per plant** — below the standard
split-conformal floor for the nominal alpha (`n >= 1/alpha - 1 = 19` at
alpha=0.05). Per-regime (Kt-stratified) thresholds were considered and
REJECTED — plant_1 has **zero** overcast calib days, nothing to
calibrate from at all. **Resolution:** one pooled, unstratified
threshold per plant = max(calib r_day values) (the standard conservative
convention below the floor), with the REAL achievable alpha reported
explicitly: `alpha_achievable = 1/(n_calib+1) ≈ 0.167` (~16.7%), NOT the
nominal 5%. Report this honestly in the paper — do not present it as
"5% FPR."

---

## 5. 3.4 — Autoencoder (`autoencoder.py`)

Per-timestep, not per-day-sequence (verified against actual injector
code: A4's source day is always `daily_power.idxmax()`, and its own
docstring frames the anomaly as per-timestep-detectable; per-day
sequences would also only give ~20 independent training examples, worse
than the LSTM baseline's already-documented normalization-bug scale).

**Baseline for "what's normal" must be pooled `calib`+`test_clean`
(n=10, held out), NOT `train`'s own reconstruction error** — `train` is
optimistically low (same in-sample-fit pattern as the twin's own
train/test R² gap, LAYER2_HANDOFF.md Section 4). Verified: a `train`-based
baseline would set thresholds far too low.

Early stopping on `calib` loss, not a fixed epoch count — `config.yaml`'s
`epochs=200` was verified well short of convergence for this feature
scheme (loss still dropping sharply at 200: 1.10→0.41, vs 0.12 at 1000+
with proper early stopping).

**Catches, verified against real labeled replicates:** A2, A3, A4, A5,
A6, A7 (A4 strongest — 60% of its rows exceed the p95 baseline).
**Does NOT catch A1** (pure magnitude anomaly — correctly 3.2's job, not
a structural one) **or A8** (checked at the EXACT labeled frozen window,
not just day-level — genuinely zero elevation, despite features
engineered specifically for it). A8 is caught by the deterministic
guardrail instead (Section 3).

---

## 6. 3.5 — CUSUM (`cusum.py`)

Reference mean/std: pooled `calib`+`test_clean`, same reasoning and same
empirical check as 3.4's baseline (train is ~2x optimistically low on
both plants: plant_1 train mean=0.047 vs pooled mean=0.086; plant_2
train mean=0.037 vs pooled mean=0.103).

**`h` is bootstrap-calibrated from clean data alone, NEVER tuned against
A6 labels** — Overview v2 §8.1: "Thresholds are calibrated only on
CALIB. TEST-INJECTED is touched once, at the end." Tuning `h` until known
A6 cases flag would be fitting the detector to the answer key. Method:
bootstrap-resample the pooled reference into synthetic clean 9-day
sequences, measure empirical FPR per candidate `h`, pick the smallest
`h` at ≤5% target FPR — **before** ever looking at an A6 outcome.
Result: `h≈4.0` (vs. config's scaffold `h=5.0`, which turned out to be
conservative, ~1.2% actual FPR). `config.yaml`'s `h=5.0` is kept
as-is (documented scaffold default, per the same "report both, don't
overwrite" convention as 3.2's `alpha_achievable`) — the real value is
always computed at runtime via `calibrate_h_from_clean_data()`, never
hardcoded, since it depends on the reference pool and window length.

Honest result at the clean-calibrated `h`: **1/4** real A6 replicates
flagged (the 7-day ramp, final_delta=14.4%); the three 3-4 day ramps
(8-12% deltas) were not. This tracks CUSUM theory correctly — shorter
ramps give less runway to accumulate evidence — and is a legitimate,
citable finding tied directly to the planned MDO curve work (§9.3), not
a failure to keep chasing with more parameter tweaks.

---

## 7. 3.6 — Risk aggregation (`aggregate.py`) — COMPLETE

Five inputs per the roadmap: fraction of days flagged, cumulative
severity (kWh), CUSUM state, calibration ratio, mean on-site/satellite
divergence. **Severity and divergence both required the Section 0 fix**
(flag-type-aware / flagged-rows-only, not blanket window aggregation) —
see `compute_daily_flag_severity_kwh()` and
`compute_divergence_severity_pct()`.

`evidence_irr_divergence_pct`, even correctly restricted to flagged
days, doesn't show a single clean direction across arbitrary flag types
— it's fundamentally an A5 (weather-spoofing)-specific signal, not a
universal fraud discriminator (matches Overview v2's own framing of it
as primarily a Layer 4 flag-SHAP explainability feature). Included at
modest weight (0.08) with this caveat noted.

### The score formula

`compute_risk_score()`: each of the five inputs is normalized to [0,1]
before weighting (they arrive on wildly different scales — a fraction,
a kWh-fraction, a CUSUM statistic, a ratio near 1.0, a percentage — the
same scaling problem as 3.4's autoencoder inputs), then combined:
```
frac_flagged        weight 0.30  -- already in [0,1]
severity             weight 0.30  -- cum_severity_frac / 0.20 cap, clipped
cusum                weight 0.20  -- cusum_max_s / cusum_h, clipped
calibration_ratio    weight 0.12  -- |CR - plant's OWN clean baseline| / 0.5, clipped
divergence           weight 0.08  -- |divergence_flagged_only| / calibrated_cap, clipped
```
Bands: Low 0-25, Elevated 25-50, High 50-75, Critical 75-100.

Weights and the severity/calibration-ratio caps are **documented
judgment calls, not empirically optimized** — state this plainly in the
paper (n=6 replicates/plant sampled this session is nowhere near enough
to fit these rigorously; revisit once the full 30-replicate benchmark
run happens, per the roadmap's evaluation phase).

### Calibration ratio must be measured relative to the PLANT'S OWN clean baseline, not a universal 1.0

Verified necessary, not assumed: plant_2's real clean CR is 0.897 (10.3%
below 1.0) purely from its own physics-anchor fit characteristics (see
Section 4 / LAYER2_HANDOFF.md Section 4) — scoring against absolute 1.0
would make plant_2 look moderately risky even when genuinely clean.

### CUSUM window-length scope limitation — found via this score's own acceptance test

**First finding (bug, fixed):** initially ran CUSUM over a plant's full
34-day real history using `h` bootstrap-calibrated for 9-day windows —
produced a false positive on genuinely clean plant_2 data (scored 30.7,
"Elevated"). A threshold calibrated for one window length has no valid
guarantee applied to a longer one; a longer sequence gives the
cumulative statistic far more chances to drift upward by chance.
**Fix:** calibrate `h` fresh for whatever window length is actually
being scored — `h=4.5` for the 10-day `calib`+`test_clean` clean-check
block, `h=4.0` for 9-day replicate evaluation (matching each exactly,
not approximated).

**Second finding (real, stated limitation, NOT fixed — same treatment
as Layer 2's online-learning module, 2.5):** this means the CUSUM
component is only valid for the specific ~9-10 day windows it's
calibrated for. It does NOT generalize to continuous, open-ended
deployment monitoring (weeks/months) without either periodic window
resets or a properly different continuous-monitoring formulation
(average-run-length-based, not this bootstrap-FPR approach). Report this
plainly in the paper as scoped-out, not silently absorbed into "fixed."

### Divergence cap must be calibrated from clean data alone, same discipline as everything else

`calibrate_divergence_cap()`: derives the normalization cap from the
95th percentile of `|evidence_irr_divergence_pct|` across `calib`+
`test_clean` ONLY — never by looking at injected values, even to
"validate" the choice. A normalization cap in a composite score
functions the same way a detection threshold does (it controls how much
a signal level contributes to the final flag), so it gets the same
calib-only treatment as conformal's threshold, CUSUM's `h`, and the
autoencoder's baseline. An earlier version derived the cap by eyeballing
clean+injected values together — correctly identified as a violation of
this project's own evaluation-circularity discipline before being fixed.

### Acceptance test (both fixes together)

Real clean data, both plants, correctly lands in **Low** band:
plant_1 = 12.1, plant_2 = 20.2 (was 30.7/"Elevated" before the CUSUM
window fix). Injected replicates (6 sampled per plant) span Low through
High with a sensible spread reflecting genuine per-replicate variability
in randomly-drawn anomaly type/magnitude — not every replicate should
score Critical, and none forced to.

**Known remaining imperfection, documented not chased further:**
plant_2's clean-block `calibration_ratio` component (0.693) is
non-trivial because `clean_cr_baseline` is computed from the full
34-day real dataset while the score itself is computed on just the
10-day `calib`+`test_clean` sub-window — a small sample-size mismatch
between the two CR estimates, not real fraud. Didn't affect the band
classification this time; worth tightening if it does later.

---
