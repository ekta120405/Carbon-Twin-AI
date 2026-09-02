# CarbonTwinAI — Detailed Project Flow

This document is written so that someone with **zero prior context** on
this project can read it start to finish and understand exactly what the
system does, how every piece works, what data it uses, how it was tested,
and what the results are — down to specific function names, file paths,
and real numbers pulled from the actual repository, not paraphrased.

Where a claim can be verified against a specific file or a specific
`reports/tables/*.json` output, that path is given so it can be checked
directly. This document doesn't introduce anything new — it consolidates
and cross-references what's spread across `docs/CarbonTwinAI_Project_Overview_v2.md`
and the per-layer `*_HANDOFF.md` files, at a level of detail meant to be
self-contained.

---

## Part 1 — What this project is, in one page

### 1.1 The problem

A solar power plant earns carbon credits based on how much clean
electricity it *says* it generated. That claimed number comes from the
plant itself. If a plant exaggerates its output — even slightly — it gets
paid for carbon credits it didn't actually earn. Checking the plant's own
sensors more carefully doesn't fix this, because the plant controls those
sensors too.

### 1.2 The fix

Compare the plant's claim against something the plant has **zero ability
to influence**: independent satellite weather data, plus the basic
physics of how solar panels convert sunlight into electricity. If the
weather says a cloudy day should have produced modest power, but the
plant claims a sunny day's worth of output, that mismatch is the signal —
regardless of what the plant's own sensors say.

### 1.3 The one-line formal definition

> CarbonTwinAI estimates what a solar project's generation *should have
> been* from weather data the operator cannot manipulate, tests the
> project's self-reported generation against that estimate using a
> statistically calibrated one-sided decision rule, and explains,
> quantifies, and prices every flag it raises in tonnes of CO₂e.

Three components, always in this order:
1. **An estimator** — expected generation ĝ(t), with uncertainty bounds, built ONLY from independent weather.
2. **A test statistic** — how far the claim sits above the estimate, normalized by plant capacity.
3. **A decision rule** — a calibrated statistical threshold, plus deterministic physical-impossibility checks.

### 1.4 "Nowcasting," not "forecasting" — and why this distinction matters

The system does **not** predict the future. NASA POWER weather data for a
given day is already measured and already known by the time the system
runs (with roughly a one-week latency for that data to become reliably
available — the honest product claim is "T+1 week verification," not
real-time). The system estimates what generation *should have been*,
given weather that has *already happened* — a counterfactual estimate,
not a forecast. This distinction changes which model families are
appropriate and which accuracy metrics make sense; treating it as a
forecasting problem would be the wrong framing.

### 1.5 The governing statistical constraint

Detection power is bounded by how noisy the twin's residual is. A
systematic over-claim of magnitude δ is only detectable at aggregation
window length n when:

```
δ  >  z_α · σ / sqrt(n_eff)
```

where `n_eff` is much smaller than n because solar residuals are strongly
autocorrelated. Consequence: **instantaneous detection of small
over-claims is physically impossible; aggregation to the daily level is
mandatory.** Quantifying exactly how much fraud is catchable at what
timescale is one of this project's actual deliverables (the MDO curve,
Part 6).

---

## Part 2 — Architecture: four layers, strict order

```
                 ┌──────────────────────┐   ┌──────────────────────┐
   AUDITED  ───► │  Claimed generation  │   │ Independent weather  │ ◄─── TRUSTED
                 │  + on-site sensors   │   │ NASA POWER (1° grid) │
                 └──────────┬───────────┘   └──────────┬───────────┘
                            │                          │
              ══════════════╪══════════════════════════╪═══════════════
              LAYER 1       ▼                          ▼
              acquisition · timezone contract · CSI-preserving downscaling
              physics-ceiling QC · feature engineering · capacity estimation
              contiguous day-block splits (train/calib/test_clean/test_injected)
              ═══════════════════════════╪═══════════════════════════════
              LAYER 2                    ▼
              physics anchor (pvlib/PVWatts)         ──┐
              PI-GBQR quantile twin (weather → power)  ├─► expected(t), q10..q90
              robust training + calibration ratio      │
              ═══════════════════════════╪═══════════════════════════════
              LAYER 3                    ▼
              capacity-normalised residual · conformal calibration
              one-sided flagging · night/stuck-sensor guardrails
              autoencoder structural scorer · CUSUM sequential drift
              secondary-evidence escalation · risk aggregation
              ═══════════════════════════╪═══════════════════════════════
              LAYER 4                    ▼
              twin-SHAP · flag-SHAP · carbon accounting
              evidence bundle · severity ranking · dashboard
              ═══════════════════════════════════════════════════════════
```

**The one invariant enforced everywhere**: nothing derived from claimed
generation or on-site sensors may ever enter Layer 2's feature matrix.
This is checked by an automated test (`TC-L2-01`), not just a convention —
if the twin could see the number it's supposed to be checking, it could
learn to agree with it, and the whole exercise would be pointless.

---

## Part 3 — Layer 1: Data

### 3.1 Data sources and trust levels

| Stream | Source | Role | Resolution | Trust |
|---|---|---|---|---|
| Claimed generation | Kaggle `anikannal/solar-power-generation-data`, `Plant_{1,2}_Generation_Data.csv` | The thing being audited | 15-min, 22 inverters/plant, 34 days (May–Jun 2020) | **Untrusted** |
| On-site sensors | `Plant_{1,2}_Weather_Sensor_Data.csv` | Evidence only, never fed to the twin | 15-min, 1 sensor/plant | **Untrusted** |
| Independent weather | NASA POWER API (hourly) | Twin input | Solar 1°×1° (CERES SYN1deg), meteo 0.5°×0.625° (MERRA-2) | **Trusted** |
| Solar geometry | `pvlib` (computed, not downloaded) | Clear-sky ceiling, solar position | 15-min, exact | **Deterministic** |
| Grid emission factor | CEA India CO₂ Baseline Database v21.0 (Nov 2025) | kWh → tCO₂e | Annual constant | **Reference** |

Known, disclosed constraints: no coordinates are published for the
plants (a sensitivity analysis was run across the assumed grid cell and
its neighbours); NASA POWER is a ~110 km grid-cell average, not a point
measurement, so accuracy degrades under convective cloud (exactly the
May–June monsoon-onset regime this dataset covers); Plant 2's DC power
magnitudes differ from Plant 1's by roughly an order of magnitude
(inverter-level vs. string-level scaling — normalized per plant before
any modeling); the dataset is 34 days, one season, two sites — no
seasonal generalization is claimable, and this is stated plainly rather
than hidden.

### 3.2 Module: timezone contract (`src/layer1_data/acquisition.py`)

Kaggle timestamps are naive local time (IST, UTC+05:30); NASA POWER
hourly data is UTC. A 5.5-hour misalignment would make the system predict
night-time generation and flag every daylight hour. **Contract**: every
timestamp is stored tz-aware UTC, localized exactly once at ingestion. A
validation assertion checks that the plant's empirical daily generation
peak falls within ±90 minutes of computed solar noon; the pipeline aborts
if it doesn't.

### 3.3 Module: clear-sky-index-preserving downscaling (`src/layer1_data/downscale.py`)

NASA POWER is hourly; the system needs 15-minute resolution. Naive
forward-fill or linear interpolation on irradiance directly would flatten
the diurnal shape. Instead: (1) compute 15-minute clear-sky GHI via
`pvlib` (carries exact solar geometry), (2) compute hourly clear-sky
index `Kt = GHI_nasa / GHI_clearsky_hourly`, (3) linearly interpolate `Kt`
to 15 minutes (cloud opacity varies smoothly; solar geometry does not),
(4) reconstruct `GHI_15min = Kt_interp × GHI_clearsky_15min`.

### 3.4 Module: physics ceiling QC (`src/layer1_data/quality.py`)

Defines what "clean" means, physically rather than by assumption:

```
P_ceiling(t) = P_dc_nameplate × (GHI_clearsky(t)/1000) × η_max × (1 − γ·(T_cell(t) − 25)) × 1.05
```

Rows where `claimed(t) > P_ceiling(t)` are physically impossible — excluded
from twin training and flagged at maximum severity in Layer 3. Two
tolerances exist in `config.yaml`'s `layer1.quality`: `ceiling_tolerance_training
= 1.32` (generous, training exclusion only — recalibrated from an observed
+27–28% NASA-vs-onsite divergence) and `ceiling_tolerance_fraud = 1.05`
(strict, reserved for Layer 3's own gross-overshoot flags). Also removed:
night-time rows outside `min_solar_elevation_deg = 10°`, stuck-sensor runs
(identical value ≥ 8 consecutive daylight timestamps), and negative power.

### 3.5 Module: feature engineering (`src/layer1_data/features.py`)

**Twin features** (independent only, `twin_*` prefix): GHI, DNI, DHI,
clear-sky GHI, clear-sky index Kt, T2M, RH2M, WS2M, surface pressure,
precipitation, solar zenith/azimuth/elevation, air mass, extraterrestrial
irradiance, POA irradiance (via `pvlib` transposition), estimated cell
temperature, hour-of-day sin/cos, day-of-year sin/cos, short GHI/Kt lags.

**Evidence features** (`evidence_*` prefix, Layer 4 only, never Layer 2):
on-site irradiation, on-site ambient/module temp, and divergence terms
`evidence_irr_divergence_pct` / `evidence_temp_divergence` — a large
positive divergence is the fingerprint of on-site sensor tampering.

### 3.6 Module: capacity estimation (`src/layer1_data/capacity.py`)

Nameplate capacity isn't published. Estimated per plant as the 99.5th
percentile of plant-level summed AC power over `Kt > 0.9` timestamps, then
frozen — the entire Layer 3 metric depends on this constant staying fixed.

### 3.7 The four-bucket split protocol (`src/layer1_data/splits.py`)

Contiguous day-blocks, **never shuffled** (shuffling 15-minute data would
leak information across the split boundary):

```python
assign_day_blocks(train_days=(1,20), calib_days=(21,25), test_clean_days=(26,30), test_injected_days=(31,34))
```

| Bucket | Days | Used for | Must NEVER be used for |
|---|---|---|---|
| `train` | 1–20 | Teaching the twin normal weather→power behavior | Deciding any threshold |
| `calib` | 21–25 | Setting the conformal detection threshold | Training the twin |
| `test_clean` | 26–30 | Confirming the system stays quiet on real, honest, unseen data | Training or calibrating anything |
| `test_injected` | 31–34 | Raw material (+ reused `test_clean` days) for the 30 synthetic test scenarios | Training or calibrating anything |

`calib` (5) + `test_clean` (5) = 10 real clean days that never touch
`test_injected` — this pooled set is used later for reference statistics
(never for the decision threshold itself; see Part 5.3).

**Why this exact split can't just be rebalanced** (e.g. "give calibration
more days"): `train`'s 20 days are already barely enough to train the
twin without overfitting (Part 4.4); the frozen injector (`injector-v1.0`,
Part 8) is tagged against these exact boundaries, and changing them would
require re-tagging and re-verifying it — which the project's own
evaluation-circularity discipline (Part 6) treats as out of scope for a
mid-stream rebalancing.

**Layer 1 output contract**: one tz-aware, UTC-indexed table per plant:
`[timestamp, independent_features…, evidence_features…, claimed_ac_power,
P_ceiling, is_daylight, is_clean, split_tag]`.

---

## Part 4 — Layer 2: The digital twin

### 4.1 Two sub-models, always combined in this order

**(1) Physics anchor** (`src/layer2_twin/physics_anchor.py`, `pvlib`/PVWatts) —
zero-training-data, deterministic formula: sun position → clear-sky
brightness → panel temperature effect → expected AC output. Three roles:
defines the physics ceiling (Part 3.4), gives every flag a model-free
second opinion, and provides the calibration-ratio diagnostic (4.5).

**(2) PI-GBQR** ("Physics-Informed Gradient-Boosted Quantile Regression",
`src/layer2_twin/twin_model.py`) — the one learned model actually used in
production. It does **not** predict power from scratch; it learns the
**correction on top of the physics anchor**:
`y = (claimed − physics_anchor) / ac_nameplate_kw`. LightGBM, three
separate boosters for q10/q50/q90 quantiles (`reports/models/<plant>/twin_q{10,50,90}.txt`,
saved via `Booster.save_model()` text format — never pickled, for
version stability across machines/LightGBM releases).

### 4.2 Two mandatory properties that shaped model selection

Per the project's own design requirement (stated *before* any benchmark
comparison was run — Part 6.1 covers the benchmark that validated this
choice afterward): the twin must output **quantiles** (q10/q50/q90, not
one number — Layer 3 needs the uncertainty spread) and support **fast,
exact explanations** (gradient-boosted trees give this natively via
TreeSHAP; a deep model would need a slower, approximate explainer). These
two requirements, not raw accuracy alone, are why PI-GBQR was the
starting choice.

### 4.3 A hard-coded physical-plausibility rule

The model is mathematically forbidden from predicting that power goes
*down* when sunlight goes *up*, or *up* when panels get hotter (heat
reduces efficiency). Enforced via LightGBM's `monotone_constraints`:
`poa_global: 1`, `clearsky_index: 1`, `temp_cell: -1` (`config.yaml`).

**A real, load-bearing gotcha**: LightGBM's monotone-constraint mechanism
rejects any piecewise-linear-gradient objective — confirmed for both
`"quantile"` (any alpha) and `"regression_l1"` (MAE). Only `"regression"`
(L2/MSE, smooth Hessian) is accepted alongside a monotone constraint.
Consequence: `twin_q50.txt` is trained with `objective="regression"` — a
monotone-constrained **conditional mean**, not a true median regressor.
`twin_q10`/`twin_q90` are genuine unconstrained quantile models. The
dashboard and every doc call this the "central estimate," never "median,"
for exactly this reason.

### 4.4 Overfitting — resolved for Plant 1, honestly unresolved for Plant 2

Plant 1: real overfitting was found and largely resolved via tuned
hyperparameters (`num_leaves` 15→7, `n_estimators` 600→80, `lambda_l2`
5→10). Plant 2: clear-regime train R²=0.789 vs. test R²=0.307 (gap 0.48)
and partly-cloudy train R²=0.740 vs. test R²=0.140 (gap 0.60) remain
**confirmed, unresolved** — the twin actually underperforms the pure
physics-only baseline in both regimes there. A calib-split fallback
selector (twin vs. physics-only, per plant/regime) was considered and
explicitly rejected — it would contaminate Layer 3's conformal
calibration with a selection decision made on the same split, and
`calib_days` is only 5 days, too little to support yet another decision.

### 4.5 The overcast fallback

Only 9–12 training rows exist per plant at `Kt < 0.4` — not enough to
trust any learned correction. Below this threshold, the twin is bypassed
entirely; the system uses the physics anchor alone. `kt_switch = 0.4`
deliberately matches `config.yaml`'s `layer3.conformal.csi_bins[1]`
boundary — Layer 2's fallback and Layer 3's regime stratification must
agree on where "overcast" starts, or the two layers silently disagree.

### 4.6 Calibration ratio diagnostic

`CR = median(claimed) / median(physics_anchor)` over clear-sky
timestamps. A plant whose claims can only be fitted by assuming an
implausible system efficiency is itself evidence of systematic inflation,
independent of any single-day test. Real clean values: Plant 1 ≈ **1.028**,
Plant 2 ≈ **0.897** — Plant 2's own physics-anchor fit characteristics put
it 10.3% below 1.0 even when genuinely clean, which is why the risk score
(Part 5.7) measures deviation from **each plant's own** clean baseline,
never from a universal 1.0.

### 4.7 Online learning — scoped, deliberately not built

`window_days: 20` equals `train_days` exactly — a rolling window isn't
smaller than the static twin's own training set, just sliding. With 34
total days and days 21–34 reserved, a 20-day window can advance across at
most 14 distinct positions — too short a horizon for genuine concept
drift to manifest. Combined with Plant 2's already-documented overfitting
fragility, the marginal evidentiary value of building and debugging this
was judged low. Reported as an explicit scope decision in the paper, not
silently dropped.

**Layer 2 output contract**: `[timestamp, expected_ac, q10, q90,
physics_anchor, calibration_ratio, claimed_ac]`.

---

## Part 5 — Layer 3: Anomaly detection

### 5.1 The residual (`src/layer3_anomaly/residual.py`)

```
r(t) = (claimed(t) − expected_q50(t)) / P_ac_nameplate
```

Capacity-normalized (not the ratio-to-expected form, which divides by
~0 at dawn/dusk). Aggregated to **daily energy** — the primary decision
level — using only daylight rows:

```
R_day(d) = Σ claimed(t)·Δt − Σ expected(t)·Δt        # kWh
r_day(d) = R_day(d) / (P_ac_nameplate × daylight_hours(d))
```

**Structural blind spot, by design, not oversight**: `r_day` is daylight-only,
so it is structurally blind to night-time fabrication — this is exactly
why a separate guardrail exists for that case (5.2).

### 5.2 Two deterministic guardrails, checked *before* any statistical test

**`flag_night_injection()`** — targets A3. Two earlier approaches were
tried and rejected on real data: reusing Layer 1's precomputed
`exceeds_ceiling_fraud` column (stale — the injector never recomputes
derived QC columns after modifying `ac_power`; confirmed 0/11 real
A3-injected nights showed it as True), and a flat kW tolerance or
recomputed-ceiling comparison (both produced high false positives at real
IST sunrise/sunset, since `is_daylight`'s 10° elevation cutoff includes a
real twilight zone with legitimate generation up to ~2000 kW). **Resolution**:
gate on `twin_clearsky_ghi < 1.0 W/m²` (the actual PVWatts formula input —
zero by construction there, no calibration gap) combined with a tolerance
of 1.5% of nameplate capacity. Verified: 0/1039 false positives (Plant 1),
4/1097 ≈ 0.36% (Plant 2 — a small, known, irreducible overlap with A3's own
minimum injection magnitude). Coverage: 11/11 (Plant 1), 5/5 (Plant 2)
real injected nights caught.

**`flag_stuck_sensor()`** — targets A8. Same staleness problem with
Layer 1's `is_stuck` column. Recomputed fresh: N=8 consecutive identical
daylight values. Coverage 8/8 (Plant 1), 7/7 (Plant 2), 0/25 false
positives on genuinely clean data. **Known, accepted limitation**: this
rule also fires on N3 (grid curtailment) whenever the cap binds
continuously for ≥8 slots. **Two different fixes were attempted and both
reverted**: (1) a sign-of-`(claimed − expected)` discriminator caused A8
coverage to collapse from 8/8→4/8 and 7/7→1/7 (a freeze near the daily
peak mimics the same sign-change pattern as legitimate curtailment); (2) a
physics-ceiling-dip discriminator, tested at full 30-replicate scale
(n=15 A8, n=12 N3), gave **zero** N3 specificity improvement (0.583→0.583)
while costing 2 real A8 detections (15/15→13/15). Both reverted for
different reasons — the first cost too much for a real gain, the second
gained nothing at all. This is now a twice-attempted, twice-rejected,
documented-not-hidden limitation.

### 5.3 Conformal calibration (`src/layer3_anomaly/conformal.py`)

Split conformal, fit **only** on the 5 real `calib` days. Standard theory
needs `n ≥ ceil(1/α) − 1 = 19` days for a valid 5%-alpha threshold via the
proper order statistic; with n=5, far below that floor, the module falls
back to the conservative convention `threshold = max(calib r_day values)`
and reports the **real** achievable rate: `alpha_achievable = 1/(n+1) ≈
0.167` (16.7%), stated honestly rather than presented as "5% FPR."
Real fitted thresholds: **Plant 1 = 0.1666**, **Plant 2 = 0.2085**. One pooled,
unstratified threshold per plant — per-regime (Kt-stratified) thresholds
were considered and rejected, since Plant 1 has **zero** overcast calib
days to calibrate from at all.

**Why not pool `test_clean` in too, to get to n=10?** Considered and
explicitly rejected in the module's own comments: `test_clean` is the
project's last independent, never-calibrated sanity check on this exact
threshold — see Part 9 for the real bug it already caught once.

**Flagging is strictly one-sided**: `r_day > threshold` → `flagged_overclaim`;
`r_day < 0` → `fault_underclaim` (an operational-fault channel, never
fraud); otherwise `normal`. Two-sided testing was tried in ablation (d)
and rejected — it costs real specificity, concentrated on N4 (sensor
dropout: Plant 1 specificity 1.0→0.714).

### 5.4 Autoencoder structural scorer (`src/layer3_anomaly/autoencoder.py`)

Complementary to the magnitude-based conformal test — catches STRUCTURAL
anomalies (day-swap, flatline) where individual values look locally
plausible but the *joint* relationship between weather, the twin's
expectation, and the claim is wrong. Per-timestep, not per-day-sequence
(a day-sequence model would give ~20 independent training examples — an
order of magnitude too few). Inputs: the twin's 20 weather features,
q10/q50/q90, claimed power, and two engineered diff features
(`ac_power_diff1`, `residual_diff1` — targeting A8's frozen-claim
signature specifically). Threshold: 95th percentile of pooled
`calib+test_clean` reconstruction error. Run-length: longest over-threshold
run ever observed on that same clean pool, plus one (Plant 1 = **29**,
Plant 2 = **14** — a naive "any single timestep" rule was tested and
rejected, see ablation (f) below). Catches A2, A3, A4, A5, A6, A7 (A4
strongest, 60% of its rows exceed baseline); does **not** catch A1 (pure
magnitude — correctly conformal's job) or A8 (checked at the exact
labeled frozen window — genuinely zero elevation, despite features
engineered specifically for it; A8 is caught by the deterministic
guardrail instead).

### 5.5 CUSUM sequential drift detector (`src/layer3_anomaly/cusum.py`)

Targets the hardest realistic case: a persistent 2–15% inflation ramped
in over 3–9 days, too small to trip any single-day threshold.

```
z_t = (r_day(t) − μ_ref) / σ_ref
S_0 = 0;  S_t = max(0, S_{t−1} + z_t − k)
flag day t if S_t > h
```

`μ_ref`/`σ_ref` from pooled `calib+test_clean` (n=10) — **not** `train`,
which is provably ~2x optimistically biased (Plant 1 train mean=0.047 vs.
pooled mean=0.086; Plant 2 train mean=0.037 vs. pooled mean=0.103). `h` is
bootstrap-calibrated from clean data alone (2000 resamples, smallest `h`
achieving ≤5% empirical FPR), **never** tuned against A6 labels — real
value ≈ **4.0** for a 9-day window at `k=0.25`. **Scope limitation, stated
not silently absorbed**: `h` is only valid for the exact window length it
was calibrated for; running a 9-day-calibrated `h` over a full 34-day
sequence was found to produce a false positive on genuinely clean Plant 2
data (scored "Elevated" instead of "Low") — fixed by recalibrating `h`
fresh for whatever window length is actually being scored, every time.
Honest result at this calibration: only 1 of 4 real A6 replicates flagged
(the 7-day/14.4%-delta ramp; the three shorter 3–4 day ramps at 8–12%
delta were missed) — a legitimate finding about sequential-test power at
this data scale, not a bug.

### 5.6 Secondary-evidence escalation (`src/layer3_anomaly/escalation.py`)

Upgrades a day already classified `"normal"` to `"flagged_overclaim"`
ONLY if **both**: (1) its `r_day` sits within `band_width × threshold`
and `threshold` (strictly `<` on the upper bound — see Part 9's real bug),
AND (2) at least one secondary signal corroborates — the autoencoder's
run-length flag, or `cusum_s_t > cusum_frac × cusum_h`. Shipped
parameters: `band_width=0.8`, `cusum_frac=0.5` — chosen via honest
tune(replicates 0–19)/holdout(replicates 20–29) validation after an
initial, more aggressive pair (0.7/0.0) looked "free" but turned out to
be tuned against the same data used for final scoring; the honest
redo showed a real (smaller) specificity cost, and a further real-data
check caught an actual bug (Part 9) before the final conservative
parameters shipped. Result: pooled recall **58.5%→64.8%**, F1
**0.677→0.723**, hard-negative specificity **unchanged at 92.0%** (the
identical 28 false positives, verified, not just the same count).

### 5.7 Risk aggregation (`src/layer3_anomaly/aggregate.py`)

A 0–100 project-level score from five normalized, weighted components:

```
frac_flagged        weight 0.30
cum_severity_frac   weight 0.30   (capped at 20% of total claimed energy)
cusum_max_s/h       weight 0.20
|CR − clean CR|      weight 0.12   (this plant's OWN clean baseline, never 1.0)
|divergence|          weight 0.08   (A5-specific corroborator, weighted lowest)
```

Bands: Low 0–25, Elevated 25–50, High 50–75, Critical 75–100. Weights are
a **documented judgment call, not empirically fit** — stated plainly, not
disguised as optimized. Real acceptance-test scores on genuinely clean
data: Plant 1 = 12.1 (Low), Plant 2 = 20.2 (Low).

**Severity is flag-type-aware, not one generic formula** — this took
three separate bug-fixes to get right (the same "don't dilute a localized
signal across a mostly-clean window" mistake, found three times): a
night-injection day's severity uses only the actual dark-hour energy
claimed (an early bug inflated one real case from a correct ~500 kWh to
an incorrect ~173,000 kWh — over 300x too high); a stuck-sensor day's
severity uses only the frozen run's duration, not the whole day; an
overclaim day correctly uses the whole day's excess energy.

**Layer 3 output contract**: `[timestamp/day, residual, conformal_bound,
flag, flag_type, ae_score, cusum_stat, severity]` + one project-level risk record.

---

## Part 6 — Benchmark comparisons (model selection evidence)

### 6.1 Twin model comparison — the real numbers, same held-out test split

| Model | Plant 1 R² | Plant 2 R² |
|---|---|---|
| Physics-only (zero training data) | 0.835 | **0.751** |
| **PI-GBQR (production)** | 0.866 | 0.681 |
| Ridge regression | 0.790 | −0.026 |
| LSTM | **0.870** | −0.728 |
| TFT (Temporal Fusion Transformer) | 0.483 | 0.179 |
| TSMixer | 0.240 | −0.206 |

Sources: `reports/tables/{plant}_twin_metrics.json` (physics-only,
PI-GBQR), `{plant}_baseline_metrics.json` (Ridge, LSTM),
`{plant}_tft_r2_metrics.json` / `{plant}_tsmixer_r2_metrics.json` (deep
models — these test on a slightly smaller n, 426/414 vs. 480, since their
architectures need a sequence window and lose a few boundary rows; a
disclosed, not hidden, comparability caveat).

**Honest reading**: this is not a clean sweep for PI-GBQR. LSTM edges it
out on Plant 1; PI-GBQR actually underperforms plain physics on Plant 2.
What the table *does* show clearly: TFT and TSMixer are consistently
worse than every simpler model on both plants (this dataset's ~500–900
rows isn't enough for transformer-scale architectures to show an
advantage); LSTM and Ridge are both wildly inconsistent across plants
(fine on Plant 1, collapse to negative R² on Plant 2); **PI-GBQR is the
only model that's reasonable on both plants** — never best, never
catastrophic. Combined with the two structural requirements in Part 4.2
(quantile output, fast exact SHAP), this is why PI-GBQR shipped — chosen
upfront for architecture reasons, and this comparison table is the
after-the-fact check that the choice was defensible, not a leaderboard
PI-GBQR topped outright.

### 6.2 Detector-side ablations (`src/evaluation/ablations.py`)

| # | Comparison | Result |
|---|---|---|
| (a) | Satellite vs. on-site weather input | On-site more accurate (Plant 1: 0.866→0.892 R²) — quantifies the accuracy cost paid for tamper-resistance; never deployed, satellite stays production |
| (b) | With/without physics anchor | Plant 2's twin underperforms physics-only (0.681 vs. 0.751) — same finding as 6.1, sourced from the same file, not re-derived |
| (c) | Conformal vs. fixed 2σ threshold | Conformal wins clearly: F1 0.723 vs. 0.619, AUC-PR 0.766 vs. 0.677; identical specificity (0.920) on both, investigated and confirmed guardrail-driven, not either magnitude threshold |
| (d) | One-sided vs. two-sided | One-sided: recall 0.425, specificity 0.971. Two-sided: recall 0.430, specificity 0.943 — two-sided's cost concentrates on N4 sensor-dropout hard negatives |
| (e) | Online vs. batch twin | Not built — scoped decision, see Part 4.7 |
| (f) | Conformal vs. autoencoder-alone vs. naive-OR ensemble | Conformal (production): specificity 0.920. Autoencoder alone (naive "any timestep" rule): specificity **0.246**. Naive OR ensemble: specificity **0.194**. Confirms why the autoencoder is a corroborator (Part 5.6), never a standalone detector |
| (g) | Leave-one-plant-out | See Part 7.3 |

---

## Part 7 — Validation: how the whole system was actually tested

### 7.1 The frozen synthetic fraud generator (`src/injection/`)

Because no real, labeled fraud dataset exists (nobody publishes "here's
a plant that definitely cheated"), the project built its own. **Governance
protocol, to prevent evaluation circularity**: the injector is written by
a separate sub-team, frozen and git-tagged (`injector-v1.0`) *before* any
detector threshold is tuned. Anomaly parameters are sampled randomly from
declared ranges (`config/anomalies.yaml`), never hand-picked. Ground-truth
labels live in a file the detector code never imports
(`data/ground_truth/{plant}_labels.parquet`). Thresholds are calibrated
only on `calib`; `test_injected` (and the injected replicates built from
it) is touched exactly once, at the end.

**Eligible pool**: 9 contiguous days (`test_clean` days 26–30 +
`test_injected` days 31–34). 30 replicates per plant are built by
bootstrap-resampling within that pool (`per_replicate_positive_frac=0.35`
→ ~3 of 9 days get a fraud injection; `per_replicate_hard_negative_frac=0.30`
→ ~3 of 9 days get a legitimate oddity; the rest stay untouched, real,
honest data) — up to 270 labeled day-instances per plant.

### 7.2 The exact mechanics of every injected anomaly type (`src/injection/taxonomy.py`)

Each function takes one day's 15-minute slice and returns a modified
`ac_power` series. **These are the actual algorithms, verified directly
from source**, not a paraphrase:

| Code | Name | Exact mechanism |
|---|---|---|
| **A1** | Constant over-claim | `claimed_new = claimed × (1 + δ)` for the WHOLE day, `δ ~ Uniform(0.02, 0.40)` |
| **A2** | Peak-hour shaving | Same multiplicative bump, but ONLY applied where local clock hour ∈ `[10, 15]`; `δ ~ Uniform(0.05, 0.30)` |
| **A3** | Night injection | For every timestamp where `is_daylight == False`, sets `claimed = magnitude_frac × capacity_kw`, `magnitude_frac ~ Uniform(0.02, 0.15)` — i.e. fabricates a flat non-zero output at night |
| **A4** | Day-swap | Takes a genuinely sunny day's `ac_power` curve and overlays it onto a cloudy target day, aligned slot-by-slot by time-of-day; the target day's real weather features are left untouched — correct (cloudy) weather next to an implausible (sunny) claim |
| **A5** | Weather spoofing | Inflates BOTH `claimed` and the on-site sensor's `evidence_onsite_irradiation` together by the same factor, `~ Uniform(0.10, 0.50)` — simulating a tampered on-site sensor "justifying" the inflated claim. The independent NASA-derived twin features are untouched — this is exactly what the flag-SHAP divergence feature is meant to catch |
| **A6** | Slow drift | Across a contiguous block of `n_days ~ randint(3, 9)`, day `i`'s multiplier is `final_delta × (i+1)/n_days` — a linear ramp from near-zero up to `final_delta ~ Uniform(0.05, 0.15)` by the last day. Deliberately built to never cross a single-day magnitude threshold; only CUSUM is meant to catch it |
| **A7** | Capacity violation | Rescales the whole day so its peak exceeds nameplate capacity: `target_peak = capacity_kw × overshoot`, `overshoot ~ Uniform(1.02, 1.25)`, then `scale = target_peak / current_peak` applied to every timestamp |
| **A8** | Flatline / frozen replay | Picks a random daylight start position and freezes `claimed` at that one value for `run_length ~ randint(8, 40)` consecutive daylight slots — simulates a stuck meter or a replayed recording |

### 7.3 The exact mechanics of every hard-negative type (`src/injection/hard_negatives.py`)

Two families: **selected** (a real qualifying event found in the actual
data, used unmodified) and **synthesized** (built the same way as the
positive taxonomy). All must NOT be flagged as fraud:

| Code | Name | Family | Exact mechanism |
|---|---|---|---|
| **N1** | Inverter outage | Selected | Fits a linear regression of mean daily power against mean daily Kt across all real days, then picks the day with the largest *negative* residual — genuinely underperforming relative to what its own clear-sky index predicts (separates real outages from merely cloudy days) |
| **N2** | Soiling loss | Synthesized | `claimed_new = claimed × (1 − loss_per_day × day_position)`, a small (0.2%/day) linear efficiency decay across the 9-day window |
| **N3** | Grid curtailment | Synthesized | `claimed_new = claimed.clip(upper = capacity_kw × cap_frac)`, `cap_frac ~ Uniform(0.5, 0.8)` — a hard output cap, the real-world event that collides with the stuck-sensor guardrail (Part 5.2) |
| **N4** | Sensor dropout | Synthesized | Sets a random run of `4–24` consecutive daylight slots to `NaN` — should be caught by Layer 1 QC and never even reach Layer 3 as a value to test |
| **N5** | Heavy cloud day | Selected | The real day within the eligible pool with the lowest mean Kt — genuinely cloudy, should widen the conformal interval, never trigger a flag |
| **N6** | Extreme heat | Synthesized | `claimed_new = claimed × (1 + γ × offset)`, `γ = −0.004` (matching the twin's own temperature-coefficient assumption), `offset ~ Uniform(4, 8)°C` — physically lower output at high panel temperature, something the twin's temperature feature should already explain |
| **N7** | Cloud enhancement | Selected | The real day within the eligible pool with the highest mean on-site-vs-satellite irradiance divergence — a known satellite-grid-averaging limitation, not fraud |

### 7.4 Full benchmark results (`python -m src.evaluation.benchmark`, `src/evaluation/benchmark.py`)

Pooled, both plants, all 30 replicates each (540 day-instances, 193
positive), production flag definition:

| Metric | Value |
|---|---|
| Precision | 0.817 |
| Recall | 0.648 |
| F1 | 0.723 |
| AUC-PR | 0.766 |
| Hard-negative specificity | 0.920 (161/175) |

**Per-anomaly-type recall**: A3 (night injection) = **1.000**, A8
(flatline) = **1.000**, A4 (day-swap) = 0.742, A7 (capacity violation) =
0.741, A1 (constant overclaim) = 0.759, A5 (weather spoofing) = 0.700,
A2 (peak-hour shaving) = 0.259, A6 (slow drift) = **0.056** (the weakest
category — see Part 10).

**Per-hard-negative-type specificity**: N1 = 1.000, N7 = 0.966, N6 =
0.962, N5 = 0.957, N2 = 0.909, N4 = 0.886, **N3 (curtailment) = 0.583**
(the one specificity weak point, the documented stuck-sensor collision).

### 7.5 Minimum Detectable Overstatement curve (`src/evaluation/mdo_curve.py`) — the flagship result

Sweeps injected magnitude (1–30% of claimed power) across three
aggregation windows: a single 15-minute reading, a single hour, or the
full day. n=9 held-out clean days per cell (one day excluded — it was
already a false positive at zero injection, a documented, investigated
exclusion, not silently dropped).

| Magnitude | 15-min window | Hourly window | Daily window |
|---|---|---|---|
| 12% | 0.0 | 0.0 | 0.111 |
| 20% | 0.0 | 0.0 | 0.444 |
| 30% | 0.0 | 0.0 | 0.778 |

**The headline finding**: a fraud pattern confined to a single 15-minute
reading or a single hour is undetected at **every magnitude tested, up to
30%** — the system's core decision only ever looks at a full day's total
energy, so a brief spike dilutes into an otherwise normal day's total.
This is the single most honest, disclosed limitation this project
reports, and it's shown directly on the dashboard's Accuracy & Testing tab.

### 7.6 Leave-one-plant-out generalization (`src/evaluation/leave_one_plant_out.py`)

| Plant scored | Twin used | Precision | Recall | Specificity |
|---|---|---|---|---|
| Plant 1 | own | 0.922 | 0.621 | 0.943 |
| Plant 1 | Plant 2's twin (cross) | 0.371 | **1.000** | **0.159** |
| Plant 2 | own | 0.742 | 0.673 | 0.897 |
| Plant 2 | Plant 1's twin (cross) | 0.676 | 0.490 | 0.897 |

Cross-plant generalization is **asymmetric**: Plant 2's (weaker, more
overfit) twin transplanted onto Plant 1 causes severe over-flagging
(recall shoots to 1.0, but specificity collapses to 0.16). Plant 1's twin
transplanted onto Plant 2 degrades far more gracefully. Conclusion,
stated plainly: this system is validated per-plant, not proven to
generalize automatically to a brand-new site without retraining.

### 7.7 Real-data acceptance checks (independent of the synthetic tests entirely)

Beyond the synthetic replicates, the system is checked against genuinely
real, verified-clean plant data (the `real_clean` block — `calib` +
`test_clean`, 10 days, never touched by any calibration): does the risk
score correctly land in the "Low" band? Real results: Plant 1 = 12.1,
Plant 2 = 20.2 — both Low, as expected. This check is separate from, and
more load-bearing than, the synthetic tests for one specific failure
class — see Part 9.

### 7.8 Automated tests (`tests/`, root-level `test_*.py`, `pytest.ini`)

A `pytest` suite (40+ tests per the project's own Definition of Done)
including the leakage guard `TC-L2-01` (hard-asserts the twin's feature
matrix contains only `twin_*`-prefixed columns, never `claimed`/`evidence_*`),
the solar-noon timezone assertion, and per-module sanity scripts
(`test_conformal_sanity.py`, `test_cusum_sanity.py`,
`test_residual_sanity_v2.py`) that hand-verify the core math against
known, independently computed values.

### 7.9 The tune/holdout validation discipline (used for every parameter choice)

Whenever a genuinely new parameter needs choosing (never a raw threshold —
those are always calib-only), the 30 replicates are split into
`TUNE_REPLICATES = range(0,20)` and `HOLDOUT_REPLICATES = range(20,30)`.
The parameter is swept and chosen using tune-set labels only; the chosen
configuration is then evaluated **once** on the untouched holdout set. If
the gain survives, it's real; if it collapses toward baseline, it was
overfit to the known 30 cases. This exact pattern was used for
secondary-evidence escalation (Part 5.6) and for every improvement
attempt in Part 10.

---

## Part 8 — Layer 4: Explainability, carbon, and the dashboard

### 8.1 Twin-SHAP (`src/layer4_xai/twin_shap.py`) — "why did we expect this much power?"

SHAP over the twin's feature space, for one representative timestamp per
flagged day (the daylight row with the largest `|residual|` — explaining
every 15-minute slot of a flagged day would be mostly redundant, since
weather doesn't change fast enough within a day to tell a distinct
story). **A real unit/scope gotcha**: the raw booster predicts the
residual-correction target as a *fraction* of nameplate capacity, not kW
— any SHAP value must be multiplied by `ac_nameplate_kw` before display.
**A real finding worth stating in the paper**: the roadmap's assumption
that low GHI / high cell temp would be the top negative drivers does
*not* hold in practice — the physics anchor already captures that
dominant relationship, so the twin's *residual* correction (the only
thing SHAP explains) reflects secondary effects instead: pressure,
humidity, day-of-year, precipitation, wind. Confirmed across ~10 flagged
days on both plants.

### 8.2 Flag-SHAP (`src/layer4_xai/flag_shap.py`) — "why is this suspicious?"

Two separate paths:
- **Path A** (`flagged_overclaim`): z-score ranking of day-level evidence
  features (`irr_divergence_pct_mean`, `temp_divergence_mean`,
  `onsite_irradiation_mean`, `night_claimed_kwh`, `yield_jump_frac`)
  against the clean `calib+test_clean` reference (n=10 — too small to
  responsibly train a model, so this is a direct statistical ranking).
- **Path B** (`flagged_night_injection`, `flagged_stuck_sensor`): a
  deterministic template stating the exact rule and trigger values that
  fired — nothing statistical to attribute for an if-statement.

**Honest calibration caveat**: `yield_jump_frac` is the top Path-A driver
in most overclaim cases, but this is partly circular — an overclaim day
almost definitionally differs from the day before. `irr_divergence_pct_mean`
is the more independent, on-site-vs-satellite check.

### 8.3 Carbon accounting (`src/layer4_xai/carbon.py`)

```
excess_credits (tCO2e) = excess_kWh / 1000 × grid_emission_factor
value_at_risk = excess_credits × carbon_price
```

Real configured values: `grid_emission_factor_tco2_per_mwh = 0.7383`
(CEA India CO₂ Baseline Database v21.0, the RE-crediting variant, *not*
the corporate Scope-2 WAEF); `carbon_price_usd_per_tco2 = 5.00` (low end
of the $4–6 range for grid-connected renewable avoidance credits,
reflecting a documented ~28% EMDE pricing discount vs. advanced-economy
credits). **Two mandatory caveats**, always rendered alongside any dollar
figure: a vintage mismatch (the emission factor is dated 2025, the
generation data is from 2020) and price uncertainty (the voluntary
carbon market has no single agreed price). Terminology is always "at
risk," never "confirmed fraud" or "credits over-issued."

### 8.4 Evidence bundles (`src/layer4_xai/evidence.py`)

Combines twin-SHAP + flag-SHAP + carbon into one exportable JSON record
per flagged day — what an actual verifier would receive. Every field is
independently cross-checked against its source module at build time, not
just internally self-consistent.

### 8.5 Severity ranking (`src/layer4_xai/severity_ranking.py`)

Sorts flags by `value_at_risk_usd` (the one quantity comparable across
all three flag mechanisms) and reports a `confidence_ratio` as an honest,
**mechanism-specific** column — never implied to be a unified statistic
comparable across flag types.

### 8.6 The dashboard (`dashboard/app.py`, `dashboard/lib/`)

A single-page Streamlit app, five tabs, no page-reload between them (so
the sidebar's plant/period selection always persists). Deliberately
**zero implementation-detail vocabulary** on screen — no "Layer 3.6",
"CUSUM", "conformal", "SHAP", "F1/AUC-PR" anywhere in rendered text; all
translated via `dashboard/lib/ui.py`'s label dictionaries.

- **Overview** — the risk-score verdict banner, three stat tiles, a
  flag-type bar chart, and a collapsed "how is this calculated" note.
- **Evidence** — claimed vs. expected power over time (with the twin's
  q10–q90 band, explicitly captioned as *not* the conformal interval —
  the real statistical guarantee is on the daily aggregate, shown in the
  second chart below it: daily `r_day` bars vs. the conformal threshold line).
- **Carbon Credits** — the project's actual subject, given its own
  prominent tab (promoted from originally being tab 4 of 5, after
  feedback that it was too easy to miss).
- **Flag Details** — per-flag drill-down: twin-SHAP panel ("what we
  expected") and flag-SHAP panel ("why it's flagged"), each now paired
  with a short plain-language sentence synthesizing the top 1–2 drivers
  (added to make fuller use of the explainability layer — see Part 11.2),
  plus any known caveat for that flag type.
- **Accuracy & Testing** — pooled, not filtered by the sidebar: the two
  headline test numbers ("fraud caught" / "legitimate cases correctly
  left alone") and the MDO curve, relabeled in plain language ("a single
  moment" / "one hour" / "a full day").

`dashboard/lib/pipeline.py` wires all of this — it reimplements nothing;
every number is a direct call into `src/layer{2,3,4}_*`, cached via
`st.cache_resource` (needed because the cached objects hold LightGBM
boosters, which aren't reliably picklable, ruling out `st.cache_data`).

---

## Part 9 — A real bug, caught by the real-data check, worth understanding precisely

After wiring secondary-evidence escalation (Part 5.6) into the live
scoring path, the standing "real clean data should score Low" check (Part
7.7) surfaced a new problem: Plant 1's real clean block picked up a
**newly flagged** day, `2020-06-03` — which turned out to be the *exact*
`calib` day whose `r_day` **defines** the conformal threshold itself
(`threshold = max(calib r_day)`). Confirmed via exact floating-point
equality (`r_day == threshold`, diff `0.0`) — not a rounding illusion.

**Root cause**: `near_threshold`'s upper bound originally used `<=`
`conformal_threshold`. A day sitting exactly at the threshold satisfies
that inclusively, even though the primary test's own strict `>` would
never call that day anomalous — it's definitionally the most extreme
example of "normal" the threshold was calibrated from, not evidence
against it. **Fixed** by changing the comparison to strict `<`.

This is the concrete reason the real-clean check (calibrated on data the
threshold was never fit against) is treated as load-bearing throughout
this project, not a formality — the synthetic 30-replicate benchmark
would never have caught this specific bug, because it never scores the
literal calib day.

---

## Part 10 — The secondary-evidence escalation story, and the further round that followed

### 10.1 Why escalation exists

The autoencoder (5.4) and CUSUM (5.5) both carry real detection signal
that the primary test (5.3) never uses. Blindly trusting either as a
blanket OR was tested (ablation f) and rejected — it costs more in new
false positives than the recall gain is worth. Escalation asks a
narrower question: can either help WITHOUT that cost, if only trusted on
a day that's already borderline?

### 10.2 Three rounds to the final, shipped parameters

1. **First pass**: swept `band_width`/`cusum_frac` by testing directly
   against the same 30 labeled replicates used for final scoring. Found
   an apparently "free" result (+11pp recall, zero specificity cost) —
   caught and disclosed as a methodology mistake before being trusted
   (this is exactly the "tasting the dish before serving it to the
   judges" error the tune/holdout discipline in Part 7.9 exists to prevent).
2. **Second pass**: redone honestly, tuned on replicates 0–19, verified
   on 20–29. The core idea *did* generalize (holdout F1 0.716→0.794), but
   with a real, disclosed specificity cost this time (0.867→0.833) — the
   "free" result did not survive honest validation.
3. **Third pass**: wiring the pass-2 parameters into production and
   re-checking the real-clean acceptance blocks found the bug in Part 9 —
   a cost neither pass 1 nor pass 2 ever tested against, since a real,
   hand-verified honest day is a genuinely different check than an
   unseen synthetic replicate. Backed off to the final, more conservative
   `band_width=0.8, cusum_frac=0.5`.

**Final, shipped result**: recall 58.5%→64.8%, F1 0.677→0.723,
specificity unchanged at 92.0% (the identical 28 false positives).

### 10.3 A further round: four more attempts, honestly tested, none shipped

After escalation shipped, a follow-up effort targeted the two categories
it left weakest — A2 (25.9% recall) and A6 (5.6% recall) — without
touching the twin model or restructuring any layer. Full detail:
`docs/ESCALATION_IMPROVEMENT_HANDOFF.md` §8.

1. **A peak-window residual** (restricted to each day's busiest hours,
   ranked by the physics anchor to avoid overfitting to the injector's
   own clock window). **Result: zero net improvement at every
   configuration tested.** Root cause: narrowing the window increases the
   residual's noise almost exactly as fast as it concentrates the
   injected signal — the two effects cancel at this project's 5-day
   calibration scale.
2. **A faster-reacting CUSUM timescale**, checked with an honest,
   clean-data-only power pre-check *before* touching any label. **Result:
   the best candidate gained only ~2 percentage points against a
   pre-declared 15-point bar** — stopped at the gate. Lowering `k` forces
   the FPR-calibration to raise `h` back up almost proportionally,
   canceling most of the gain.
3. **A combined-evidence test** (standalone): one continuous score
   combining `r_day`, CUSUM's statistic, and the autoencoder's
   reconstruction error, standardized and averaged, with its own
   calib-only threshold, replacing the primary test entirely. **Result:
   real gains** — A2 recall roughly doubled (25.9%→40.7%), A6 recall
   jumped to 33.3% — **but hard-negative specificity collapsed from 92.0%
   to 62.9%**, because CUSUM/AE respond to any unusual pattern, not
   specifically over-claiming, so they fire on legitimate operational
   anomalies too.
4. **The same combined test, gated** the same conservative way escalation
   already gates its own corroborators. **Result: statistically
   indistinguishable from current production overall (byte-identical
   TP/FP/FN/TN), but a real, specific *loss* of 2 A2 detections**, made up
   elsewhere; A6 completely unaffected, because a slow ramp by design
   never approaches the daily threshold this gate is built around —
   independent of what corroborates within it.
5. **Pooled calibration** (using `calib+test_clean`, n=10, for the main
   threshold) — considered, never coded. Would improve the achievable
   false-positive guarantee from ~16.7% toward ~9–17%, but would remove
   the exact real-clean sanity check that caught the Part 9 bug, with no
   substitute available (`train` is already too biased to serve as one).
   Judged not worth the trade.

**The pattern across all five**: every path traced back to either the
5-day real calibration floor, or the fact that slow drift never
approaches the daily decision threshold every gating mechanism here is
built around. Production numbers are unaffected by any of this — this
round's value was diagnostic (converting "probably near a data ceiling"
from an inference into a demonstrated result), not corrective.

---

## Part 11 — This session's other changes (documentation and UI, not detection logic)

### 11.1 The "trust score" → "risk score" rename

The dashboard displayed the internal 0–100 risk score (0 = clean, 100 =
maximally suspicious) directly under the label "Trust score," with no
inversion — so a genuinely clean plant scoring 12/100 showed as "Trust
score: 12/100," reading backwards (a low "trust" number sounds like *low*
trust). Fixed by relabeling to "Risk score" everywhere (`dashboard/app.py`,
`docs/DASHBOARD_GUIDE.md`) — the number itself was never wrong, only the
label was misleading.

### 11.2 Flag Details: turning charts into sentences

The "What we expected" and "Why it's flagged" panels showed only bar
charts, requiring the viewer to interpret them. Added a short
plain-language sentence under each, synthesizing the top 1–2 SHAP-ranked
drivers directly from data Layer 4 already computes (no new backend
logic). The physical panel states direction ("pushed the estimate
up/down" — well-defined for a kW contribution); the suspicion panel
deliberately does not state direction (what "higher vs. lower than
normal" means varies per feature, and guessing wrong would misstate the
evidence rather than just under-explain it). This also surfaced and fixed
a real, pre-existing gap: the day-level aggregate evidence-feature column
names (`irr_divergence_pct_mean`, etc.) weren't in the plain-language
mapping dictionary, so they were falling through to raw, ugly column
names — now fixed.

---

## Part 12 — Repository map

```
carbontwin-ai/
├── config/                 config.yaml (every calibrated number, documented inline),
│                            plants.yaml, anomalies.yaml (frozen injector parameters)
├── data/
│   ├── raw/                 kaggle/, nasa_power/
│   ├── processed/            train/calib/test_clean/test_injected parquet + injected/ (30 replicates × 2 plants)
│   └── ground_truth/         sealed labels, read once by benchmark.py
├── src/
│   ├── layer1_data/          acquisition, timezone, downscale, quality, features, capacity, splits
│   ├── layer2_twin/          physics_anchor, twin_model (PIGBQRTwin), train_twin, baselines
│   ├── layer3_anomaly/       residual, conformal, cusum, autoencoder, escalation, aggregate
│   ├── layer4_xai/           twin_shap, flag_shap, carbon, evidence, severity_ranking
│   ├── injection/            taxonomy.py (A1-A8), hard_negatives.py (N1-N7), run_injection.py (FROZEN)
│   └── evaluation/           harness (shared scoring), benchmark, mdo_curve, leave_one_plant_out,
│                              ablations, stuck_sensor_fix
├── dashboard/                app.py (5 tabs), lib/pipeline.py (wiring/caching), lib/ui.py (labels)
├── tests/                    pytest suite
├── reports/
│   ├── models/                per-plant twin boosters + autoencoder weights (gitignored)
│   ├── tables/                every benchmark/validation JSON output
│   └── evidence_bundles/      per-flag exportable records (gitignored, regenerable)
└── docs/                      every *_HANDOFF.md, this file, and the plain-language explainers
```

---

## Part 13 — Where to go for more detail

This document consolidates; it doesn't replace the source docs, which go
deeper on their own layer:

| Topic | Document |
|---|---|
| Full project definition, aim, timeline | `docs/CarbonTwinAI_Project_Overview_v2.md` |
| Plain-language architecture walkthrough | `docs/HOW_THE_SYSTEM_WORKS.md` |
| Layer 2 gotchas and internals | `docs/LAYER2_HANDOFF.md` |
| Layer 3 gotchas and internals | `docs/LAYER3_HANDOFF.md` |
| Layer 4 modules 4.1–4.5 | `docs/LAYER4_MODULES_1_TO_5_HANDOFF.md` |
| Dashboard wiring | `docs/LAYER4_DASHBOARD_HANDOFF.md` |
| Every dashboard element explained | `docs/DASHBOARD_GUIDE.md` |
| Full Task 7 validation methodology | `docs/TASK7_HANDOFF.md` |
| Consolidated result numbers | `docs/TASK7_FINDINGS.md` |
| Results in plain language | `docs/PROJECT_RESULTS_SUMMARY.md` |
| Escalation feature + the further improvement round | `docs/ESCALATION_IMPROVEMENT_HANDOFF.md` |

---

## Part 14 — Honest summary of what this project does and doesn't claim

**Does claim**: a genuinely independent, weather-based verification
method, implemented correctly, tested against realistic sealed scenarios
(not just described in theory), with every weak spot measured and
disclosed rather than hidden. 65% of simulated fraud caught, 92% of
legitimate oddities correctly left alone, two fraud types caught with
100% reliability (night injection, frozen sensor).

**Does not claim**: 100% fraud detection (no realistic system could);
that a flag is proof of fraud (only a calibrated statistical red flag,
achievable false-positive rate ~16.7%, not the nominal 5%); real-time
operation (T+1 week, due to weather-data latency); generalization beyond
the two plants and one season tested; detection of brief (sub-daily)
fraud at any tested magnitude up to 30% — a measured, disclosed, and
currently unresolved limitation of the daily-decision architecture.
