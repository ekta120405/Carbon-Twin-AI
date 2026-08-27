# CarbonTwinAI — Final Project Overview (v2.0)

**An Audit-Grade Digital Twin for Continuous Verification of Renewable Energy Carbon Credits**

B.Tech Computer Engineering, Capstone Project, A.Y. 2026–2027
MPSTME, NMIMS Mumbai · Guide: Prof. Abhishek Vichare
Team: Ishitaa Jain (B051), Ekta Baphana (B013), Aayra Charania (B027), Nishi Chhajed (B029)

---

## 0. Purpose of this document

This supersedes the v1 overview. Nothing has been removed from scope. Every module in v1 is retained; several have been made more precise, and five modules have been **added** because they close correctness gaps that would otherwise have made the results indefensible.

**What changed from v1 → v2**

| # | v1 | v2 | Why |
|---|---|---|---|
| 1 | Layer 2 called a "forecasting" layer | Layer 2 is a **nowcasting / counterfactual estimator** | You have the weather; you are not predicting the future. This changes model choice, features and metrics. |
| 2 | Twin may use lagged generation | Twin **never** consumes claimed generation as an input feature | Otherwise the twin learns the fraud and detects nothing |
| 3 | "Train only on clean data" | Cleanliness is established by a **physics ceiling filter first**, then robust loss | v1 assumed the answer to the question being asked |
| 4 | Deviation = (claimed − expected)/expected | **Capacity-normalised residual** + daylight mask + daily-energy aggregation | v1 metric divides by ~0 at dawn/dusk/night |
| 5 | 2σ statistical threshold | **Split conformal prediction** (distribution-free) + one-sided CUSUM | v1 assumes iid Gaussian residuals; solar residuals are neither |
| 6 | Two-sided anomaly flagging | **One-sided** (over-claim only) + explicit fault-vs-fraud discrimination | Under-generation is an O&M problem, not a credit fraud |
| 7 | SHAP on the twin | **Two-tier explanation**: twin-SHAP + flag-SHAP | v1 explains the prediction, not the suspicion |
| 8 | *(absent)* | **Layer 4.3 carbon accounting** (kWh → tCO₂e → ₹ at risk) | The project is about carbon credits; nothing converted energy to credits |
| 9 | *(absent)* | **Clear-sky-index-preserving temporal downscaling** (hourly → 15-min) | Naive interpolation destroys solar geometry |
| 10 | *(absent)* | **Hard-negative set** (legitimate faults that must NOT be flagged) | Without it, precision numbers are meaningless |
| 11 | *(absent)* | **Minimum Detectable Overstatement (MDO) curve** | This is the project's strongest publishable result |
| 12 | *(absent)* | **Frozen injector protocol** (injection team ≠ detector team) | Prevents evaluation circularity |
| 13 | Timeline 5 weeks, loosely ordered | **4 phases, 26 working days, strict dependency order** | Sequenced so that no task starts before its input exists |

---

## 1. Project definition

### 1.1 One-line concept (revised)

CarbonTwinAI estimates what a solar project's generation *should have been* from weather data the operator cannot manipulate, tests the project's self-reported generation against that estimate using a statistically calibrated one-sided decision rule, and explains, quantifies and prices every flag it raises in tonnes of CO₂e.

### 1.2 What the system formally is

Three components, in strict order:

1. **An estimator** — ĝ(t) = E[generation(t) | independent weather(t), site metadata], with uncertainty bounds
2. **A test statistic** — s(t) = (claimed(t) − ĝ(t)) / P_ac_nameplate, aggregated over a decision window
3. **A decision rule** — a one-sided conformal test with a calibrated false-positive rate α, plus a sequential CUSUM for persistent low-magnitude bias

Everything else in the system supports, explains, or reports on these three.

### 1.3 The governing constraint (state this early in the paper)

Detection power is bounded by the twin's residual spread. If the twin's residual standard deviation is σ, a systematic over-claim of magnitude δ is detectable at window length n only when

```
δ  >  z_α · σ / sqrt(n_eff)
```

where `n_eff` ≪ n because solar residuals are strongly autocorrelated. Consequence: instantaneous detection of small over-claims is physically impossible; aggregation is mandatory. Quantifying this trade-off **is a result, not a limitation** — see §7.3.

---

## 2. Aim and objectives (revised — replaces the deck's Objectives slide)

**Aim.** To build an AI-driven digital twin that continuously and independently verifies renewable-energy carbon credit claims by estimating expected generation from tamper-resistant third-party data, testing self-reported claims against that estimate with calibrated statistical guarantees, explaining every flag, and quantifying the resulting carbon and financial exposure.

**Objectives**

1. Ingest independent irradiance and meteorological data from NASA POWER for the plant's location, maintained in strict isolation from all plant-reported data, so that tampered on-site sensors cannot influence verification.
2. Construct a hybrid digital twin combining a physics-based clear-sky performance anchor (`pvlib` / PVWatts) with a data-driven quantile regression model, trained exclusively on independent weather covariates.
3. Detect over-claims using distribution-free conformal prediction intervals with a calibrated false-positive rate, augmented by sequential change detection for persistent low-magnitude bias, and aggregate timestamp-level evidence into a project-level risk score.
4. Discriminate fraudulent over-reporting from legitimate operational faults through one-sided testing and validation against a curated hard-negative set.
5. Explain every flag through a two-tier SHAP framework identifying both the physical drivers of expected output and the features driving the suspicion.
6. Quantify each flag in tonnes of CO₂e and monetary value at risk using published grid emission factors.
7. Validate the complete pipeline against a frozen synthetic anomaly injection benchmark, reporting precision, recall, F1, AUC-PR, specificity on hard negatives, and the minimum detectable overstatement as a function of aggregation window.
8. Deliver an interactive audit dashboard and an IEEE-format research paper.

**SDG alignment:** SDG 7 (Affordable and Clean Energy), SDG 13 (Climate Action).

---

## 3. System architecture

Four layers are retained, matching the digital-twin pattern established in the literature (physical asset → data → analytics/simulation → service). Each layer now has explicit sub-modules and a defined output contract.

```
                 ┌──────────────────────┐   ┌──────────────────────┐
   AUDITED  ───► │  Claimed generation  │   │ Independent weather  │ ◄─── TRUSTED
                 │  + on-site sensors   │   │ NASA POWER (1° grid) │
                 └──────────┬───────────┘   └──────────┬───────────┘
                            │                          │
              ══════════════╪══════════════════════════╪═══════════════
              LAYER 1       ▼                          ▼
              1.1 acquisition · 1.2 timezone contract · 1.3 CSI downscaling
              1.4 physics ceiling QC · 1.5 feature engineering
              1.6 capacity estimation · 1.7 block splits
              ═══════════════════════════╪═══════════════════════════════
              LAYER 2                    ▼
              2.1 physics anchor (pvlib/PVWatts)   ──┐
              2.2 ML quantile twin (weather → power) ├─► expected(t), q10..q90
              2.3 robust training + calibration ratio│
              2.4 online learning module           ──┘
              ═══════════════════════════╪═══════════════════════════════
              LAYER 3                    ▼
              3.1 capacity-normalised residual · 3.2 conformal calibration
              3.3 one-sided flagging · 3.4 autoencoder scorer
              3.5 CUSUM sequential drift · 3.6 risk aggregation
              ═══════════════════════════╪═══════════════════════════════
              LAYER 4                    ▼
              4.1 twin-SHAP · 4.2 flag-SHAP · 4.3 carbon accounting
              4.4 evidence bundle · 4.5 severity ranking · 4.6 dashboard
              ═══════════════════════════════════════════════════════════
```

**Non-negotiable invariant:** no artefact derived from `Claimed generation` or on-site sensors may ever enter Layer 2's feature matrix. This is enforced by a unit test (§8, TC-L2-01), not by convention.

---

## 4. Layer 1 — Data Layer

### 4.1 Data sources

| Stream | Source | Role | Resolution | Trust |
|---|---|---|---|---|
| Claimed generation | Kaggle `anikannal/solar-power-generation-data` — `Plant_{1,2}_Generation_Data.csv` | The thing being audited | 15-min, 22 inverters/plant, 34 days (May–Jun 2020) | **Untrusted** |
| On-site sensors | Kaggle `Plant_{1,2}_Weather_Sensor_Data.csv` (ambient temp, module temp, irradiation) | Evidence only — feeds Layer 4 divergence features | 15-min, 1 sensor/plant | **Untrusted** |
| Independent weather | NASA POWER API (hourly) | Twin input | Hourly; solar 1°×1° (CERES SYN1deg), meteo 0.5°×0.625° (MERRA-2) | **Trusted** |
| Solar geometry | `pvlib` (computed, not downloaded) | Clear-sky ceiling, solar position | 15-min, exact | **Deterministic** |
| Grid emission factor | CEA India CO₂ Baseline Database | kWh → tCO₂e | Annual constant | **Reference** |

**Parameters to request from NASA POWER:** `ALLSKY_SFC_SW_DWN` (GHI), `ALLSKY_SFC_SW_DNI`, `ALLSKY_SFC_SW_DIFF`, `CLRSKY_SFC_SW_DWN`, `T2M`, `RH2M`, `WS2M`, `PS`, `PRECTOTCORR`, `ALLSKY_KT` (clear-sky index).

### 4.2 Known dataset constraints (documented, not hidden)

- The Kaggle dataset publishes **no coordinates**. Plant locations must be assumed. Run a sensitivity analysis across the assumed cell and its neighbours; report the effect on twin nRMSE.
- NASA POWER solar is a ~110 km grid-cell average, not a point measurement. Expect degraded twin accuracy under convective cloud — which is exactly the May–June monsoon-onset regime of this dataset.
- NASA POWER latency: CERES SYN1deg runs at NRT minus ~3–4 months; FLASHFlux fills to within ~5–7 days. The honest product claim is **T+1 week verification**, not real-time. State this explicitly.
- Plant 2's DC power magnitudes differ from Plant 1's by roughly an order of magnitude (inverter-level vs string-level scaling). Normalise per plant before any cross-plant modelling.
- 34 days, single season, two sites. No seasonal generalisation is claimable.

### 4.3 Module 1.2 — Timezone contract (do this on day one)

Kaggle timestamps are naive local time (IST, UTC+05:30). NASA POWER hourly is UTC. A 5.5-hour misalignment produces a twin that predicts night-time generation and flags every daylight hour.

**Contract:** every timestamp in the pipeline is stored as `tz-aware UTC`. Localisation happens exactly once, at ingestion, per source. A validation assertion checks that the empirical daily generation peak falls within ±90 minutes of computed solar noon; the pipeline aborts if it doesn't.

### 4.4 Module 1.3 — Clear-sky-index-preserving downscaling (hourly → 15-min)

Do **not** forward-fill or linearly interpolate irradiance. Instead:

1. Compute 15-minute clear-sky GHI at the site with `pvlib` (Ineichen or Haurwitz model) — this carries the exact solar geometry.
2. Compute hourly clear-sky index `Kt = GHI_nasa / GHI_clearsky_hourly`.
3. Linearly interpolate `Kt` to 15 minutes (cloud opacity varies smoothly; solar geometry does not).
4. Reconstruct `GHI_15min = Kt_interp × GHI_clearsky_15min`.

This preserves the diurnal shape that naive interpolation flattens, and is worth a paragraph in the paper.

### 4.5 Module 1.4 — Physics ceiling QC (this is how "clean data" is defined)

For every timestamp, compute the physically maximum possible AC output:

```
P_ceiling(t) = P_dc_nameplate × (GHI_clearsky(t)/1000) × η_max × (1 − γ·(T_cell(t) − 25)) × 1.05
```

Rows where `claimed(t) > P_ceiling(t)` are **physically impossible**. They are excluded from twin training and automatically flagged at maximum severity in Layer 3. This is the bootstrap of trust: physics, not data, decides what counts as clean.

Also remove: night-time rows outside the sunrise–sunset window (solar elevation < 10°), stuck-sensor runs (identical value ≥ 8 consecutive timestamps during daylight), negative power, and inverter-level dropouts.

### 4.6 Module 1.5 — Feature engineering

*Twin features (independent only):* GHI, DNI, DHI, clear-sky GHI, clear-sky index Kt, T2M, RH2M, WS2M, surface pressure, precipitation, solar zenith, solar azimuth, air mass, extraterrestrial irradiance, plane-of-array irradiance (computed via `pvlib` transposition), estimated cell temperature (Faiman/SAPM from T2M, WS2M, POA), hour-of-day sin/cos, day-of-year sin/cos, and short lags of GHI/Kt (t−1, t−2) to capture thermal inertia.

*Evidence features (Layer 4 only, never Layer 2):* on-site irradiation, on-site ambient temp, on-site module temp, and the divergence terms `onsite_irr − nasa_ghi_scaled`, `onsite_temp − T2M`. A large positive divergence is the fingerprint of on-site sensor tampering.

### 4.7 Module 1.6 — Capacity estimation

Nameplate capacity is not published. Estimate `P_ac_nameplate` per plant as the 99.5th percentile of plant-level summed AC power over clear-sky-index > 0.9 timestamps. Freeze this constant; the entire Layer 3 metric depends on it.

### 4.8 Module 1.7 — Splitting protocol (leakage control)

- **Never** random-shuffle 15-minute data. Split by **contiguous day blocks**.
- Four disjoint sets: `TRAIN` (days 1–20), `CALIB` (days 21–25, clean only, used solely for conformal calibration), `TEST-CLEAN` (days 26–30), `TEST-INJECTED` (days 31–34, plus reinjected copies of TEST-CLEAN).
- Additionally run **leave-one-plant-out**: train on Plant 1, evaluate on Plant 2, and vice versa. This is your only available generalisation evidence.

**Layer 1 output contract:** a tz-aware UTC-indexed table per plant containing `[timestamp, independent_features…, evidence_features…, claimed_ac_power, P_ceiling, is_daylight, is_clean, split_tag]`.

---

## 5. Layer 2 — Twin / Nowcasting Layer

### 5.1 Module 2.1 — Physics anchor (PVWatts / pvlib)

A deterministic, non-learned estimate of expected AC output from POA irradiance, cell temperature, system size and loss assumptions. It serves three purposes:

1. Defines the physics ceiling used in Layer 1.4.
2. Provides a model-independent second opinion on every flag.
3. Provides the **calibration ratio** diagnostic (§5.4).

### 5.2 Module 2.2 — ML quantile twin

Maps independent weather features → expected AC power. Two mandatory properties:

- **Quantile output** (at minimum q10, q50, q90) rather than a point estimate, because Layer 3 needs uncertainty and cloud-regime-dependent interval width.
- **Fast, exact explanations**, because Layer 4 must explain every flag.

The specific model family, and the deep-learning baseline it is benchmarked against, is decided in **Task 2**. The architecture and all downstream contracts are model-agnostic — Layer 3 consumes `(expected, q10, q90)` regardless of what produced them.

*Constraint:* claimed generation and on-site sensor values are forbidden as inputs. Enforced by test TC-L2-01.

### 5.3 Module 2.3 — Robust training

Train on `TRAIN ∩ is_clean ∩ is_daylight`. Use a robust objective (pinball loss for quantiles, Huber for the point head) so that any residual undetected anomalies in the training window cannot drag the fit. Report training-set influence: refit on a 10%-contaminated training set and show the twin's predictions shift by less than a stated tolerance.

### 5.4 Module 2.4 — Calibration ratio diagnostic (new, high value)

Compute `CR = median(claimed) / median(physics_anchor)` over clear-sky timestamps. A plant whose claims can only be fitted by assuming an implausible system efficiency is *itself* evidence of systematic inflation, independent of any timestamp-level test. Report CR per plant per week as a standing global indicator.

### 5.5 Module 2.5 — Online learning (retained from v1)

Incremental refit on a rolling window as new verified-clean data arrives, so the twin tracks seasonal and soiling drift without full retraining. Guard: only data that passed the physics ceiling **and** was not flagged in the previous cycle may enter the update — otherwise the twin slowly learns the fraud. Evaluate online vs. batch on the final held-out block.

**Layer 2 output contract:** `[timestamp, expected_ac, q10, q90, physics_anchor, calibration_ratio, claimed_ac]`.

---

## 6. Layer 3 — Anomaly Detection Layer

### 6.1 Module 3.1 — Residual definition (replaces v1's ratio metric)

```
r(t) = (claimed(t) − expected(t)) / P_ac_nameplate          # capacity-normalised
```

Computed only where `is_daylight`. Aggregated to the decision window:

```
R_day(d) = Σ_t claimed(t)·Δt − Σ_t expected(t)·Δt           # daily energy discrepancy, kWh
r_day(d) = R_day(d) / (P_ac_nameplate × daylight_hours(d))  # normalised
```

The primary decision is made at **daily energy** level. Timestamp-level residuals are retained for explanation and localisation, not for the primary flag.

### 6.2 Module 3.2 — Conformal calibration (replaces the 2σ threshold)

Split conformal / conformalised quantile regression on the `CALIB` block:

1. Compute nonconformity scores on clean calibration data.
2. Take the ⌈(n+1)(1−α)⌉-th order statistic as the threshold.
3. Stratify calibration by clear-sky-index bin (clear / partly cloudy / overcast) so interval width adapts to cloud regime.

This gives a **distribution-free, finite-sample** guarantee on the false-positive rate with no Gaussian assumption. It is the methodological differentiator versus every paper in the literature table.

### 6.3 Module 3.3 — One-sided flagging (fault vs. fraud)

Flag only when `claimed` exceeds the upper conformal bound. Under-generation is recorded as an **operational fault** in a separate channel — surfaced on the dashboard as an O&M finding, never as a credit-fraud flag. This single change eliminates the dominant false-positive mode (inverter outages) and makes the metric honest.

### 6.4 Module 3.4 — Autoencoder anomaly scorer (retained from v1)

Trained on clean `(features, expected, claimed)` joint patterns; high reconstruction error indicates a relationship the model has never seen. Serves as a complementary detector for structural anomalies (day-swaps, replayed profiles) that a residual magnitude test can miss. Reported both standalone and ensembled with the conformal detector.

### 6.5 Module 3.5 — Sequential drift detection

One-sided CUSUM over daily normalised residuals. Catches the hardest and most realistic fraud: a persistent 2–4% inflation that never trips any single-day threshold. Report the run-length to detection as a function of inflation magnitude.

### 6.6 Module 3.6 — Risk aggregation

Project-level risk score combining: proportion of days flagged, cumulative unexplained excess energy, CUSUM state, calibration ratio, and mean on-site/satellite sensor divergence. Output a 0–100 score with a documented, non-arbitrary mapping and a categorical band (Low / Elevated / High / Critical).

**Layer 3 output contract:** `[timestamp/day, residual, conformal_bound, flag, flag_type, ae_score, cusum_stat, severity]` + one project-level risk record.

---

## 7. Layer 4 — Explainability, Carbon Accounting & Application Layer

### 7.1 Module 4.1 — Twin-SHAP ("why did we expect only 25 MW?")

SHAP over the twin's feature space for each flagged period. Answers the physical question: low GHI, high cell temperature, high air mass, etc.

### 7.2 Module 4.2 — Flag-SHAP ("why is this suspicious?")

A separate, small explainer over *residual* features: on-site vs. satellite irradiance divergence, sign persistence, magnitude relative to the conformal bound, day-over-day yield discontinuity, night-time generation, ceiling violation. This is the layer that distinguishes "cloudy day, model uncertain" from "sensors tampered". v1 conflated these two questions; separating them is what makes the explanation actionable.

### 7.3 Module 4.3 — Carbon accounting (new)

```
excess_energy(kWh) → excess_credits(tCO₂e) = excess_kWh / 1000 × EF_grid
value_at_risk = excess_credits × carbon_price
```

Using the CEA India grid emission factor (~0.71–0.82 tCO₂/MWh — cite the specific edition used) and a stated carbon price. Report per flag, per project, per reporting period. This is what converts a machine-learning output into an auditor's finding.

### 7.4 Module 4.4 — Evidence bundle

For each flag, an exportable record: period, claimed vs. expected vs. physics-anchor curves, conformal bounds, residual, both SHAP attributions, sensor divergence, tCO₂e at risk, severity, and recommended action. This is the deliverable an actual verifier would receive.

### 7.5 Module 4.5 — Severity ranking

Rank flags by `severity = f(magnitude, confidence, persistence, tCO₂e at risk)` so auditors triage the highest-value investigations first.

### 7.6 Module 4.6 — Dashboard (Streamlit)

Five views: (1) project risk overview and score, (2) expected vs. claimed time series with conformal band and flags, (3) flag detail with both SHAP panels, (4) carbon and financial exposure, (5) evaluation/benchmark view showing detector performance and the MDO curve.

---

## 8. Synthetic anomaly injection framework

### 8.1 Governance protocol (prevents evaluation circularity)

- The injector is written by **one sub-team** and **frozen** (git-tagged) before any detector threshold is tuned.
- Anomaly parameters are sampled randomly from declared ranges, not hand-picked.
- Ground-truth labels are held in a separate file the detector code never imports.
- Thresholds are calibrated **only** on `CALIB` (clean). `TEST-INJECTED` is touched once, at the end.

### 8.2 Positive anomaly taxonomy (must be detected)

| ID | Anomaly | Mechanism | Realism |
|---|---|---|---|
| A1 | Constant over-claim | claimed × (1+δ), δ ∈ [0.02, 0.40] | Systematic inflation |
| A2 | Peak-hour shaving | inflate only 10:00–15:00 | Hides in high-variance hours |
| A3 | Night injection | non-zero output at zero irradiance | Crude fabrication; ceiling test catches it |
| A4 | Day-swap / replay | substitute a sunny day's profile for a cloudy day | Classic profile fabrication |
| A5 | Weather spoofing | inflate on-site sensor readings to justify inflated output | Tests the divergence feature |
| A6 | Slow drift | δ ramps 0 → 15% over the window | Tests CUSUM; hardest case |
| A7 | Capacity violation | output above nameplate | Trivial positive control |
| A8 | Flatline / freeze | repeated value with plausible magnitude | Meter tampering |

### 8.3 Hard-negative set (must NOT be flagged as fraud)

| ID | Event | Expected system behaviour |
|---|---|---|
| N1 | Inverter outage (real, present in Plant 1) | Fault channel, not fraud |
| N2 | Progressive soiling loss | Fault channel |
| N3 | Grid curtailment (simulated) | Fault channel |
| N4 | Sensor dropout / missing rows | Handled by QC, no flag |
| N5 | Genuine heavy-cloud day | Widened conformal interval, no flag |
| N6 | Extreme heat derating | Absorbed by cell-temperature feature |

Specificity on hard negatives is a headline metric. Reporting it is what separates this from a toy evaluation.

### 8.4 Injection ratio

Inject into 15–20% of test days (mixed types, mixed severity), preserving realistic class imbalance. Report AUC-PR, not just ROC-AUC, because of that imbalance.

---

## 9. Evaluation protocol

### 9.1 Twin (Layer 2)
MAE, RMSE, nRMSE (% of capacity), MBE, R², skill score vs. the physics anchor, and vs. a persistence baseline. Reported separately for clear / partly cloudy / overcast regimes — a single aggregate number hides the failure mode that matters.

### 9.2 Detector (Layer 3)
Precision, recall, F1, AUC-PR, AUC-ROC against injected labels. Empirical false-positive rate on `TEST-CLEAN` versus the nominal conformal α (this validates the calibration guarantee). Specificity on the hard-negative set. Per-anomaly-type recall breakdown.

### 9.3 Minimum Detectable Overstatement curve — the flagship result
Sweep injected magnitude δ ∈ {1, 2, 3, 5, 8, 12, 20, 30}% × aggregation window ∈ {15-min, hourly, daily, weekly, monthly}. Plot detection rate at fixed α. The resulting surface answers: *"How much fraud can satellite-only verification actually catch, and over what horizon?"* No paper in the literature review answers this.

### 9.4 Explanation quality (Layer 4)
Fraction of flags where the top-2 flag-SHAP features match the injected mechanism (e.g. A5 → sensor divergence ranked first). This gives a quantitative explainability metric rather than screenshots.

### 9.5 Ablations
(a) satellite weather vs. on-site weather as twin input; (b) with vs. without physics anchor; (c) conformal vs. fixed 2σ threshold; (d) one-sided vs. two-sided; (e) online vs. batch twin; (f) conformal vs. autoencoder vs. ensemble; (g) leave-one-plant-out generalisation.

---

## 10. Technology stack (revised)

| Component | Technology | Purpose |
|---|---|---|
| Language | Python 3.11 | — |
| Data | pandas, numpy, pyarrow | Handling, columnar storage |
| Solar physics | **pvlib** | Clear-sky, solar position, POA transposition, cell temp, PVWatts |
| Independent data | NASA POWER API (`requests`) | Trusted weather |
| Twin — primary | *decided in Task 2* | Quantile nowcast |
| Twin — deep baseline | *decided in Task 2* (PyTorch) | Benchmark |
| Conformal | `mapie` or ~40 lines of custom code | Calibrated intervals |
| Autoencoder | PyTorch | Structural anomaly scoring |
| Classical baselines | scikit-learn | Ridge, RF, Isolation Forest |
| Explainability | SHAP | Twin-SHAP + flag-SHAP |
| Dashboard | Streamlit + Plotly | Application layer |
| Testing | pytest | 40+ test cases (Task 7) |
| Config | Hydra or YAML + pydantic | Reproducibility |
| Tracking | MLflow (optional) or CSV logs | Experiment records |
| Paper | LaTeX (IEEEtran), Overleaf | Deliverable |

---

## 11. Repository structure

```
carbontwin-ai/
├── config/            # config.yaml, plants.yaml, anomalies.yaml
├── data/
│   ├── raw/           # kaggle/, nasa_power/
│   ├── interim/       # aligned, resampled
│   └── processed/     # train/calib/test parquet + injected sets
├── src/
│   ├── layer1_data/       acquisition.py, timezone.py, downscale.py,
│   │                      quality.py, features.py, capacity.py, splits.py
│   ├── layer2_twin/       physics_anchor.py, twin_model.py, train.py,
│   │                      online.py, calibration_ratio.py
│   ├── layer3_anomaly/    residual.py, conformal.py, autoencoder.py,
│   │                      cusum.py, aggregate.py
│   ├── layer4_xai/        twin_shap.py, flag_shap.py, carbon.py,
│   │                      evidence.py, severity.py
│   ├── injection/         injector.py (FROZEN), hard_negatives.py
│   ├── evaluation/        twin_metrics.py, detector_metrics.py,
│   │                      mdo_curve.py, ablations.py
│   └── utils/             io.py, logging.py, validators.py
├── dashboard/         app.py + pages/
├── tests/             test_layer1.py … test_integration.py
├── notebooks/         01_eda … 06_results
├── reports/           figures/, tables/, evidence_bundles/
└── paper/             main.tex, refs.bib, figures/
```

---

## 12. Team allocation

| Member | Primary ownership | Secondary |
|---|---|---|
| Ishitaa Jain | Layer 2 (twin, physics anchor, training, online learning) | Twin evaluation, paper §Method |
| Ekta Baphana | Layer 1 (acquisition, timezone, downscaling, QC, features, splits) | Data section of paper, EDA figures |
| Aayra Charania | Layer 3 (residual, conformal, autoencoder, CUSUM, aggregation) | MDO curve, ablations |
| Nishi Chhajed | **Injection framework (frozen, isolated)** + Layer 4 (SHAP, carbon, dashboard) | Test suite, paper §Results |

Nishi owning injection while Aayra owns detection enforces §8.1 separation structurally, not just procedurally.

---

## 13. Build sequence and timeline

Extended to **26 working days (~5 calendar weeks)**. Tasks follow the exact order specified; nothing starts before its input exists.

### Phase A — Foundation (Days 1–5) · covers Tasks 2, 3, 4

| Day | Work | Owner | Gate |
|---|---|---|---|
| 1 | **Task 2**: model selection decision, benchmarked plan | All (review) | Model + baseline fixed in writing |
| 1 | **Task 3**: environment, GPU/CPU decision, `requirements.txt`, repo skeleton | Ishitaa | `pytest` runs on empty suite |
| 2 | **Task 4**: Kaggle extraction, integrity checks | Ekta | 4 CSVs verified, row counts match |
| 2–3 | **Task 4**: coordinate assumption, NASA POWER API extraction, sensitivity cells | Ekta | Hourly weather for all 34 days + neighbours |
| 3 | EDA: inverter behaviour, missingness, Plant 2 scaling, known outages | Ekta, Nishi | EDA notebook + N1 outage window identified |
| 4 | **Timezone contract + solar-noon assertion** | Ekta | Assertion passes on both plants |
| 4–5 | Injection framework v1 written and **frozen/tagged** | Nishi | Git tag `injector-v1.0`, labels sealed |
| 5 | **Gate A review** | All | Data aligned, injector frozen, model chosen |

### Phase B — Layers 1 & 2 (Days 6–12) · Task 5 + first half of Task 6

| Day | Work | Owner |
|---|---|---|
| 6 | CSI-preserving downscaling, POA transposition, cell temperature | Ekta |
| 7 | Physics ceiling QC, cleaning, capacity estimation | Ekta, Ishitaa |
| 8 | Feature engineering, split protocol, processed datasets written | Ekta |
| 8 | Physics anchor (PVWatts) implemented and sanity-checked | Ishitaa |
| 9–10 | Twin training, quantile heads, hyperparameter search | Ishitaa |
| 11 | Deep baseline + classical baselines, regime-stratified metrics | Ishitaa |
| 11 | Calibration ratio diagnostic | Ishitaa |
| 12 | Online learning module | Ishitaa |
| 12 | **Gate B review**: twin nRMSE reported by cloud regime; leakage test TC-L2-01 passing | All |

### Phase C — Layers 3 & 4 (Days 13–19) · second half of Task 6

| Day | Work | Owner |
|---|---|---|
| 13 | Residual computation, daily aggregation | Aayra |
| 14 | Conformal calibration, CSI-stratified, one-sided | Aayra |
| 15 | Autoencoder scorer | Aayra |
| 16 | CUSUM, risk aggregation, project scoring | Aayra |
| 15–16 | Twin-SHAP and flag-SHAP | Nishi |
| 17 | Carbon accounting module, evidence bundle export | Nishi |
| 18–19 | Streamlit dashboard, all five views | Nishi |
| 19 | **Gate C review**: end-to-end run produces flags with explanations and tCO₂e | All |

### Phase D — Validation & Paper (Days 20–26) · Tasks 7, 8

| Day | Work | Owner |
|---|---|---|
| 20–21 | **Task 7**: full test suite (unit, integration, invariant, statistical) | Nishi + all |
| 21–22 | Benchmark run on `TEST-INJECTED`, hard negatives, per-type recall | Aayra |
| 22–23 | **MDO curve** sweep, ablation matrix, leave-one-plant-out | Aayra, Ishitaa |
| 23–24 | Figures, tables, results freeze | All |
| 24–26 | **Task 8**: IEEE paper drafting, internal review, submission format | All |
| 26 | Final integration test, demo rehearsal, deck update | All |

### Buffer policy
Days 12, 19 and 24 are gates. If a gate slips, the *next* phase absorbs the delay — Phase D is never compressed, because the results and paper are the graded deliverables.

---

## 14. Risk register

| Risk | Impact | Mitigation | Fallback |
|---|---|---|---|
| Assumed coordinates are wrong | Twin accuracy degrades | Sensitivity across 9 grid cells; pick best by clear-sky-day fit | Report as documented limitation; results still valid relatively |
| Twin nRMSE too high on cloudy days | Detection power drops | CSI-stratified conformal intervals | MDO curve *is* the result; a high MDO is a finding, not a failure |
| NASA POWER API unavailable/rate-limited | Blocks Phase A | Download early (Day 2), cache to disk, commit the cache | `pvlib`-simulated clear-sky-only fallback for the demo path |
| Only 34 days → overfitting | Inflated metrics | Block splits, leave-one-plant-out, robust loss, small model | Report with explicit variance bounds over repeated block splits |
| Autoencoder underperforms on 3k rows | Weak Layer 3.4 | Keep it small, ensemble it rather than rely on it | Conformal detector is the primary path; AE reported as ablation |
| SHAP too slow for full test set | Phase C slip | Explain flagged periods only, not all timestamps | Sample 100 flags for the explainability metric |
| Timeline slip | Paper quality suffers | Hard gates at Days 12/19/24 | Online learning (2.5) and the AE (3.4) are the two modules whose *depth* can be reduced last — but both still ship |

**De-risking guarantee.** Even in the worst case where the ML twin generalises poorly, the physics anchor alone detects gross fabrication (A3, A7, and A1 at large δ), the conformal machinery still produces calibrated intervals, and the MDO curve still produces a publishable quantitative result. The project cannot produce "nothing".

---

## 15. Definition of done

- [ ] `python -m src.run_pipeline --plant 1` executes end to end from raw data to evidence bundles
- [ ] Leakage test TC-L2-01 passes: twin feature matrix provably contains no claimed/on-site column
- [ ] Solar-noon assertion passes on both plants
- [ ] Twin metrics reported by cloud regime, against physics anchor and classical baselines
- [ ] Empirical FPR on clean test data within tolerance of nominal conformal α
- [ ] Precision / recall / F1 / AUC-PR reported per anomaly type
- [ ] Specificity on all six hard negatives reported
- [ ] MDO curve produced across 8 magnitudes × 5 windows
- [ ] Every flag carries two SHAP panels and a tCO₂e figure
- [ ] Dashboard runs all five views on a fresh machine from `requirements.txt`
- [ ] ≥ 40 tests passing
- [ ] IEEE paper complete with all references resolved (including the blank [11], [12], [13])

---

## 16. Amendments required to the Title Approval deck

1. **Objectives slide** — replace "Train an LSTM model…" with objectives 2–7 from §2 (model-family-neutral wording).
2. **Tech Stack table** — update the Forecasting Model row after Task 2; add rows for `pvlib` (physics anchor) and conformal prediction.
3. **Aim slide** — "instantly identifying" → "identifying, quantifying and explaining", and add the carbon-quantification clause.
4. **Feasibility slide** — add the honest latency note (T+1 week, not real-time).
5. **References** — fill [11], [12], [13]; suggested additions: a conformal prediction reference (Vovk et al. or Angelopoulos & Bates), a `pvlib`/PVWatts reference, and the CEA CO₂ Baseline Database.
6. **Research Gap slide** — sharpen from "no one has combined these" to the four-point contribution claim in §2, foregrounding calibrated one-sided detection, fault-vs-fraud discrimination, and the MDO characterisation.
