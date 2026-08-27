# Layer 4, Modules 4.1-4.5 — Handoff Notes

Status as of this commit: **4.1, 4.2, 4.3, 4.4, 4.5 built and verified
against real repo data (both plants, multiple injected replicates) —
every number in this document was independently reproduced on Ishitaa's
own machine, not just in a sandbox.** 4.6 (dashboard) is the only
remaining module.

This document exists for the same reason LAYER2_HANDOFF.md and
LAYER3_HANDOFF.md exist: several real gotchas were found and fixed while
building these modules that are NOT in docs/LAYER4_ROADMAP.md (the
roadmap's own §3 flags its own sketch as unverified against the real
repo — this document is the verification that was missing). Whoever
builds 4.6 needs to know these or the dashboard will silently misrepresent
what the underlying modules actually do.

---

## 0. File locations

```
src/layer4_xai/twin_shap.py         Module 4.1
src/layer4_xai/flag_shap.py         Module 4.2
src/layer4_xai/carbon.py            Module 4.3
src/layer4_xai/evidence.py          Module 4.4
src/layer4_xai/severity_ranking.py  Module 4.5

verify_twin_shap_module.py          root-level, one per module,
verify_flag_shap_module.py          matching this project's existing
verify_carbon_module.py             check_*.py / score_*.py convention.
verify_evidence_module.py           Re-run any of these against the real
verify_severity_ranking_module.py   repo to reproduce every number below.
```

`dashboard/app.py` + `dashboard/pages/` is the target location for 4.6,
per docs/LAYER4_ROADMAP.md §"File:".

---

## 1. Module 4.1 (Twin-SHAP) — two unit/scope gotchas, neither in the roadmap

**Gotcha 1 — units.** `PIGBQRTwin`'s raw LightGBM boosters predict `y_hat`,
the residual-correction TARGET AS A FRACTION of `ac_nameplate_kw` (Layer
2's `y = (claimed - physics_anchor) / ac_nameplate_kw`), **not** kW and
**not** `expected_q50`. Verified by direct reconstruction:
`shap_values.sum(axis=1) + expected_value == raw_booster.predict(X)`
exactly, and that quantity is a small fraction (e.g. -0.008), not a
kW-scale number. **Any SHAP value must be multiplied by `ac_nameplate_kw`
before being shown as a kW contribution.** Even after that conversion,
SHAP only explains the twin's *correction on top of* the physics anchor —
never the physics anchor itself, which is deterministic pvlib/PVWatts
arithmetic with no model behind it, hence no SHAP attribution.
**Dashboard implication:** any panel showing Twin-SHAP contributions must
label them "twin's correction," never "why we expected X kWh" unqualified
— the physics anchor's own contribution needs its own explanation, and
that explanation is "it's physics, not a model."

**Gotcha 2 — the low-Kt fallback silently discards the booster.**
`twin.predict()`'s `kt_switch=0.4` fallback means the learned correction
is not used at all when `twin_kt < 0.4` — Layer 3 uses `physics_ac_kw`
directly. Verified on a real plant_1 row (`twin_kt=0.357`): fallback-path
`expected_q50` was 2229.9 kW; what SHAP would have explained had the
fallback not existed was 1318.6 kW — a ~900 kW gap on a single timestep.
`twin_shap.py`'s `TwinShapExplainer.explain_timestamp()` checks
`twin_kt` against `kt_switch` BEFORE calling SHAP and returns a
`mode="physics_anchor_fallback"` result with an empty
`top_features_kw` and an explanatory `note` instead of running TreeSHAP
on a prediction Layer 3 never used. **Dashboard implication:** any
flagged-day detail view must branch on `physical_explanation.mode` —
`"shap"` shows the feature-contribution chart, `"physics_anchor_fallback"`
must show the `note` text instead and MUST NOT render an empty/misleading
SHAP chart.

**Real finding, worth stating in the paper, not fixing:** the roadmap's
stated "definition of done" (low GHI / high cell temp as top negative
contributors) does not hold in practice. Real top drivers across both
plants are almost always `twin_doy_sin`, `twin_ps`, `twin_rh2m`,
`twin_prectotcorr`, `twin_ws2m` — pressure, humidity, day-of-year,
precipitation, wind. This is because the physics anchor already captures
the dominant GHI/cell-temp relationship, so the twin's *residual*
correction (the only thing SHAP explains) naturally reflects the
secondary effects the physics model missed, not the primary irradiance
signal. Confirmed across ~10 flagged-day explanations on both plants —
not a one-off.

**A real bug that was caught and fixed, worth watching for in future
edits:** the first implementation crashed with a dtype error —
`x_row[feature_cols].to_frame().T` silently inherits `object` dtype from
the mixed-type parent row (bool columns like `is_daylight` mixed with
floats), and LightGBM's predictor rejects `object` dtype input. Fixed by
rebuilding the row as an explicit `float`-typed DataFrame. If this
function is ever refactored, re-verify the dtype survives the rebuild.

**Scope decision** (matches roadmap's own principle, applied one level
deeper): `pick_representative_timestamp()` explains one representative
timestamp per flagged DAY (the daylight row with the largest `|residual|`),
not every 15-minute slot — explaining every timestep of a flagged day is
mostly redundant since weather doesn't change fast enough within a day to
tell a distinct story.

---

## 2. Module 4.2 (Flag-SHAP) — two paths, one real data gap, one honest calibration note

**Two separate code paths — do not conflate in the dashboard:**
- **Path A** (`flagged_overclaim` only): deviation-based z-score ranking
  of day-level evidence features against a clean reference pool
  (`calib`+`test_clean`, n=10 real days per plant — NOT a trained model,
  n is too small to responsibly train one). `rank_suspicion_drivers()`.
- **Path B** (`flagged_night_injection`, `flagged_stuck_sensor`):
  deterministic template explanations reporting the exact rule and
  trigger values that fired. Nothing statistical to attribute — these are
  if-statements, not models. `explain_night_injection_flag()`,
  `explain_stuck_sensor_flag()`.

**Real data gap found before writing code:** `evidence_yield_discontinuity`
(listed as a candidate feature in the roadmap) does **not exist** in the
real processed parquet — `build_evidence_features()`'s
`if daily_yield is not None` branch never fires because `daily_yield`
isn't a column anywhere upstream. Substituted with `yield_jump_frac`
(day-over-day `claimed_energy_kwh` diff, normalized by `ac_nameplate_kw`)
— same intent ("sudden jump vs. the plant's own recent trend"), computed
from data that actually exists.

**Honest calibration note, confirmed across multiple replicates:**
`yield_jump_frac` is the top Path-A driver in most `flagged_overclaim`
cases. This is expected but partially circular — an overclaim day almost
definitionally differs from the day before, so this feature correlates
with the flag's own definition, unlike `irr_divergence_pct_mean` (an
independent on-site-vs-satellite check). **Dashboard implication:** if
Path A rankings are shown to a stakeholder, this should be noted next to
`yield_jump_frac` specifically, the same way Layer 3 was honest that
`evidence_irr_divergence_pct` alone isn't a universal discriminator.

**Guardrail caveats must travel with their flag type, not get lost:**
- `flagged_stuck_sensor` carries `STUCK_SENSOR_CAVEAT` — this rule also
  fires on legitimate grid-ordered curtailment (N3); a fix attempt was
  tried and reverted after it caused a severe coverage regression on real
  A8 examples (see LAYER3_HANDOFF.md §3). **This flag must display as
  "suspected," never "confirmed," fraud.**
- `flagged_night_injection` bundles (built in 4.4) carry an additional
  caveat: the Twin-SHAP physical_explanation panel for these days
  describes an unrelated daylight moment, not the actual flagged
  night-time event (see §1 above and §3 below) — residual.py's daily
  residual is structurally computed from daylight rows only, so no
  daylight timestamp can represent the real violation.

**Plant-specific baseline, not a bug:** plant_2's clean reference pool has
nonzero `night_claimed_kwh` on every single clean day (15-99 kWh range);
plant_1's is exactly 0.0 on all 10. Z-scores are computed against each
plant's own mean/std, so this doesn't break the math, but it's consistent
with plant_2's already-known weaker measurement fidelity (city-centroid
coordinates, not satellite-verified — LAYER2_HANDOFF.md §4).

---

## 3. Module 4.3 (Carbon) — pure arithmetic, two mandatory caveats

`compute_carbon_exposure(excess_kwh, ef, price)`: `(kwh/1000) * ef` then
`* price`. **Always calls `compute_daily_flag_severity_kwh()` for the
excess-energy number — never recomputes it.** A day-wide or naive
excess-energy number would reintroduce the exact dilution bug fixed three
times in Layer 3 (see LAYER3_HANDOFF.md and aggregate.py's module
docstring). Rejects negative `excess_kwh` with a `ValueError` as a guard
against a caller accidentally passing a signed residual.

**Real config values** (updated from Task-3 placeholders `0.79`/`10.0`):
```
grid_emission_factor_tco2_per_mwh: 0.7383   # CEA CO2 Baseline Database v21.0
carbon_price_usd_per_tco2: 5.00             # low end of $4-6/tCO2e VCM range
```

**Two mandatory caveats, exported as constants
(`VINTAGE_MISMATCH_CAVEAT`, `PRICE_UNCERTAINTY_CAVEAT`,
`REQUIRED_CAVEATS`)** — the dashboard MUST display both anywhere carbon
numbers appear, not just compute silently:
1. Vintage mismatch: EF is dated Nov 2025 (FY2024-25); the dataset's
   generation is from 2020.
2. Price uncertainty: the VCM is opaque with wide dispersion
   ($0.25-$27/tCO2e observed); the configured price is a representative
   low-end point estimate, not a quoted market rate.

**Terminology:** outputs are "tCO2e AT RISK" / "value AT RISK," never
"credits over-issued" or "fraud confirmed" — Layer 3's flags are
statistical suspicion (achievable conformal alpha ~16.7%, see
LAYER3_HANDOFF.md §4), not adjudicated fact.

**Verified on real clean data:** plant_1 clean block (calib+test_clean,
n=10) → $0 exposure. plant_2 → $0.37 total, from one
`flagged_night_injection` day — this is the documented ~0.36% guardrail
false-positive overlap, not a new bug.

---

## 4. Module 4.4 (Evidence Bundle) — assembly only, one added caveat

Combines 4.1+4.2+4.3 into one JSON record per flagged day
(`build_evidence_bundle()`, `build_evidence_bundles_for_plant()`,
`save_evidence_bundles()`). No new modeling. Every field is independently
cross-checked against the source module at verification time (e.g.
`severity_kwh` matched a *freshly, separately called*
`compute_daily_flag_severity_kwh()`, not just internal self-consistency).

**Added `NIGHT_INJECTION_TWIN_SHAP_CAVEAT`** — appears in
`known_caveats` specifically for `flagged_night_injection` bundles,
stating plainly that the physical_explanation panel describes a different
(daylight) moment than the one that triggered the flag. This is the kind
of thing that's easy to miss when assembling a dashboard view that just
iterates `known_caveats` generically — make sure this specific caveat
renders visibly next to the physical_explanation panel for night-injection
bundles, not just in a generic footnote list.

Bundle structure (real, not the roadmap's earlier sketch):
```json
{
  "plant": str, "date": str, "flag_type": str, "severity_kwh": float,
  "physical_explanation": {
    "mode": "shap" | "physics_anchor_fallback", "twin_kt": float, "note": str,
    "physics_ac_kw": float, "expected_q50_kw": float,
    "twin_correction_kw": float | null,          // null in fallback mode
    "top_features_kw": [[feature, kw_value], ...] // empty in fallback mode
  },
  "suspicion_explanation": {
    "mode": "statistical",                        // Path A
    "top_drivers": [[feature, z_score], ...], "raw_values": {...}, "reference_n": int
  } | {
    "mode": "guardrail",                           // Path B
    "text": str, "trigger_values": {...}
  },
  "carbon_exposure": {"excess_tco2e": float, "value_at_risk_usd": float},
  "known_caveats": [str, ...]
}
```
`reports/evidence_bundles/` is `.gitignore`'d except `.gitkeep` (same
pattern as `reports/figures/`) — bundles are treated as regenerable
output, not committed artifacts, by existing project convention.

---

## 5. Module 4.5 (Severity Ranking) — confirmed no overlap with Layer 3, built new

**Overlap check, done before writing code (per roadmap's own
instruction):** read all of `aggregate.py`. `compute_risk_score()` takes
one `RiskScoreInputs` (five already project-level-aggregated numbers) and
returns one `RiskScoreResult` — no per-flag object anywhere. Confirmed:
**no overlap, 4.5 is genuinely new**, not an extension of 3.6.

**Design choice, stated plainly:** does NOT collapse severity and
confidence into a third arbitrarily-weighted composite score on top of
3.6's already five-input weighted score. `rank_flags()` sorts by
`value_at_risk_usd` (the one quantity comparable across all three flag
mechanisms) and reports `confidence_ratio` as an honest separate column.

**`confidence_ratio` is mechanism-specific, NOT a unified statistic** —
say this explicitly in any dashboard text, don't imply it's a p-value or
z-score comparable across flag types:
```
flagged_overclaim        -> (r_day - threshold) / threshold
flagged_night_injection  -> worst_ac_power_kw / tolerance_kw - 1
flagged_stuck_sensor     -> run_length / run_length_threshold - 1
```
All three mean "fractional excess past this flag's own trigger," which is
comparable in spirit (0 = barely triggered, larger = more decisive) but
not a rigorous unified confidence measure.

**Real example showing why severity and confidence must stay separate
(plant_2, replicate_000):**
```
2020-06-13  flagged_night_injection  value_at_risk_usd=137.28  confidence_ratio=8.56
2020-06-08  flagged_night_injection  value_at_risk_usd=0.42    confidence_ratio=0.29
```
Same flag type, wildly different severity AND confidence — collapsing
these into one composite score would hide this distinction from an
auditor.

---

## 6. Carryover caveats from Layers 2/3 that surface through the Layer 4 UI

These aren't new to Layer 4, but any dashboard view that shows model
performance, flags, or plant comparisons will surface them, so they need
to be visible, not just buried in a methods appendix:

- **plant_2's twin underperforms physics-only** in clear and
  partly-cloudy regimes (confirmed, unresolved overfitting — see
  LAYER2_HANDOFF.md §4). A naive "twin beats physics-only" dashboard claim
  is false for plant_2.
- **q50 is a monotone-constrained conditional MEAN, not a true median**
  (LightGBM's monotone-constraints mechanism rejects any
  piecewise-linear-gradient objective). If the dashboard labels
  `expected_q50` as "median expected output," that's imprecise — say
  "central estimate."
- **34-day dataset span** — any regime-level statistic shown for a small
  n (e.g. overcast, n=9-12) should show n alongside the number, per the
  project's own established rule (LAYER3_HANDOFF.md §7's TFT/plant_1
  overcast R²=-61.648 on n=7 example).
- **Achievable conformal alpha is ~16.7%, not the nominal target** — if
  the dashboard states a coverage guarantee, state the achieved number,
  not the configured target.

---

## 7. What 4.6 (dashboard) still needs, per the roadmap

Five views: project risk overview (consumes 3.6's `RiskScoreResult`
directly, no new computation), expected-vs-claimed time series with flags
overlaid, per-flag detail (consumes 4.4's evidence bundles + 4.5's
ranking), carbon/financial summary (consumes 4.3, must show both
mandatory caveats), and an evaluation view explicitly flagged in the
roadmap as a placeholder pending Task 7's full benchmark — don't block the
rest of the dashboard on that view being complete.
