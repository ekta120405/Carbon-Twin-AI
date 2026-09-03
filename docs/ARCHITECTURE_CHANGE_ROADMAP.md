# Architecture Change Roadmap — Cross-Conformal Calibration and Beyond

Status: **proposal only, nothing implemented.** This document is written
so a fresh Claude Code session — with no memory of the conversation that
produced it — can pick this up and execute it correctly, including
knowing what NOT to do and why.

---

## 0. How to use this document

Feed this whole file to a new session as the starting prompt, e.g.:
*"Read docs/ARCHITECTURE_CHANGE_ROADMAP.md and start on Workstream 1,
following its ground rules and validation plan exactly."*

Read `docs/detailed project flow.md` first (or have the new session read
it) for full system context — this roadmap assumes that level of
background and doesn't re-explain the whole architecture. This document
only covers what's new: three proposed structural changes, why they're
expected to help, and exactly how to build and validate them safely.

---

## 1. Why this roadmap exists

Across an earlier round of work, four different, honestly-validated
attempts were made to improve the two weakest detection categories
(A2_peak_hour_shaving, A6_slow_drift) **without** changing the twin model
or the detection architecture. All four converged on the same root cause
and were not shipped — full detail in `docs/ESCALATION_IMPROVEMENT_HANDOFF.md`
§8:

1. A peak-window residual signal — zero net improvement (noise inflation
   from the small calibration set canceled the signal concentration).
2. A faster-reacting CUSUM timescale — honest power pre-check showed only
   ~2pp gain against a 15pp bar, stopped before touching labels.
3. A combined-evidence test (standalone) — real gains on both weak
   categories, but hard-negative specificity collapsed (92.0%→62.9%).
4. The same test, gated conservatively — safe, but net zero, and worse on
   A2 specifically.

**The diagnosis, stated plainly**: every attempt was recalibrated the
same way as the existing system — a threshold fit from only **5 real
calibration days** (`calib_days: [21,25]`, `config/config.yaml`). This
gives an achievable false-positive-rate guarantee of ~16.7%, not the
nominal 5% (`conformal.py`'s own module docstring). A fifth idea —
pooling `test_clean` into the same threshold to get to n=10 — was
considered and explicitly rejected, because `test_clean` is this
project's last independent, never-calibrated sanity check on the
threshold (it already caught a real bug once: escalation flagging the
exact day that defines its own threshold — `ESCALATION_IMPROVEMENT_HANDOFF.md`
§4).

**This roadmap proposes the structural fix that respects that constraint
instead of fighting it**: don't pool more data into the *same* single
threshold-fit; change *how* the threshold is fit so it can use far more
of the real data while keeping an equivalent (or stronger) validity
guarantee.

---

## 2. Ground rules — read before writing any code

1. **Work in an isolated git branch or worktree, never directly on
   `master`.** This is a bigger, more central change than any prior
   addition (it touches the core threshold-computation path used by
   *everything* downstream — the dashboard, every `src/evaluation/*`
   script, every reported number). Nothing should land on `master` until
   validated and explicitly approved by the user.
2. **Never tune anything against `data/ground_truth/` directly.** Every
   parameter choice must use only real clean-data properties, or the
   honest tune(replicates 0–19)/holdout(replicates 20–29) split already
   established in this project (`src/evaluation/_experiment_split_validate.py`
   is the reusable template).
3. **The frozen injector does NOT need to be re-tagged for this work.**
   None of the three workstreams below touch the eligible injection pool
   (`test_clean`/`test_injected`, days 26–34) or the injector's own logic
   (`src/injection/`) — they only change how `train`/`calib` (days 1–25)
   are used internally. Confirm this remains true before building
   anything that changes the day-block boundaries themselves.
4. **`test_clean` stays untouched as an independent sanity check.**
   Workstream 1 gets more calibration data from `train`+`calib` via
   cross-validation, not by pooling `test_clean` in. The real-clean
   acceptance check (does the system score real, verified-clean data as
   "Low"?) must still pass using `test_clean` as an untouched holdout,
   same as it always has.
5. **Every workstream needs, before being considered done**: (a) the
   tune/holdout replicate validation, (b) the real-clean/real-full
   acceptance re-check, (c) a full N1–N7 hard-negative re-check, (d) a
   full re-run of `benchmark.py`, `mdo_curve.py`, `leave_one_plant_out.py`,
   and `ablations.py`, (e) a dashboard `AppTest` regression sweep, (f) a
   new handoff doc recording the honest result — whatever it turns out to be.
6. **Report negative results as rigorously as positive ones.** This
   project's own culture (stated repeatedly across its docs) treats an
   honestly-tested "didn't work, here's why" as equally valuable as a
   shipped improvement. Don't keep tuning until something looks good —
   follow the pre-declared validation plan and report what it says.

---

## 3. Current baseline — what any of this needs to beat, or at minimum not break

Pooled, both plants, all 30 replicates (540 day-instances, 193 positive),
production flag definition (`reports/tables/task7_benchmark_full.json`):

| Metric | Value |
|---|---|
| Precision | 0.817 |
| Recall | 0.648 |
| F1 | 0.723 |
| AUC-PR | 0.766 |
| Hard-negative specificity | 0.920 (161/175) |
| Achievable conformal alpha | 0.167 (nominal target: 0.05) |
| Real conformal thresholds | Plant 1 = 0.1666, Plant 2 = 0.2085 |
| Real-clean acceptance scores | Plant 1 = 12.1 (Low), Plant 2 = 20.2 (Low) |

Per-type recall for the two weak categories: A2_peak_hour_shaving =
25.9% (n=27), A6_slow_drift = 5.6% (n=18).

---

## 4. Workstream 1 (priority, do this first) — Cross-conformal calibration

### 4.1 The idea

Instead of one twin trained on `train` (20 days) with the threshold fit
on a separate, disjoint `calib` (5 days), use a **k-fold scheme across
the pooled `train`+`calib` days (25 total)**: for each fold, train a twin
on the other folds and compute residuals on the held-out fold. Pool all
the held-out residuals (~25 values instead of 5) to fit the threshold.

**Why this is valid, not just "more data pooled in"**: the entire reason
`test_clean` can't be pooled into the current single-fit threshold is
that doing so would eliminate the last check that the threshold isn't
contaminated by data the model was trained on. Cross-conformal doesn't
have that problem — every residual used for calibration comes from a
fold the twin never saw during that fold's training, by construction.
`test_clean` still stays completely untouched as an independent
real-data acceptance check, exactly as it does today.

### 4.2 Expected effect

Achievable alpha should move from ~16.7% toward something much closer to
the nominal 5% (n≈25 clears, or comes close to, the `n ≥ 19` floor
`conformal.py::min_n_for_nominal_alpha` already defines). This should
give real headroom to tighten the threshold, which should lift recall
across **multiple** categories (not just the two previously-targeted weak
ones), since it fixes the underlying calibration-noise problem every
prior attempt hit, rather than targeting one symptom of it.

### 4.3 Concrete implementation plan

- **New function in `src/layer3_anomaly/conformal.py`**: something like
  `fit_cross_conformal_threshold(pooled_residuals: pd.Series, alpha_nominal: float) -> ConformalResult`
  — reuse the existing order-statistic logic once n clears the floor
  (the function already has this branch, currently unreachable at n=5).
- **New training/scoring loop**, likely in `src/layer2_twin/twin_model.py`
  or a new small module, e.g. `src/layer2_twin/cross_validation.py`:
  k-fold split of days 1–25 (contiguous day-blocks per fold, not
  shuffled — same leakage-avoidance discipline as the existing splits),
  train `PIGBQRTwin` per fold, score the held-out fold, collect residuals.
- **`dashboard/lib/pipeline.py::load_plant_base`**: this currently loads
  ONE persisted twin per plant (`PIGBQRTwin.load()`) and fits the
  threshold from `calib_dates` only. Needs to either (a) also run the
  cross-validation loop once (cached, since it's deterministic) to
  produce the new threshold, while still loading the single production
  twin for actual scoring, or (b) restructure more significantly if the
  k-fold twins themselves need to be used differently. Keep the
  **production-scoring twin** as the one twin trained on the full
  `train` set (or reconsider whether it should now be an ensemble/average
  of the k fold twins — this is a real design decision to make
  explicitly, not silently default one way).
- **Everything downstream that reads `base["conformal_result"].threshold`**
  keeps working unmodified if the interface (`ConformalResult`) doesn't
  change shape — check `harness.py`, `escalation.py`, `mdo_curve.py`
  (which uses `r_day > threshold` directly, bypassing `classify_days()`),
  and `ablations.py`'s ablation (c) and (d) for any hardcoded assumption
  about `n_calib=5`.

### 4.4 Validation plan

1. Compute the new threshold, report `n_calib` (should be ~25) and the
   new `alpha_achievable` explicitly.
2. Full benchmark re-run — compare against the baseline in §3 for every
   metric, not just the headline ones.
3. Real-clean/real-full acceptance re-check — this is the single most
   important check for this specific workstream, since it's exactly the
   check class that would catch a subtle calibration-leakage bug (see
   `ESCALATION_IMPROVEMENT_HANDOFF.md` §4 for the precedent).
4. Full N1–N7 hard-negative re-check.
5. Re-run `mdo_curve.py`, `leave_one_plant_out.py`, `ablations.py` —
   check whether any of their own assumptions about the threshold's
   source break.
6. If the new threshold is tighter (lower), specifically re-examine
   whether hard-negative specificity holds — a lower threshold catches
   more true positives but is also more prone to catching legitimate
   under-generation edge cases near zero.
7. Write `docs/CROSS_CONFORMAL_CALIBRATION_HANDOFF.md` with the honest
   result, whatever it is.

### 4.5 Estimated effort

~4–6 hours of focused work, given the validation footprint (this touches
the core path everything downstream depends on).

---

## 5. Workstream 2 (depends on Workstream 1) — Multi-resolution testing

### 5.1 The idea

With a meaningfully larger calibration pool from Workstream 1, revisit
the peak-window idea from the earlier round — but this time, run it as a
genuinely **independent, separately-calibrated test** (its own threshold,
fit the same cross-conformal way), combined with the primary daily test
via a formal multiple-testing correction (e.g. Holm-Bonferroni across the
two tests) rather than an ad hoc OR-gate or near-threshold band.

**Why this failed before, and why it might not this time**: the earlier
peak-window attempt (`ESCALATION_IMPROVEMENT_HANDOFF.md` §8, attempt 1)
failed because narrowing the aggregation window increases residual noise
about as fast as it concentrates the signal, and the max-of-5-days
threshold convention inflated by nearly the same factor as the injected
anomaly — a calibration-data problem, not a conceptual flaw in the idea
itself. If Workstream 1 succeeds, the peak-window threshold would be
fit from the same much-larger pool, which directly addresses this
specific failure mode.

### 5.2 Do not attempt this before Workstream 1 lands and is validated

This is explicitly sequenced — building this on the current 5-day
calibration would very likely reproduce the exact same failure already
documented. If you're a fresh session picking this up and Workstream 1
hasn't shipped yet, do that first.

### 5.3 Estimated effort

~2–3 hours, given Workstream 1's cross-conformal machinery already exists
to reuse.

---

## 6. Workstream 3 (independent of the other two) — Replace CUSUM with an ARL-based sequential test

### 6.1 The idea

`src/layer3_anomaly/cusum.py`'s current design is explicitly scoped as
**only valid for the specific window length it's bootstrap-calibrated
for** (its own module docstring states this plainly — running a 9-day
`h` over a longer sequence was found to produce a false positive on real
clean data once already, fixed by recalibrating `h` fresh per window
length every time). This is a real architectural ceiling, not a tuning
problem: continuous, open-ended monitoring isn't supported at all.

Replace it with a **properly Average-Run-Length (ARL) calibrated**
sequential test — either a formally-calibrated CUSUM (calibrated for a
target in-control ARL rather than a fixed-window bootstrap FPR) or a
Sequential Probability Ratio Test (SPRT). This class of method is
designed from the ground up for continuous monitoring without a rigid
window boundary.

### 6.2 Why this might help A6 specifically

The earlier "faster CUSUM" attempt (`ESCALATION_IMPROVEMENT_HANDOFF.md`
§8, attempt 2) found that lowering `k` forces the FPR-calibrated `h`
back up almost proportionally, canceling most of the sensitivity gain —
but that finding is specific to the *bootstrap-FPR-at-fixed-window*
calibration procedure, not necessarily to sequential testing in general.
An ARL-based formulation calibrates differently (target mean time
between false alarms under the null, not a fixed-window empirical FPR)
and may not hit the same wall. This needs to be checked directly, not
assumed.

### 6.3 Validation plan

Same honest-first-check discipline as the original CUSUM power pre-check
(`src/evaluation/_experiment_cusum_fast_power.py` is the template): before
touching any label, simulate synthetic ramps (using only the injector's
published parameter ranges, `config/anomalies.yaml`) against bootstrap-
resampled real clean sequences, and check whether the new formulation's
detection power on short/subtle ramps clears a pre-declared bar. Only
proceed to labeled tune/holdout validation if it does.

### 6.4 Estimated effort

~3–4 hours — the calibration procedure itself is more novel to derive
than reusing the existing bootstrap machinery, but the change is more
self-contained (less entangled with the rest of the system) than the
other two workstreams.

---

## 7. Decision criteria — what makes any of this worth keeping

Mirror the standard already used throughout this project (see the CUSUM
fast-timescale pre-check's explicit 15-point bar as a precedent):

- **Workstream 1**: keep it if the achievable alpha genuinely tightens
  AND the real-clean acceptance check still passes AND hard-negative
  specificity doesn't regress. A tighter alpha with a real specificity
  cost is not an unambiguous win — report both numbers, let the user
  decide the tradeoff explicitly, don't default to "ship it" on your own.
- **Workstream 2**: keep it only if it improves A2 and/or A6 recall on
  the **untouched holdout replicates**, not just the tune set, with zero
  or clearly-justified specificity cost.
- **Workstream 3**: keep it only if the honest pre-check (§6.3) clears a
  pre-declared bar (suggest reusing the same 15-percentage-point
  threshold the earlier CUSUM attempt used, unless a better-justified
  number is derived first) before ever touching labeled data.

**In all three cases**: a rigorously-tested negative result is a valid,
complete outcome — write it up honestly (a new `*_HANDOFF.md` per
workstream, following the existing convention) rather than continuing to
adjust parameters until something looks good.

---

## 8. Reference material

| Topic | File |
|---|---|
| Full system architecture and every layer's internals | `docs/detailed project flow.md` |
| The four prior attempts and their exact root-cause findings | `docs/ESCALATION_IMPROVEMENT_HANDOFF.md` §8 |
| Conformal calibration internals | `src/layer3_anomaly/conformal.py` |
| CUSUM internals and its documented window-length limitation | `src/layer3_anomaly/cusum.py` |
| The twin model | `src/layer2_twin/twin_model.py` |
| Per-plant wiring / caching | `dashboard/lib/pipeline.py` |
| The shared evaluation harness every Task 7 script reuses | `src/evaluation/harness.py` |
| The honest tune/holdout validation pattern to reuse | `src/evaluation/_experiment_split_validate.py` |
| The frozen injector (do not modify) | `src/injection/taxonomy.py`, `src/injection/hard_negatives.py`, `config/anomalies.yaml` |
