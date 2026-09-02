# Secondary-Evidence Escalation — Handoff Notes

Status as of this commit: **built, validated (including two rounds of
honest self-correction), wired into production, and re-verified against
the full Task 7 evaluation suite plus the dashboard.** This is a
post-Task-7 improvement — Task 7 (7.0-7.6, `docs/TASK7_HANDOFF.md`) was
already complete and committed when this work started. It exists as its
own document, not folded permanently into `TASK7_HANDOFF.md`, because it's
a genuinely separate body of work with its own file list, its own bugs,
and its own validation story — the same reasoning that gives Layer 2, 3,
and 4 each their own handoff doc.

This document exists for the same reason every other `*_HANDOFF.md` in
this project does: several real things were found while building this
that aren't obvious from reading the final code, and whoever touches this
feature next needs them, or a change will silently reintroduce a bug that
was already found and fixed once.

---

## 0. One-paragraph summary

Two existing Layer 3 components — the autoencoder (3.4) and CUSUM (3.5) —
carry real fraud-detection signal that `classify_days()` (3.3) never
uses. Blindly trusting either one as a blanket OR was already tested in
Task 7 (ablations (c)/(f)) and found to cost more in new false positives
than the recall gain was worth. This work asks a narrower question
instead — can either one help WITHOUT that cost, if only trusted on a day
that's already borderline on the main test? Answer: yes, but getting
there required catching a methodology mistake (tuning against the same
data used for final scoring) and a real bug (a day could be flagged that
was literally the calibration day that defined the threshold itself)
before the result could be trusted. Final, shipped state: pooled recall
58.5%→64.8%, F1 67.7%→72.3%, hard-negative specificity unchanged at
92.0% (literally the same 28 false positives, not just the same count).

---

## 1. File locations

```
src/layer3_anomaly/escalation.py               the escalation logic itself
src/layer3_anomaly/autoencoder.py              train_and_calibrate(), save/load_autoencoder(),
                                                compute_daily_ae_flag() added (existing module extended)
src/layer3_anomaly/train_autoencoder_model.py  production training driver (NEW -- none existed before)

dashboard/lib/pipeline.py                      get_scored_context() wired to call escalation
src/evaluation/harness.py                      score_dataframe() wired to call escalation (same logic, 2nd call site)

config/config.yaml                             layer3.escalation: enabled/band_width/cusum_frac,
                                                with the full calibration story in the comments --
                                                read this before changing either number

reports/models/<plant>/autoencoder.pt          persisted weights -- GITIGNORED, see SS6 below
reports/models/<plant>/autoencoder_meta.json   persisted threshold/run_length/scaler/input_cols -- committed

src/evaluation/_experiment_ae_runlength.py     exploratory script -- KEPT, see SS5
src/evaluation/_experiment_cusum_smart.py      exploratory script -- KEPT, see SS5
src/evaluation/_experiment_split_validate.py   exploratory script -- KEPT, see SS5
reports/tables/_experiment_*.json              their outputs -- KEPT, see SS5

reports/tables/task7_benchmark_full.json       regenerated -- reflects escalation now
reports/tables/leave_one_plant_out.json        regenerated -- reflects escalation now
reports/tables/ablations.json                  regenerated -- reflects escalation now
reports/tables/mdo_curve.json                  regenerated but numerically UNCHANGED (bypasses labels)
reports/tables/stuck_sensor_fix_reattempt.json regenerated but numerically UNCHANGED (independent code path)

docs/TASK7_HANDOFF.md SS6                      short pointer to this document, not a duplicate
docs/TASK7_FINDINGS.md                         updated throughout to the post-escalation numbers
docs/PROJECT_RESULTS_SUMMARY.md SS7            the same story, in plain language for a non-technical reader
docs/HOW_THE_SYSTEM_WORKS.md SS6.8             the same story, as part of the plain-language architecture walkthrough
```

**Nothing in `src/layer3_anomaly/escalation.py` reimplements Layer 3
detection logic.** It only reads `daily["r_day"]`, an already-computed
CUSUM statistic, and an already-computed autoencoder day-flag, and
upgrades a label. All the actual detection math still lives where it
always did.

---

## 2. What the escalation logic actually does

`apply_secondary_evidence_escalation()` upgrades a day already classified
`"normal"` by `classify_days()` to `"flagged_overclaim"` ONLY if:

1. Its `r_day` sits close below the main conformal threshold — strictly
   between `band_width * threshold` and `threshold` (both bounds
   exclusive at the top; see SS4 for why the upper bound must be `<`, not
   `<=`).
2. AND at least one secondary signal corroborates:
   - the autoencoder's calibrated run-length flag
     (`autoencoder.compute_daily_ae_flag()` — a run of `run_length`
     consecutive over-threshold DAYLIGHT reconstruction-error timesteps,
     not the earlier "any single timestep" rule Task 7's ablation (f)
     still uses deliberately, as a documented baseline comparison), OR
   - CUSUM's own running statistic exceeding `cusum_frac * h`.

It NEVER touches `flagged_night_injection`, `flagged_stuck_sensor`, or
`fault_underclaim` — only ever upgrades `"normal"`.

**Where it runs, and why it must run before severity:** wired into both
`dashboard/lib/pipeline.py`'s `get_scored_context()` and
`src/evaluation/harness.py`'s `score_dataframe()`, in both cases
immediately after `classify_days()` and CUSUM are computed, but BEFORE
`compute_daily_flag_severity_kwh()` — every downstream number (risk
score, carbon exposure, evidence bundles, and every Task 7 evaluation
script) needs to see the escalated labels, not the pre-escalation ones.
Both call sites run the identical logic deliberately (not two divergent
implementations) — if you ever change one, change the other.

---

## 3. The autoencoder finally has a real production life

Confirmed via `grep` before this work started: no training driver or
persisted model existed anywhere in the committed codebase for the
autoencoder (LAYER3_HANDOFF.md discusses its behavior in detail, but that
was from an earlier, uncommitted throwaway run). Three additions to
`src/layer3_anomaly/autoencoder.py` fixed this:

- **`train_and_calibrate()`** — THE canonical training+calibration path.
  Trains on `train`, then calibrates TWO quantities from `calib`+
  `test_clean` ONLY (never injected data):
  - `threshold`: 95th percentile of reconstruction error over ALL clean
    rows (day+night).
  - `run_length`: the longest run of consecutive over-threshold
    reconstruction-error timesteps ever observed on that SAME clean pool,
    **restricted to daylight rows**, plus one.

  **A real bug caught while writing this**, worth restating precisely
  because it's exactly the kind of thing that's easy to silently
  reintroduce: the threshold calculation legitimately uses ALL clean rows
  (day+night — matches the original ablation-study code), but the
  run-length calculation MUST be daylight-only, because the day-level
  flag itself (`compute_daily_ae_flag`) only ever looks at daylight
  timesteps. The first draft of the consolidated function used the same
  (unfiltered) pool for both quantities. Caught by checking the new
  function's output against already-known values from the standalone
  experiment scripts (plant_1 run_length=29, plant_2=14) — they only
  matched once the daylight filter was restored for the run-length
  calculation specifically. If you ever refactor this function, re-run
  `python -m src.layer3_anomaly.train_autoencoder_model --plant all` and
  confirm the printed run_length values still match 29 / 14 before
  trusting anything downstream.

- **`save_autoencoder()` / `load_autoencoder()`** — text/JSON-based
  persistence (model weights via `torch.save`'s state dict, everything
  else — including the fitted scaler's mean/scale — as plain JSON
  floats), consistent with this project's established "never pickle, for
  version stability" convention (LAYER2_HANDOFF.md SS1). See SS6 below for
  why the weights file itself still can't be committed despite this.

- **`compute_daily_ae_flag()`** — the run-length-aware day-level flag,
  used by escalation. Distinct from `src/evaluation/ablations.py`'s
  `score_replicate_autoencoder()`, which deliberately keeps the OLD "any
  single timestep" rule — that function is Task 7 ablation (f)'s
  documented baseline comparison and must NOT be changed to match this
  one, or the ablation stops measuring what it says it measures.

`src/layer3_anomaly/train_autoencoder_model.py` is the new production
training driver:
```
python -m src.layer3_anomaly.train_autoencoder_model --plant plant_1
python -m src.layer3_anomaly.train_autoencoder_model --plant plant_2
python -m src.layer3_anomaly.train_autoencoder_model --plant all
```
Verified output on the real data (re-run this and confirm these match
before trusting a retrain):
```
[plant_1] calibrated: threshold=0.8826, run_length=29
[plant_2] calibrated: threshold=2.7492, run_length=14
```

---

## 4. A real regression, caught before being trusted: flagging the threshold's OWN calibration day

After wiring escalation into `get_scored_context()`, re-checking this
project's own long-standing "real, verified-clean data should score Low
with few or no flagged days" acceptance property (LAYER3_HANDOFF.md SS7;
also the dashboard demo's strongest moment, DASHBOARD_GUIDE.md step 7)
surfaced a real problem: plant_1's real 10-day clean block picked up a
NEW `flagged_overclaim` day that had never been flagged before —
`2020-06-03`, and it turned out to be the exact `calib` day whose
`r_day` **defines** the conformal threshold itself
(`threshold = max(calib r_day)`, `conformal.py`'s
`fit_conformal_threshold()`). Confirmed via `repr()` that `r_day ==
threshold` exactly (`0.1666345522267894 == 0.1666345522267894`, diff
`0.0`) — not a display-rounding illusion, a genuine exact-equality case.

**Root cause:** `near_threshold`'s upper bound originally used
`<= conformal_threshold`. A day sitting exactly AT the threshold
satisfies that inclusively, even though `classify_days()`'s own primary
test correctly uses strict `>` and would never call that day anomalous —
it's definitionally the most extreme example of "normal" the threshold
was calibrated from, not evidence against it.

**Fix:** changed to strict `<` in `escalation.py`'s `near_threshold`
(see the comment inline there — this is now a load-bearing detail, not
cosmetic). **Never loosen that comparison back to `<=` without
re-checking this exact scenario first** — re-run the real-clean check in
SS7 below.

---

## 5. The parameter-selection story, including a mistake worth learning from — and why the exploratory scripts are kept, not deleted

Three rounds, each one correcting the last. All three are preserved as
real code (`src/evaluation/_experiment_*.py`, underscore-prefixed to mark
them as exploratory rather than part of the maintained pipeline) rather
than deleted after the fact, specifically so this story is reproducible,
not just narrated:

1. **`_experiment_ae_runlength.py`, then `_experiment_cusum_smart.py`
   (first pass)**: swept `band_width`/`cusum_frac` by directly testing
   their effect against the SAME 30 labeled replicates used for final
   scoring, kept whatever scored best. Found an apparently "free" result
   (recall +11pp, zero specificity cost). **This broke the project's own
   rule, followed everywhere else, of never tuning against
   `TEST-INJECTED`** (Overview v2 SS8.1) — caught and disclosed before
   being trusted, not after, once the pattern was recognized as identical
   in shape to "tasting the dish before serving it to the judges."

2. **`_experiment_split_validate.py` (second pass, honest split
   validation)**: tuned `band_width`/`cusum_frac` on replicates 0-19
   only, verified on the untouched replicates 20-29. The core idea DID
   generalize (holdout F1 0.716→0.794), but with a real, disclosed
   specificity cost this time (0.867→0.833) — the "free" result from pass
   1 did not survive honest validation, confirming the concern was
   justified, not hypothetical. Chosen pair at this point:
   `band_width=0.7, cusum_frac=0.0`.

3. **Production wiring + a check pass 2 never ran (third pass)**: wiring
   pass 2's parameters into `dashboard/lib/pipeline.py` and re-checking
   the real-clean/real-full acceptance blocks (SS4 above) found a NEW cost
   that neither pass 1 nor pass 2's replicate-based validation ever
   tested against — real, hand-verified honest data is a genuinely
   different check than "an unseen replicate from the same frozen
   injector," and passing the latter is not proof of the former. Backed
   off to the more conservative `band_width=0.8, cusum_frac=0.5`
   (`config.yaml`'s actual, shipped values) specifically because it
   restores the acceptance blocks to a clean "Low" band on both plants
   while still keeping a real, substantial pooled gain (F1 0.677→0.723)
   at zero specificity cost on the pooled hard-negative test.

**The lesson, stated plainly for whoever touches this next:**
"generalizes to held-out synthetic test cases from the same generator"
and "behaves correctly on real data" are two different claims, and this
project's 30-replicate pool cannot substitute for checking the second one
directly. **If `layer3.escalation`'s parameters are ever re-tuned, re-run
both checks — the replicate-based benchmark AND the real-clean/real-full
sweep in SS7 below — before trusting any new result, not just the
replicate one.**

Do not delete the `_experiment_*.py` scripts or their
`reports/tables/_experiment_*.json` outputs — they're the actual,
re-runnable evidence behind this section, not just a narrative.

---

## 6. Operational note: the `.pt` file is gitignored, on purpose

`reports/models/<plant>/autoencoder.pt` is covered by the repo's existing
`.gitignore` rule (`*.pkl`, `*.pt`, `*.txt.model` are all excluded
together) — the same policy that already keeps this project's LightGBM
twins in text format instead of pickled (LAYER2_HANDOFF.md SS1: pickled
binaries aren't guaranteed to load correctly across different
library/Python versions or machines; `torch.save()` is pickle-based under
the hood, so it falls under the identical concern). This is a **policy
decision, not a file-size limit** — the actual file is 5.7KB, nowhere
near any host's size threshold.

**Practical consequence:** on a fresh checkout, `reports/models/<plant>/autoencoder.pt`
will not exist. `autoencoder_meta.json` (small, plain JSON) IS committed,
so its presence without a matching `.pt` file is an easy, visible signal
that the training step still needs to run. This is handled gracefully by
`load_plant_base()` — `base["autoencoder"]` is `None` when the file is
missing, and both call sites check for that and simply skip escalation
rather than crashing; the system still runs correctly, just falling back
to the pre-escalation Task 7 baseline numbers. **Run this once after any
fresh checkout, before expecting escalation to do anything:**
```
python -m src.layer3_anomaly.train_autoencoder_model --plant all
```

---

## 7. Final, verified results

**Pooled (both plants, all 30 replicates each), production flag:**

| Metric | Pre-escalation | Post-escalation |
|---|---|---|
| n | 540 (193 positive, 347 negative) | 540 |
| Precision | 0.801 | **0.817** |
| Recall | 0.585 | **0.648** |
| F1 | 0.677 | **0.723** |
| AUC-PR | 0.725 | 0.766 |
| Hard-negative specificity | 0.920 (161/175) | **0.920 (161/175) — identical, same 28 FPs** |

**Per-type recall, notable movers:** A1 constant-overclaim 69.0%→75.9%;
A2 peak-hour-shaving 14.8%→25.9%; A4 day-swap 58.1%→74.2%; A5
weather-spoofing 66.7%→70.0%; A7 capacity-violation 70.4%→74.1%. A3/A8
unchanged at 100% (guardrail-driven, untouched by escalation by
construction). A6 slow-drift essentially unchanged (5.6%) at this final,
conservative calibration — the one category this work did not manage to
meaningfully help.

**Every per-type hard-negative specificity is byte-for-byte identical
before/after** (confirmed via the same-28-false-positives check, not just
matching counts) — the escalation logic never fires on a hard-negative
day at this calibration.

**Real-clean/real-full acceptance check** (the check that drove the final
parameter choice, SS5 point 3): both plants' `real_clean` and `real_full`
sources score in the "Low" risk band, matching pre-escalation behavior,
with one small, persistent, strongly-corroborated exception on plant_1
that does not change its band.

**Downstream re-verification performed, all passed:**
- `python -m src.evaluation.benchmark` — re-run, numbers above confirmed.
- `python -m src.evaluation.mdo_curve` — re-run, numerically UNCHANGED
  (confirms it bypasses `classify_days()`/escalation by design — it uses
  `r_day > threshold` directly).
- `python -m src.evaluation.leave_one_plant_out` — re-run; cross-plant
  asymmetry pattern holds, escalation mildly amplifies plant_2-twin's
  already-poor generalization onto plant_1 (specificity 0.307→0.159 for
  that specific cross-twin case) — an expected, not surprising,
  interaction (a badly-miscalibrated twin produces more borderline days
  for escalation to act on).
- `python -m src.evaluation.ablations` — re-run; ablation (c)'s
  `fixed_2sigma` arm also now includes escalation (applied relative to
  ITS OWN threshold, since `score_dataframe` applies escalation
  unconditionally relative to whatever `base["conformal_result"].threshold`
  it's given) — still a fair, like-for-like comparison of the underlying
  threshold choice, just with the same enhancement layered onto both
  arms; ablation (d) is unaffected by design (built from raw `r_day`
  comparisons, bypasses `classify_days()`/escalation entirely); ablation
  (f)'s naive-autoencoder-rule baseline is deliberately UNCHANGED (still
  the old "any timestep" rule, since that's the documented comparison
  point escalation's smarter rule is measured against).
- `python -m src.evaluation.stuck_sensor_fix` — re-run, numerically
  UNCHANGED (independent code path, doesn't touch `classify_days()`
  output at all).
- Full dashboard regression via `streamlit.testing.v1.AppTest` across 6
  plant/source combinations (both plants × `injected_010`/`injected_000`,
  `real_clean`, `real_full`) — zero exceptions, both before and after the
  final conservative parameter choice.
