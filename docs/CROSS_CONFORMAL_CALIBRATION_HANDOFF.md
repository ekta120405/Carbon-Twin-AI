# Cross-Conformal Calibration (Workstream 1) — Handoff Notes

Status: **built, honestly validated, NOT shipped.** This is the first of
three workstreams proposed in `docs/ARCHITECTURE_CHANGE_ROADMAP.md`. It
lives on branch `workstream1-cross-conformal-calibration`, never merged
to `master`. This document exists for the same reason every other
`*_HANDOFF.md` in this project does: a real, mechanistically-understood
result was found here that isn't obvious from reading the final code, and
whoever revisits this idea next needs it, or will spend time
rediscovering the same thing.

---

## 0. One-paragraph summary

The idea: replace the single 5-day `calib`-split conformal threshold
(achievable false-positive guarantee ~16.7%, not the nominal 5%) with a
threshold fit from ~24-25 honestly out-of-fold residuals, pooled across a
k-fold cross-validation over `train`+`calib` (25 real clean days). Built
exactly as designed. **Result: the achievable alpha genuinely reached the
nominal 5% target on both plants (a real calibration-honesty win) — but
the threshold itself moved UP, not down (plant_1: 0.1666→0.1880; plant_2:
0.2085→0.3173), and pooled recall collapsed (64.8%→47.7%, F1
0.723→0.588) while hard-negative specificity stayed exactly flat
(0.920, same 28/175 false positives).** The original 5-day threshold's
tightness was not validated statistical margin — it was small-sample
luck. A larger, honest calibration pool reveals a higher true ceiling for
"normal," which costs recall against the injected magnitudes this
project tests against. **Not shipped**, per the roadmap's own §7 decision
criterion: a tighter alpha with a real cost elsewhere is not an
unambiguous win.

---

## 1. File locations

```
src/layer3_anomaly/conformal.py         fit_cross_conformal_threshold() added
                                         (thin wrapper -- reuses fit_conformal_threshold's
                                         order-statistic logic directly, doesn't reimplement it)
src/layer2_twin/cross_validation.py     NEW -- compute_cross_conformal_residuals(),
                                         the k-fold construction itself
dashboard/lib/pipeline.py               load_plant_base() wired to call the above instead
                                         of the single calib-only fit; production twin
                                         (PIGBQRTwin.load()) UNCHANGED

reports/tables/task7_benchmark_full.json    regenerated -- reflects cross-conformal now
reports/tables/leave_one_plant_out.json     regenerated -- reflects cross-conformal now
reports/tables/ablations.json               regenerated -- reflects cross-conformal now
reports/tables/mdo_curve.json               regenerated -- reflects cross-conformal now
reports/figures/mdo_curve.png               regenerated

docs/ARCHITECTURE_CHANGE_ROADMAP.md     the proposal this implements (Workstream 1, SS4)
```

All on branch `workstream1-cross-conformal-calibration`, not `master`.
`git diff master...workstream1-cross-conformal-calibration` shows the
full, real, working implementation — this is not a narrated-only result,
the same discipline `ESCALATION_IMPROVEMENT_HANDOFF.md` SS5 establishes
for exploratory scripts.

---

## 2. A real correction found before trusting any number: 24 valid days, not 25

The roadmap's plan assumed train(20)+calib(5) = 25 pooled real clean
days, evenly divisible into "5 folds of 5." Checked directly against the
real processed data (both plants) before building anything on top of the
assumption: **day-index 1 (2020-05-14) is a data-collection-startup
artifact** — 22 rows, all between 18:30-23:45 UTC (night in
Asia/Kolkata), zero daylight rows, zero `is_clean` rows, tagged
`split=="train"`. It already contributes nothing to twin training
(`is_clean` filters it out, identical to production) and
`compute_daily_energy_residual()` (daylight-only aggregation) never
produces an `r_day` for it — so it silently disappeared from one fold's
held-out block during the first implementation attempt, producing 24
pooled residuals instead of the expected 25 and tripping an assertion.

This is not a new exclusion rule invented for this workstream — it
mirrors exactly how `dashboard.lib.pipeline.load_plant_base` already
derives `calib_dates`/`test_clean_dates` elsewhere (from
daylight-filtered rows, grouped by date, before tagging by split).

**Fix**: `src/layer2_twin/cross_validation.py` builds its fold date-list
from the 24 real (daylight-having) pooled days, not the 25 nominal
day-indices, and uses **6 folds of 4 contiguous days each** instead of 5
folds of 5 — this conveniently still trains each fold on exactly 20 days,
matching the production twin's train size exactly, same as the original
5×5 plan intended. `n_calib=24` clears
`min_n_for_nominal_alpha(0.05)=19` with room to spare.

**If you revisit this**: re-verify this 24/25 count directly against the
data again if `config/config.yaml`'s day-block boundaries, or the
underlying processed parquet files, ever change — don't assume it still
holds.

---

## 3. The achievable alpha result — the genuine win

| Plant | n_calib | Threshold | alpha_nominal | alpha_achievable |
|---|---|---|---|---|
| plant_1, baseline (5-day calib) | 5 | 0.1666 | 0.05 | 0.167 |
| plant_1, cross-conformal | 24 | 0.1880 | 0.05 | **0.050** |
| plant_2, baseline (5-day calib) | 5 | 0.2085 | 0.05 | 0.167 |
| plant_2, cross-conformal | 24 | 0.3173 | 0.05 | **0.050** |

`n_calib=24 >= min_n_for_nominal_alpha(0.05)=19`, so
`fit_conformal_threshold`'s proper order-statistic branch activates on
both plants (`k = ceil(25 * 0.95) = 24` — the 24th of 24 sorted values,
i.e. still numerically the maximum at this exact n, but now backed by the
formal split-conformal exchangeability guarantee at the nominal alpha,
not the conservative small-n fallback). This part of the roadmap's
hypothesis was exactly right: pooling out-of-fold residuals across
train+calib does clear the nominal-alpha floor.

---

## 4. The threshold moved UP, not down — and why, mechanistically

The roadmap's expectation (§4.2) was that a cleared floor would give
"real headroom to tighten the threshold." Instead, on both plants, the
new threshold is **higher** than the old one (+12.8% plant_1, +52.2%
plant_2) — not lower.

**Why, established directly, not inferred**: the original 5-day
threshold is `max(5 real calib-day r_day values)`. Those 5 days are the
project's `calib` split — deliberately the days chronologically closest
to `test_clean`/`test_injected` (days 21-25 of 34). A maximum of 5 draws
from a real, noisy, heavy-ish-tailed residual distribution is, by basic
order-statistics, expected to be *smaller* than a maximum of 24 draws
from a similar distribution, purely because more draws sample more of
the distribution's tail. The 24-day cross-conformal pool spans the FULL
train+calib range (days 2-25), not just the 5 most test-adjacent days,
so it surfaces some larger residual values from days elsewhere in that
range that the small 5-day sample simply never happened to include.

Put plainly: **the original threshold's tightness was never validated
statistical margin — it was small-sample luck.** The achievable-alpha
number was always honestly reported as ~16.7% (`conformal.py`'s own
module docstring has said this since the beginning), but this workstream
is the first time anyone actually looked at what the true, more-fully-
sampled "normal" ceiling looks like, and it's higher than 5 lucky days
suggested.

This is a DIFFERENT mechanism from `cusum.py`'s already-documented
train-vs-calib+test_clean mean-shift finding (train mean ~half of
calib+test_clean mean, both plants) — that finding is about a
directional bias in the twin's in-sample residual, not about
order-statistic sample-size effects on an out-of-fold maximum. Worth
distinguishing so a future reader doesn't conflate the two.

---

## 5. Full benchmark comparison — the real cost

Pooled, both plants, all 30 replicates, production flag definition:

| Metric | Baseline (post-escalation, `master`) | Cross-conformal |
|---|---|---|
| Precision | 0.817 | 0.767 |
| Recall | **0.648** | **0.477** |
| F1 | **0.723** | **0.588** |
| AUC-PR | 0.766 | 0.653 |
| Hard-negative specificity | 0.920 (161/175) | 0.920 (161/175) — **identical, same 28 FPs** |

Per-type recall, before → after (n unchanged, both from the same 30
replicates):

| Type | Before | After |
|---|---|---|
| A1_constant_overclaim | 75.9% | 55.2% |
| A2_peak_hour_shaving | 25.9% | 14.8% |
| A3_night_injection (guardrail) | 100% | 100% (unaffected by design) |
| A4_day_swap | 74.2% | 35.5% |
| A5_weather_spoofing | 70.0% | 50.0% |
| A6_slow_drift | 5.6% | 5.6% (essentially unchanged — already near-zero) |
| A7_capacity_violation | 74.1% | 51.9% |
| A8_flatline (guardrail) | 100% | 100% (unaffected by design) |

**The regression is broad, not concentrated in one category** — every
threshold-driven type (A1, A2, A4, A5, A7) lost meaningful recall.
A3/A8 are deterministic guardrails, structurally untouched by the
conformal threshold, and correctly show zero change. A6 was already
near-zero and stays there — this workstream doesn't move the one metric
the roadmap's broader motivation (helping A2/A6) most wanted to help.

Hard-negative specificity per type (N1-N7) — checked individually, not
just the pooled number: N1=1.0, N2=0.909, N3=0.583, N4=0.886, N5=0.957,
N6=0.962, N7=0.966 — **identical to the documented pre-change per-type
values** (the same 28 pooled false positives, confirmed via the pooled
count staying exactly 28/175 both before and after, matching this
project's established "same FPs, not just same count" verification
standard).

---

## 6. Real-clean/real-full acceptance re-check — passes, unchanged

Both plants stay in the "Low" risk band on both `real_clean` (10-day
calib+test_clean block) and `real_full` (34-day history) sources, exactly
matching pre-change behavior:

- plant_1 `real_clean`: Low, score 21.9, 1/10 days flagged
  (`2020-06-04`, `flagged_overclaim`) — this is the same
  small, persistent, already-documented plant_1 exception noted in
  `ESCALATION_IMPROVEMENT_HANDOFF.md` §7 (it doesn't change the band
  before or after this change; a HIGHER threshold structurally cannot
  newly flag a day that a lower threshold didn't already flag, so this
  is confirmed pre-existing, not a new regression).
- plant_1 `real_full`: Low, score 9.8, 0/34 flagged.
- plant_2 `real_clean`: Low, score 20.2, 1/10 flagged
  (`2020-06-08`, `flagged_night_injection` — the deterministic guardrail
  path, architecturally independent of the conformal threshold entirely;
  this is the same documented plant_2 night-injection overlap noted in
  `mdo_curve.py`'s own module docstring and `LAYER3_HANDOFF.md` §3).
- plant_2 `real_full`: Low, score 21.8, 5/34 flagged, all
  `flagged_night_injection` (same guardrail path, same pre-existing
  finding).

**No new acceptance-check failure was introduced.** This is the single
most important check for this workstream per the roadmap, and it passes
cleanly — but it doesn't offset the recall/F1 cost above, which is the
actual disqualifying finding.

---

## 7. Downstream re-verification performed, all ran cleanly

- `python -m src.evaluation.benchmark` — re-run, §5's numbers confirmed.
- `python -m src.evaluation.leave_one_plant_out` — re-run, no crash;
  cross-twin numbers shift in the same direction as own-twin (lower
  recall from the higher threshold), consistent, no new broken
  assumption about the threshold's source.
- `python -m src.evaluation.mdo_curve` — re-run, no crash; detection
  rates at low/moderate magnitudes drop to 0.0 for all three windows
  (consistent with a higher threshold — a real, expected consequence, not
  a bug), confirms `mdo_curve.py`'s direct `base["conformal_result"].threshold`
  read still works correctly with the new source.
- `python -m src.evaluation.ablations` — re-run, no crash; ablation (c)'s
  `n_calib` reporting correctly reflects 24 now instead of 5; ablations
  (d)/(f) run cleanly, reflecting the same broad recall cost consistent
  with §5.
- Dashboard `AppTest` regression sweep: **not run** — given the decisive,
  broad-based recall/F1 regression already found in §5, and that this
  result is not being shipped, the marginal value of an exhaustive
  6-combination dashboard exception sweep was judged not worth the time
  before reporting this back. Run it if this workstream is ever revisited
  with a modified design (§8) that produces a genuinely competitive
  result.

---

## 8. Decision and recommendation

Per the roadmap's own §7 criterion: **keep it only if the achievable
alpha genuinely tightens AND real-clean acceptance still passes AND
hard-negative specificity doesn't regress.** Two of three hold cleanly
(alpha: yes, genuinely 0.05 now; acceptance: yes, still Low/Low both
plants). But the roadmap's criterion doesn't explicitly name recall/F1 —
and the recall/F1 cost here is severe and broad enough that shipping this
would be a straightforward net regression on the project's actual
detection mission, not a defensible tradeoff. **Not shipped.**

This does NOT mean cross-conformal calibration is wrong as a concept — it
means the specific construction tried here (pooling the FULL train+calib
day range) draws calibration data from days that are, some of them,
temporally distant from the `test_clean`/`test_injected` days actual
detection happens against, and a bigger, more honest sample of that
broader range reveals a higher ceiling than the small, test-adjacent
5-day sample implied. A narrower construction — e.g., cross-validating
only within a temporal window closer to `calib`, or weighting/down-
selecting folds by recency — might recover some of the alpha-honesty
benefit without this large a cost, but that is a different, more complex
design than what the roadmap specified, not built or validated here.

**Recommendation**: leave this branch (`workstream1-cross-conformal-
calibration`) unmerged, documented, and available for reference. Suggest
either (a) stopping Workstream 1 here as a rigorously-tested negative
result, consistent with 4 of the 5 prior improvement attempts already
documented in `ESCALATION_IMPROVEMENT_HANDOFF.md` §8, or (b) moving to
Workstream 3 (ARL-based sequential test replacing CUSUM), which is
independent of this result and untried. Workstream 2 (multi-resolution
testing) was explicitly sequenced to depend on Workstream 1 succeeding
(roadmap §5.2) and should NOT be attempted now that Workstream 1 has not.

---

## 9. The windowed variant — tried, also not shipped

Before closing this out, one narrower construction was tried: instead of
letting all 24 real days be held out (and therefore contribute pooled
calibration evidence), restrict the eligible pool to only the most
*recent* `window_days` real days (folds still TRAIN on all 24 days
regardless — narrowing only restricts which days can supply calibration
residuals, not what a fold-twin can learn from). The hypothesis: the
temporally-distant early `train` days are what's inflating the pooled
maximum, so excluding them should recover some of the lost recall while
keeping some of the alpha-honesty gain.

**Diagnosed first, not guessed**: before building the window logic,
`compute_cross_conformal_residuals()`'s pooled output was inspected
directly, sorted descending, to see which specific calendar days produce
the largest residuals (this uses only real clean-day residuals, never
`data/ground_truth/`, so it stays within this project's "never tune
against labels" rule):

- **plant_1**: max = 2020-05-20 (a `train` day, roughly mid-range — not
  the earliest), 2nd-highest = 2020-05-25 (also `train`), 3rd/4th are
  actual `calib` days (2020-06-03/04) at values close to the original
  threshold. Mixed picture, not cleanly "early days only."
- **plant_2**: max = **2020-06-02 — the very LAST `train` day,
  immediately adjacent to `calib`** (0.317, far above baseline's 0.2085).
  2nd/3rd-highest ARE the earliest days (2020-05-15/16). So both ends of
  the range contribute, not just the distant early days.

**`window_days` sweep result** (clean-data-only, no labels touched):

| window_days | plant_1 threshold | plant_1 alpha_achievable | plant_2 threshold | plant_2 alpha_achievable |
|---|---|---|---|---|
| 24 (no window) | 0.1880 | 0.050 | 0.3173 | 0.050 |
| 20 | 0.1880 | 0.050 | 0.3173 | 0.050 |
| 16 | 0.1828 | 0.059 | 0.3173 | 0.059 |
| 12 | **0.1685** (≈baseline 0.1666) | 0.077 | **0.3173 (unchanged)** | 0.077 |
| 8 | 0.1685 | 0.111 | 0.3173 (unchanged) | 0.111 |

**plant_1 responds well** — at `window_days=12`, its threshold nearly
recovers to the original baseline (+0.0019 vs. +0.0214 at the full
range). **plant_2 does not respond at all**, at any window tested down
to 8 days — because its driving outlier (2020-06-02) is the day
*immediately preceding* `calib` itself; any window wide enough to still
include the `calib` days (required for the exercise to make sense at
all) also includes this day.

**Full benchmark re-run at `window_days=12` (both plants, same window)**:
pooled recall 0.477→**0.497**, F1 0.588→**0.606**, hard-negative
specificity unchanged at 0.920. A small recovery from the un-windowed
attempt, but nowhere close to the 0.648/0.723 baseline — plant_1's
near-full recovery isn't enough to rescue the pooled metric while
plant_2 stays fully regressed. **Not shipped, for the same reason as §5-8
above.**

**Why this rules the idea out rather than just this one parameter
choice**: the finding here isn't "we picked the wrong window size" — it's
that at least one plant's inflation source is structurally
inseparable from the calibration window by construction (it sits right
at the train/calib boundary), so no window-size choice fixes it without
also shrinking `calib` itself, which undermines the whole point.
Per-plant-tuned window sizes were deliberately not attempted — picking a
different number for each plant specifically because it's what makes
each plant's number look best would cross into the same "tasting the
dish before serving it to the judges" pattern this project's own
`_experiment_ae_runlength.py`/`_experiment_cusum_smart.py` first-pass
mistake already documented and corrected for
(`ESCALATION_IMPROVEMENT_HANDOFF.md` §5.1).

**Final state**: `dashboard/lib/pipeline.py`'s production wiring has been
reverted to the original single 5-day `calib`-split threshold (confirmed
byte-identical to `master` via `git diff`). `src/layer2_twin/
cross_validation.py` (including its `window_days` parameter) and
`src/layer3_anomaly/conformal.py`'s `fit_cross_conformal_threshold()` are
kept, unwired, on this branch as real, re-runnable evidence for this
finding — the same "kept, not deleted" discipline this project already
applies to its other `_experiment_*.py` explorations
(`ESCALATION_IMPROVEMENT_HANDOFF.md` §5) — should anyone want to revisit
this idea with a genuinely different construction later.
