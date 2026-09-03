# SPRT Sequential Test — Handoff Notes

Status: **built, honestly validated (clean-data pre-check AND labeled
tune/holdout), NOT shipped.** Lives on branch `sprt-sequential-test`,
never merged to `master`. This is the most thoroughly validated of the
three architecture-level explorations this session (Workstreams 1, 3,
and this one) — it's the only one that reached real labeled tune/holdout
validation, because it's the only one whose clean-data pre-check looked
genuinely promising rather than decisively negative. The labeled result
turned out to be decisively negative anyway, for a reason worth recording
carefully.

---

## 0. One-paragraph summary

A truncated, two-boundary Wald SPRT was built as a genuinely different
alternative to CUSUM (not a relabeling — Page's CUSUM is mathematically a
repeated, zero-reflected SPRT; what's different here is a real
accept-H0 boundary plus truncation calibrated DIRECTLY at a 34-day
horizon, fixing the exact "growing window" problem the ARL-CUSUM attempt
failed to solve). **The architectural goal was fully achieved**: SPRT,
calibrated honestly from real clean data at `horizon_n=34`, passes the
real-data acceptance check (`real_full` stays quiet, both plants) at
every tested `delta` up to 1.9 — something neither CUSUM's own ad hoc
per-window recalibration nor the ARL-CUSUM attempt can honestly claim at
that horizon. A synthetic, clean-data-only power pre-check then found a
modest edge over CUSUM's short-window baseline (+1 to +3 percentage
points on a short/subtle A6 grid). **The labeled tune/holdout validation
— the first point real fraud labels touched this — did not confirm
that edge. SPRT detected ZERO of the 18 real A6_slow_drift instances in
the full 30-replicate pool, at every one of 11 tested `delta` values
(0.1 through 1.9). CUSUM, already documented as weak on A6 (5.6%
recall), still catches 1 of those 18. Not shipped.**

---

## 1. File locations

```
src/layer3_anomaly/sprt.py                     run_sprt(), calibrate_sprt_boundaries()
                                                (KEPT, not wired into production)
src/evaluation/_experiment_sprt_precheck.py    clean-data-only pre-check (kept, re-runnable)
src/evaluation/_experiment_sprt_validate.py    labeled tune/holdout validation (kept, re-runnable)
reports/tables/_experiment_sprt_precheck.json  pre-check output
reports/tables/_experiment_sprt_validate.json  labeled validation output
docs/ARL_CUSUM_HANDOFF.md SS5                  the finding this workstream follows up on
```

Nothing in `dashboard/lib/pipeline.py` or `src/evaluation/harness.py` was
touched — CUSUM remains production exactly as-is, unaffected by this
exploration.

---

## 2. Why this is a genuinely different test, not CUSUM relabeled

Checked directly before building anything: Page's CUSUM
(`S_t = max(0, S_{t-1} + z_t - k)`) is mathematically a repeated,
zero-REFLECTED SPRT (Page 1954, Lorden 1971) — a naive SPRT that just
reflects at a lower boundary of zero would not be a meaningfully
different test from what already exists. What's actually different here:

1. A REAL lower (accept-H0) boundary `B`, not a reflection at zero — the
   statistic can go negative, accumulate evidence FOR "clean," and only
   resets when it explicitly crosses `B` (a genuine "confirmed clean
   segment" decision). CUSUM's floor-at-zero discards this evidence
   every step.
2. Truncation at an EXPLICIT horizon `N=34` (matching `real_full`, the
   largest real deployment length this project checks), with boundaries
   calibrated to bound the CUMULATIVE false-alarm probability across the
   WHOLE horizon directly — not translated from a shorter window via an
   ARL-style approximation (the exact mistake `docs/ARL_CUSUM_HANDOFF.md`
   found and corrected once already).

A real implementation bug was caught and fixed before trusting any
number: the first version's forced end-of-horizon tie-break rule
(`Lambda_N > 0 -> flag`) made the empirical false-positive rate almost
independent of `A` — checked directly (FPR barely moved as `A` swept
1.0-5.0) and traced to the fact that `Lambda`'s drift under H0 is small
relative to its random-walk variance over 34 steps, so the SIGN of
`Lambda_N` is close to a coin flip, swamping `A`'s actual effect. Fixed
by defaulting the forced decision to accept-clean (never flag) instead —
the standard, principled convention for a truncated sequential test, and
confirmed empirically to restore proper FPR monotonicity in `A`.
**If you ever touch `run_sprt`'s truncation branch, re-verify this
monotonicity check before trusting a new calibration.**

---

## 3. Clean-data-only pre-check — a genuine architectural win

Reused the exact-algebra ramp-injection shortcut from the ARL-CUSUM
pre-check (`r_day` shift = `day_delta * intensity`, citing
`apply_a6_slow_drift`'s exact formula — no approximation, no
re-derivation needed).

- **Acceptance check (run FIRST this time, learned from Workstream 3's
  late surprise)**: `real_full` (34 real days, both plants) stays quiet
  at every `delta` from 0.1 up through 1.9. At `delta >= 2.0`, plant_2's
  `real_full` starts getting flagged — a real, principled edge, not an
  arbitrary stopping point.
- **Synthetic detection power** (short/subtle grid: `ramp_days in
  {3,4}`, `final_delta in {0.08,0.10,0.12}`, pooled both plants): grows
  with `delta`, peaking within the acceptance-passing range at
  `delta~1.8-1.9` with mean power ~0.077-0.079, vs. CUSUM's baseline
  0.055 on the same synthetic grid — a modest **+2 to +3pp edge**, well
  short of the pre-declared 15pp bar but not a regression either.

This part of the exercise succeeded exactly as designed: a horizon-bound
calibrated directly at the length that matters produces a test that's
honestly valid there, unlike either prior attempt.

---

## 4. Labeled tune/holdout validation — the real result

Per this project's own established discipline: `A`/`B` stay honestly
calibrated from real clean data only (never labels); `delta` is the
dimension being "tuned" in the sense that matters for a tune/holdout
split (same reasoning as `escalation.py`'s `band_width`/`cusum_frac`).
Swept `delta` on replicates 0-19 (tune), selecting by
`A6_slow_drift` recall (SPRT is evaluated STANDALONE, not combined with
the primary conformal flag — matching exactly how the synthetic pre-check
measured "detection power," and matching how CUSUM's own standalone
contribution is understood in this project: CUSUM alone also has poor
overall recall/precision, which is why production only ever combines it
with the primary test, not runs it alone).

**Result, decisive and unambiguous**: `A6_slow_drift` recall was
**exactly 0.0 at every single one of the 11 tested `delta` values**
(0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.8, 1.2, 1.6, 1.8, 1.9) on the tune
set. Zero real A6 days caught, across the entire tested sensitivity
range — not a marginal shortfall, a complete miss on the intended target
category, confirmed on the FULL pooled 30-replicate set too (0/18 real
`A6_slow_drift` instances detected, at the tune-selected `delta`).
**CUSUM, on the same full pooled set, still catches 1 of those 18**
(5.6% recall — its own already-documented weak, but non-zero, number).

**A real methodology wrinkle, disclosed rather than hidden**: because
`A6_slow_drift` recall was 0.0 for every candidate, the tune-set
selection criterion had zero discriminative power and defaulted
(Python's `max()` tie-break) to the first candidate, `delta=0.1` — the
most conservative one, which in turn detects NOTHING at all across every
anomaly type on the full pooled set (0/540 TP, specificity 1.0). This
degenerate selection is a direct, honest consequence of A6 being too rare
(18 instances total) for this metric to differentiate candidates once
none of them catch any — not a bug to paper over, but worth being
explicit that the "best_delta" chosen by this run's own stated
methodology isn't independently meaningful; the real finding is the flat
zero across the WHOLE range, visible directly in the tune-sweep table
(`reports/tables/_experiment_sprt_validate.json`), not the specific
degenerate pick.

**A genuinely higher `delta` (1.6-1.9) does raise OVERALL recall
noticeably on the tune set (up to ~19.6%)** — but per-type inspection at
the full-pooled level (done for the degenerate `delta=0.1` result; the
tune sweep's own overall numbers at `delta=1.6-1.9` corroborate the
pattern directly) shows this comes from incidentally catching OTHER
anomaly types (a sufficiently sensitive generic deviation detector will
eventually trip on large single-day excursions from A1/A4/A5/A7, which
were never its target), not from A6 — and it costs REAL, measured
specificity: 0.74-0.76 at `delta=1.6-1.9` on the tune set, a real
regression from CUSUM's known-good ~0.92-0.93.

**Statistical honesty note**: with only 18 real A6 instances total, a
true detection rate anywhere near the synthetic pre-check's ~7-8%
estimate would still have roughly a 1-in-4 chance of producing zero
observed detections by pure sample-size luck (`(1-0.075)^18 ~= 0.24`) —
so this result does not PROVE the true rate is exactly zero. It does,
however, provide zero evidence FOR any positive detection capability on
the one dataset this project has, which is a sufficient basis to not
ship this.

---

## 5. Why the synthetic pre-check and the real result disagree — an honest, unresolved question

Not fully diagnosed (out of scope to fully resolve this session, flagged
for whoever revisits this), but worth stating plainly rather than
glossing over: the synthetic proxy showed measurable power on the
"short/subtle" corner specifically (`ramp_days in {3,4}`,
`final_delta in {0.08-0.12}`), while REAL A6 replicates span the FULL
frozen-injector range (`ramp_days in [3,9]`, `final_delta in
[0.05,0.15]`) — if anything, the real mix should be no harder than the
synthetic "hardest corner" on average, since it includes longer ramps
(more time to accumulate evidence) and larger magnitudes too. The
disagreement is therefore NOT simply explained by "real is harder than
synthetic" and deserves a closer look before anyone trusts a synthetic
power pre-check for a persistent-shift detector construction again in
this project — possible candidates for the real explanation (untested):
autocorrelation across real consecutine days that the bootstrap-IID
synthetic construction doesn't capture, or a genuine small-sample
artifact (§4's statistical honesty note) rather than a systematic
divergence at all.

---

## 6. Decision and recommendation

**Not shipped.** Given the labeled result is decisively negative on the
one metric that matters (real A6 recall, flat zero across the entire
tested range) rather than merely short of the pre-declared bar, no
further validation footprint (`mdo_curve`/`leave_one_plant_out`/
`ablations`/dashboard `AppTest`) was run — consistent with this
project's own "stop once decisive" discipline (the same discipline the
original faster-CUSUM attempt used, stopping at its own pre-check gate).

**What's genuinely worth keeping**: the SPRT module itself is real,
working, and architecturally correct — it is the only construction this
session that actually delivers a valid guarantee at the real 34-day
horizon (confirmed, not assumed). If anyone later builds a different
detector construction and needs a horizon-bounded sequential test as a
foundation, `calibrate_sprt_boundaries`'s methodology (calibrate directly
at the real horizon, bootstrap on clean data, never a closed-form
approximation) is reusable groundwork. The specific choice of `delta`
(a persistent Gaussian mean-shift) as the alternative hypothesis for A6
specifically is the piece that didn't pan out, not the SPRT machinery
itself.

**Also worth keeping as a methodological lesson**: a synthetic,
clean-data-only power pre-check is necessary but was NOT sufficient here
— this is the sharpest example yet, across all of this session's three
workstreams, of synthetic proxy results and real labeled performance
diverging. Future work in this space should treat a synthetic pre-check
result as a reason to proceed to labeled validation cheaply, never as a
result to report or act on by itself.
