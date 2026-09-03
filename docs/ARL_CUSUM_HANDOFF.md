# ARL-Calibrated CUSUM (Workstream 3) — Handoff Notes

Status: **built, honestly validated, NOT shipped.** Lives on branch
`workstream3-arl-cusum`, never merged to `master`. Unlike Workstream 1,
this one never touched `data/ground_truth/` at all — both of its two
hypotheses were ruled out at the clean-data-only pre-check / acceptance
stage, the cheaper stopping point this project's own methodology allows
for exactly this reason.

---

## 0. One-paragraph summary

Two independent questions were tested. **(1) Does calibrating CUSUM's
`h` against an Average-Run-Length (ARL) target — instead of the current
fixed-9-day-window bootstrap FPR — let a LOWER `k` (more sensitive to
slow drift) escape the wall the earlier "faster CUSUM" attempt hit
(`ESCALATION_IMPROVEMENT_HANDOFF.md` §8 attempt 2)?** No — confirmed via
an honest, clean-data-only power pre-check: detection power on
A6_slow_drift's short/subtle corner gets monotonically WORSE as `k`
decreases, the same wall reappearing under a structurally different
calibration philosophy, not just the original one. **(2) Independent of
`k`: does ARL calibration alone (same `k=0.25`) fix the real,
already-documented architectural gap that today's `h` is only valid for
the one window length it was calibrated for?** Also no, but for a more
interesting reason than "doesn't work": a single ARL-calibrated `h`
passed a synthetic short-window sanity check almost perfectly, but
**failed this project's own real-clean/real-full acceptance test** —
plant_2's genuine 34-day history moved from the correct "Low" band (under
today's ad hoc per-window recalibration) to "Elevated" under the new
single `h`. Root cause, established directly: an ARL0 guarantee bounds
the AVERAGE spacing between false alarms under continuous
resetting-renewal monitoring — it does NOT bound the cumulative
false-alarm probability of one single fresh run that keeps getting
longer, which is a different, harder property that this construction
never actually solved. **Neither variant is shipped**; the current ad hoc
per-window-recalibration approach, despite being architecturally
inelegant, remains the one that's actually validated to work and stays
in production.

---

## 1. File locations

```
src/layer3_anomaly/cusum.py     calibrate_h_for_arl() and
                                 calibrate_h_for_continuous_monitoring() added
                                 (both KEPT, neither wired into production)
src/evaluation/_experiment_arl_cusum_power_precheck.py   NEW, kept per this
                                 project's _experiment_*.py convention -- the
                                 full honest pre-check, re-runnable
reports/tables/_experiment_arl_cusum_power_precheck.json  its output
docs/ARCHITECTURE_CHANGE_ROADMAP.md   the proposal this implements (Workstream 3, SS6)
```

Production code (`dashboard/lib/pipeline.py`, `src/evaluation/harness.py`,
and the two `_experiment_split_validate.py`/`_experiment_cusum_smart.py`
scripts that briefly referenced the new wiring mid-investigation) is
confirmed **byte-identical to `master`** — `git diff master -- dashboard/
lib/pipeline.py src/evaluation/harness.py` is empty. Only `cusum.py`
(additive, two new functions) and the new experiment script are new on
this branch.

---

## 2. A real methodology bug caught before trusting anything: the naive ARL formula is wrong

The roadmap's plan was to derive `target_arl0` from this project's
already-declared `target_fpr=0.05` at `window_len=9` via the standard
small-probability approximation `target_arl0 = window_len/target_fpr =
9/0.05 = 180` days. **Checked directly against this module's own
simulator before trusting it, and found wrong**: the CURRENT production
`h` (which, by construction, achieves exactly 5% crossing probability
starting FRESH at `S_0=0` over a 9-day window) has a TRUE
renewal-process ARL0 of **~87-90 days on real data, both plants** — not
180. The naive formula is off by roughly 2x.

**Why, mechanistically**: CUSUM started fresh at zero needs "ramp-up
time" before it can threaten to cross `h` — the crossing hazard right
after a reset is lower than the process's steady-state (renewal-average)
hazard. A short fixed window measured from a cold start therefore crosses
LESS often than a naive `W/ARL0` extrapolation assumes. This is a real,
known CUSUM-theory subtlety (a "zero-state vs. steady-state ARL"
distinction), not a bug in the simulator — confirmed by measuring it two
independent ways (the window-based fixed-FPR calibration already in
production, and the new renewal-process simulator) and finding they
disagree by exactly the amount this theory predicts.

**Fix applied**: `target_arl0` is never taken from the formula. It's
measured empirically — calibrate the existing window-based `h` first
(already-validated production method), then measure THAT `h`'s true
renewal ARL0 via the new simulator, and use the MEASURED value as the
target for everything else. `calibrate_h_for_continuous_monitoring()`
does exactly this in two steps. **If you ever revisit this area, do not
reintroduce the `window_len/target_fpr` shortcut — it is off by ~2x, not
a rounding difference.**

---

## 3. Part B result: does ARL calibration let a lower k help? No.

Pre-check design (all clean-data-only, `config/anomalies.yaml`-sourced
parameter ranges, never `data/ground_truth/`): reused A6's exact frozen
injector formula (`src/injection/taxonomy.py::apply_a6_slow_drift`,
`day_delta = final_delta * (i+1) / n_days`) via an EXACT (not
approximate) algebraic shortcut — since `r_day = excess_energy_kwh /
(ac_nameplate_kw * daylight_hours)` and injection only multiplies
`ac_power`, a ramp's effect on `r_day` is exactly `day_delta *
(claimed_energy_kwh / (ac_nameplate_kw * daylight_hours))`, so synthetic
ramps could be injected onto bootstrap-sampled real `(r_day, intensity)`
pairs with pure arithmetic, no twin re-scoring needed — thousands of
trials in seconds.

**Sanity/continuity check first** (does the new ARL machinery reproduce
today's behavior at the SAME `k=0.25`, once `target_arl0` is measured
correctly per §2?): yes, cleanly — detection power within ~1 percentage
point of production across every tested cell (`ramp_days∈{3,4}`,
`final_delta∈{0.08,0.10,0.12}`), both plants. This confirms the corrected
methodology is sound before trusting the k-sweep built on top of it.

**k-sweep result** (`k∈{0.05,0.10,0.15,0.20,0.25}`, ARL-recalibrated `h`
at each): mean detection power over the short/subtle grid **decreases
monotonically** as `k` drops — 0.049 at `k=0.25` down to 0.002 at
`k=0.05`. Best candidate is `k=0.25` itself (unchanged), at −0.5pp vs.
baseline (noise-level). **Does not clear the pre-declared 15-percentage-
point bar — nowhere close, and the trend runs the wrong direction
entirely.**

**Why this matters beyond just repeating the earlier finding**: the
original "faster CUSUM" attempt found this same wall under
FIXED-WINDOW-FPR calibration specifically, leaving open whether it was an
artifact of that particular calibration philosophy. This pre-check
confirms the wall reappears under a STRUCTURALLY DIFFERENT calibration
target (ARL) too — a more general, more conclusive negative result than
either attempt alone. **Lowering `k` is now ruled out as a path to
A6 recall under two independent, honest tests, not one.**

---

## 4. Part A result: does the ARL fix alone (unchanged k) work? No — caught by the real-data acceptance check specifically

Since Part B failed, the roadmap's own decision gate says to still check
whether the pure architectural fix (ARL-calibrated `h` at the CURRENT
`k=0.25`, no sensitivity change intended) is worth shipping on its own —
it targets a real, independently-documented limitation regardless of the
k-lowering question.

**The short-window synthetic sanity check (§3) passed almost perfectly**
at `k=0.25` — this made Part A look like a plausible "free" structural
win. **The real-data acceptance check told a different story.** Wiring
`calibrate_h_for_continuous_monitoring`'s single per-plant `h` into
`dashboard.lib.pipeline.load_plant_base`/`get_scored_context` (replacing
the per-window-length `calibrate_h_from_clean_data` recalibration) and
scoring `real_full` (34 real days, both plants, genuinely clean, no
injected anomalies) directly:

| Plant | Source | Old (per-window h) | New (single ARL h) |
|---|---|---|---|
| plant_1 | real_full | Low, 9.8, 0/34 flagged | Low, 16.4, 1/34 flagged |
| plant_2 | real_full | **Low, 20.2, 0/34 flagged** | **Elevated, 28.1, 6/34 flagged (5 pre-existing night-injection + 1 NEW overclaim)** |

plant_2's real 34-day history — genuinely clean, previously and
correctly scoring "Low" — moved to "Elevated" under the new single `h`.
Traced to a specific mechanism, not a vague regression: the new `h=4.0`
is the SAME numeric value as the OLD 9-day-only `h` that caused the
*original* documented bug in the first place (plant_2 scoring
30.7/"Elevated" over 34 days before the ad hoc per-window-recalibration
fix). CUSUM's running statistic `S_t` naturally has more days to
accumulate drift over a 34-day window than a 9-day one, even under the
null (no real anomaly) — so `max(S_t)` came out at 3.71, 93% of the way
to `h=4.0`. That crossed `escalation.py`'s `cusum_frac * cusum_h = 0.5 *
4.0 = 2.0` corroboration gate on one day, upgrading it from `normal` to
`flagged_overclaim`.

**Why the ARL fix didn't prevent this, stated precisely**: an ARL0
guarantee bounds the AVERAGE spacing between false alarms across a
CONTINUOUS, resetting renewal process. It does NOT bound the cumulative
crossing probability of ONE SINGLE fresh run that keeps getting longer
without ever resetting — that's a different quantity (the same
"zero-state vs. steady-state" distinction from §2, reappearing in a
second, more consequential place). A single fixed `h`, calibrated only
to get the renewal-average right, does nothing to stop a single 34-day
fresh run's cumulative risk from growing with window length — which is
mechanically the SAME failure mode as the original bug this workstream
set out to fix, just re-emerging through a different door. **The
"replace per-window recalibration with one ARL-valid h" framing was
itself subtly wrong** — ARL-validity and "bounded false-alarm risk for a
single growing window" are not the same property, and this workstream
conflated them going in.

**This is exactly the class of finding only a real-data acceptance check
catches, not a synthetic proxy** — the short-ramp power pre-check only
ever exercised 3-4 day synthetic windows, never a 34-day one, so it
structurally could not have surfaced this. Consistent with this
project's own repeated lesson (`ESCALATION_IMPROVEMENT_HANDOFF.md` §5,
and this session's own Workstream 1 finding): synthetic/short-window
validation and genuine real-data behavior are different claims.

---

## 5. What's actually worth keeping from this workstream

Two genuinely reusable findings, independent of the "not shipped"
outcome:

1. **The naive `window_len/target_fpr` → `target_arl0` formula is wrong
   by ~2x on real data** (§2) — a real, citable, non-obvious correction
   to a plausible-sounding but incorrect translation. Anyone tempted to
   use this formula for ANY future ARL-based calibration in this project
   should measure the target empirically instead, exactly as
   `calibrate_h_for_continuous_monitoring` now does.
2. **An ARL0 (ARL-valid) `h` does not imply bounded risk for a single
   growing-length window** (§4) — genuinely useful to know before anyone
   next reaches for "ARL calibration" as a fix for
   window-length-dependence specifically. A correct fix for THAT specific
   problem would need to directly bound the fresh-start cumulative
   crossing probability as a function of window length (a harder,
   differently-shaped calibration problem — e.g. calibrating `h` against
   the WORST-CASE window length actually expected in deployment, or a
   genuinely different sequential-test formulation such as an SPRT with
   an explicit horizon bound) — not attempted here, out of scope for this
   session, a real candidate for a future workstream if this area is
   revisited.

---

## 6. Recommendation

**Production CUSUM stays exactly as it is** — the existing ad hoc
per-window-length `calibrate_h_from_clean_data` recalibration
(`dashboard/lib/pipeline.py`, `src/evaluation/harness.py`), despite being
architecturally inelegant (this module's own docstring already says so),
is the one that's actually empirically validated to produce correct
"Low" real-clean acceptance results on both plants at every window length
tested. Nothing here beats that. Both new `cusum.py` functions are kept,
unwired, as real working code for whoever next investigates this area —
per this project's established convention of keeping honest negative
results as re-runnable evidence, not narration.

No labeled tune/holdout validation or full Task-7 benchmark re-run was
needed for this workstream — both hypotheses were conclusively ruled out
at the clean-data pre-check / real-data acceptance-check stage, cheaper
stopping points this project's own methodology explicitly allows for
(mirroring the original "faster CUSUM" attempt, which also stopped at its
pre-check gate without ever touching a label).
