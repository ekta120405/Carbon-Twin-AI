# Layer 2 (Physics Anchor + Twin) — Handoff Notes

Status as of this commit: **core complete across all four benchmark parts.
One scoped module (2.5, Online Learning) not yet built — see "Open Items" below.**

This document exists so nobody re-discovers the same bugs, and so Layer 3
knows exactly what it's consuming and why it looks the way it does.

---

## 1. Output contract — what Layer 3 actually reads

```
reports/models/<plant>/twin_q10.txt         LightGBM text-format booster (q10)
reports/models/<plant>/twin_q50.txt         LightGBM text-format booster (q50)
reports/models/<plant>/twin_q90.txt         LightGBM text-format booster (q90)
reports/models/<plant>/twin_meta.json       feature_cols, ac_nameplate_kw, monotone_map

reports/tables/<plant>_twin_metrics.json    calibration_ratio + regime-stratified
                                             metrics for: pigbqr, physics_only, ridge, lstm

reports/benchmarks/<plant>_tft_predictions.parquet       + _metrics.json
reports/benchmarks/<plant>_tsmixer_predictions.parquet   + _metrics.json
```

Models are saved via `Booster.save_model()` (text format), **never pickled** —
version-stable across LightGBM releases, per the Task 3 portability guidance.
Do not switch this to pickle for convenience later; it breaks the "load on
any machine" guarantee.

`PIGBQRTwin.predict(X, physics_ac_kw)` returns a DataFrame with
`expected_q10`, `expected_q50`, `expected_q90` — this is the contract
Layer 3's conformal calibration consumes directly.

---

## 2. Critical gotcha: q50 is a mean estimate, not a true median

**Root cause (empirically confirmed, not assumed):** LightGBM's
`monotone_constraints` hard-rejects **any piecewise-linear-gradient
objective** — confirmed for both `"quantile"` (any alpha) and
`"regression_l1"` (MAE), both throwing the identical
`Cannot use monotone_constraints in <objective> objective` fatal error.
Only `"regression"` (L2/MSE, smooth Hessian) is accepted alongside a
monotone constraint in this LightGBM version.

**Consequence:** `twin_q50.txt` is trained with `objective="regression"`
(L2) + monotone constraints — a monotone-constrained **conditional mean**,
not a median regressor. `twin_q10.txt` / `twin_q90.txt` are genuine
unconstrained `objective="quantile"` models (no monotone constraint —
that combination is exactly what's blocked).

**If you ever touch `twin_model.py`'s `fit()` method:** do not "simplify"
this by giving q50 the same objective as q10/q90 — it will crash training
immediately, and if you route around the crash by dropping the constraint
instead of understanding it, you lose the physical-plausibility guarantee
the whole design is built around. This was verified with an isolated
minimal repro (`test_l1_monotone.py`, not committed — throwaway diagnostic)
before accepting it, not inferred from a code comment. If you want to
re-verify: try `objective="regression_l1"` + `monotone_constraints` on any
toy dataset; it will throw the same error.

**Report this precisely in the paper**, not as "q50 is the median":
> LightGBM's monotone-constraint mechanism rejects any piecewise-linear-
> gradient objective. The central estimate is therefore a monotone-
> constrained conditional-mean model (L2), not a true median regressor;
> q10/q90 remain genuine unconstrained quantile models.

---

## 3. Quantile-crossing safety net — verified clean, but re-verify on retrain

`twin.predict()` sorts `[q10_pred, q50_pred, q90_pred]` row-wise before
returning, to guarantee `q10 <= q50 <= q90`. This means the returned
`expected_q50` **can, in a crossing event, actually be a value produced by
the unconstrained q10 or q90 model**, not the constrained median booster —
the sort has no way to know which model a value "really" came from.

Confirmed via `verify_monotonicity.py`, which checks the raw constrained
booster's output separately from the post-sort pipeline output: **both
were clean, on both plants, across all four monotone features** (twin_ghi,
twin_clearsky_ghi, twin_kt, twin_cell_temp), at the specific operating
point tested (one real training row, feature swept, others held fixed).

**If the twin is ever retrained (new hyperparameters, new data, tuning
pass): re-run `verify_monotonicity.py` before trusting the result again.**
This is an empirical finding on the trained model, not a structural
guarantee — a different hyperparameter configuration or dataset could
behave differently.

---

## 4. Overfitting — resolved for plant_1, honestly unresolved for plant_2

- **Plant_1:** overfitting was real (found via train-vs-test R² comparison)
  and is now largely resolved via tuned hyperparameters
  (`num_leaves` 15→7, `n_estimators` 600→80, `lambda_l2` 5→10). Twin beats
  physics-only in every regime, small train/test gaps (0.005–0.13) except
  partly-cloudy (0.47, not fully explained, doesn't drag it below baseline).
- **Plant_2:** clear regime (train R²=0.789, test R²=0.307, gap 0.48) and
  partly-cloudy (train R²=0.740, test R²=0.140, gap 0.60) remain
  **confirmed, unresolved overfitting** — twin underperforms physics-only
  in both. Investigated via residual variance, training row counts, day
  coverage, and train-vs-test gap — all ruled out except the gap itself,
  which is real. Plausibly connected to Task 4's coordinate-confidence
  asymmetry (plant_2/nasik = city-centroid, not satellite-verified,
  weaker NASA-feature fidelity) — the calibration ratio (0.897 vs
  plant_1's 1.028) is weakly consistent with this but not conclusive.
- **Decision made:** do not chase this further — a generalized
  calib-split fallback selector (twin vs. physics-only, chosen per
  plant/regime) was considered and explicitly rejected, because
  `calib_days` is only 5 days, the same low-power problem already
  diagnosed for `test_clean_days`, plus it would contaminate Layer 3's
  conformal calibration with a selection decision made on the same split.
  **Do not revisit this without addressing that contamination risk first.**

---

## 5. Overcast (Kt < 0.4) — hard fallback, not a fix

Only 9–12 training rows in this regime for either plant — not enough to
trust any tuned model, regardless of hyperparameters. Solution: **the
twin is bypassed entirely below Kt=0.4**, falling back to the
zero-training-data physics anchor. This is a deliberate design choice
(the physics anchor needs no training data and is trustworthy at any
sample size), not a workaround.

**The Kt=0.4 threshold intentionally matches
`config.yaml`'s `layer3.conformal.csi_bins` boundary** — Layer 2's
fallback cutoff and Layer 3's cloud-regime stratification agree on purpose.
If you ever change one, change the other, or the two layers' regime
definitions will silently diverge.

---

## 6. LSTM baseline — real normalization bug, now fixed

The LSTM initially wasn't learning at all (loss flat / uninformative) due
to a feature/target scale mismatch. **If LSTM metrics ever look
implausible again** (e.g., near-zero R², wildly different from Ridge on
the same features), check the training loss trajectory first — a flat or
non-decreasing loss curve is the signature of this exact bug class, not a
data problem.

`build_sequences()` correctly **excludes** any window spanning a real
data gap rather than silently bridging across it (verified). Given
scattered QC exclusions, expect only ~20–25% of theoretically possible
sequences to survive at `seq_len=8` — this is correct behavior, not a bug,
if you see a smaller-than-expected sequence count in the console output.

---

## 7. Deep benchmark (TFT/TSMixer, `.venv-deep`) — four real bugs, now fixed

1. Timezone handling
2. NaN-inducing missing values causing catastrophic (non-finite) loss
3. Two `darts` API renames across versions
4. Train/calib leakage + an unbounded-forecast-horizon scoring error

**Headline result (real finding, not a failed exercise):** both TFT and
TSMixer underperform every simpler model (physics-only, PI-GBQR, Ridge,
LSTM) at this dataset's scale (~630–900 training rows). Plant_1 TFT
overall R²=0.483 vs. PI-GBQR's 0.866, Ridge's 0.790, LSTM's 0.870,
physics-only's 0.835. Consistent with the known pattern that
transformer-scale architectures need substantially more data than this
project has to realize their advantage over simpler models.

**Regime-level results at small n are noise, not signal** — established
pattern throughout this project (see plant_2 overfitting investigation's
overcast aside). The most extreme example: TFT/plant_1 overcast
R²=-61.648 on n=7. Always check the n for any regime-specific number
before treating it as a finding, in this table or any future one.

---

## 8. Module 2.5, Online Learning (rolling refit) — scoped, decided NOT to build

Present in `config.yaml` (`layer2.online: mode=rolling_refit, window_days=20,
refit_every_days=1`) and named in the ablation list (Overview v2 §9.5e), but
deliberately not implemented. Decision, not an oversight — reasoning below,
verified against the actual config numbers before finalizing:

`window_days: 20` equals `layer1.splits.train_days: [1,20]` exactly — a
rolling window is NOT smaller than what the static twin already trains on,
it's the same size, sliding forward. The real constraint is span, not
window size: with only 34 total days and `calib`/`test_clean`/
`test_injected` occupying days 21-34, a 20-day window can only advance
across at most 14 distinct starting positions (days 1-15) before running
out of data. That's too short a horizon for genuine concept drift to
manifest, independent of how well-tuned the refit is.

Combined with the already-documented overfitting fragility of the static
twin at this data scale (Section 4), we judged the marginal evidentiary
value of implementing and debugging a new training loop for this specific
ablation to be low relative to the cost. Paper language:

> Online learning (rolling refit, Overview v2 §5.5) was scoped but not
> implemented. The dataset's 34-day span limits a 20-day rolling window to
> at most 14 days of movement before exhausting available data — too short
> a horizon for genuine concept drift to manifest, independent of window
> size. Combined with the already-documented overfitting sensitivity of the
> static twin at this data scale, we judged the marginal evidentiary value
> of implementing and debugging a new training loop for this ablation to
> be low, and report the static batch twin throughout.

**If revisited later:** the 14-day-movement constraint is a property of
the dataset's total span, not fixable by tuning `window_days` down — a
smaller window would move more but train on less, likely worsening the
overfitting problem in Section 4 rather than helping. A genuinely
different dataset (more total days) would be needed to make this ablation
informative.