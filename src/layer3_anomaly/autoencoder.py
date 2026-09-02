"""
src/layer3_anomaly/autoencoder.py

Layer 3.4: autoencoder-based structural anomaly scorer.

Complementary to 3.2's magnitude-based conformal test: this catches
STRUCTURAL anomalies (A4 day-swap, A8 flatline) where individual values
can look locally plausible but the JOINT relationship between weather,
the twin's expectation, and the claim is wrong.

DESIGN DECISION: per-timestep, not per-day-sequence. Two reasons:

1. Sample size. Per-day sequences would give ~20 independent training
   examples (one per train-split day) -- an order of magnitude worse
   than Layer 2's LSTM baseline, which needed ~600-900 TIMESTEP-level
   rows and still required a real normalization-bug fix before it
   learned anything (see docs/LAYER2_HANDOFF.md SS6). A day-sequence
   autoencoder is architecturally similar in complexity to that LSTM,
   trained on ~1/30th the data.

2. Verified against the actual injector code (src/injection/taxonomy.py),
   not assumed: A4's source day is always daily_power.idxmax() -- the
   single highest-mean-power day in the ENTIRE dataset, not a similar-
   weather day chosen for subtlety. apply_a4_day_swap's own docstring
   states the intended mismatch is "correct (cloudy) weather, implausible
   (sunny) claim" -- explicitly a per-timestep-detectable joint anomaly,
   not one requiring sequence context. A8_flatline freezes claimed power
   for a run while weather keeps moving -- also catchable per-timestep,
   especially with the diff features below.

INPUT FEATURES (per timestep):
    twin.feature_cols (20 cols)     -- same weather features the twin uses
    expected_q10/q50/q90 (3 cols)   -- the twin's own prediction
    ac_power (1 col)                -- claimed power
    ac_power_diff1 (1 col)          -- claimed(t) - claimed(t-1), NEW,
                                        targets A8 (frozen claim -> diff=0
                                        while weather keeps changing)
    residual_diff1 (1 col)          -- residual(t) - residual(t-1), NEW

NOTE: ac_power (claimed generation) is intentionally included as an
input. This is NOT a violation of Layer 2's R1 invariant (the TWIN must
never see claimed generation) -- that rule protects the twin's
independence as an estimator uncontaminated by the audited quantity.
This autoencoder's entire purpose is different: it explicitly models the
JOINT relationship between weather, the twin's expectation, and the
claim, so seeing claimed power here is correct and intentional, not a
leak. Do not "fix" this by removing ac_power.

FEATURE SCALING: fit ONLY on train-split data (StandardScaler), applied
everywhere else. Without this, MSE loss is dominated by the largest-scale
features (ac_power ~ tens of thousands, twin_ghi ~ hundreds/thousands)
and small-scale features (hour_sin/cos in [-1,1], residual ~ [-0.05,0.3])
become invisible to training -- the same failure class that broke the
LSTM baseline before its normalization fix. Watch the training loss
trajectory before trusting reconstruction error scores; a flat/
non-decreasing loss is that bug's signature, not a data problem to
route around silently.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

AUTOENCODER_META_COLS = ["expected_q10", "expected_q50", "expected_q90", "ac_power"]
AUTOENCODER_DIFF_COLS = ["ac_power_diff1", "residual_diff1"]


def build_autoencoder_features(
    ts_table: pd.DataFrame, twin_feature_cols: list[str]
) -> pd.DataFrame:
    """
    Builds the full per-timestep input feature matrix.

    ts_table must be the FULL time-ordered output of
    build_residual_table() (src/layer3_anomaly/residual.py) -- i.e. NOT
    pre-filtered to daylight-only -- so the diff features reflect the
    true prior 15-minute timestep rather than a spurious jump across a
    day boundary or a gap.

    Returns a DataFrame with one row dropped (the very first timestamp
    in the series, which has no prior row to diff against) -- this is
    the only NaN-inducing row; everything else is well-defined.
    """
    cols = twin_feature_cols + AUTOENCODER_META_COLS
    out = ts_table[cols].copy()
    out["ac_power_diff1"] = ts_table["ac_power"].diff()
    out["residual_diff1"] = ts_table["residual"].diff()
    return out.dropna()


def fit_scaler(train_features: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler on train-split features ONLY."""
    scaler = StandardScaler()
    scaler.fit(train_features.values)
    return scaler


def build_model(input_dim: int, hidden: list[int]):
    """
    Symmetric encoder/decoder autoencoder, built from config.yaml's
    layer3.autoencoder.hidden (e.g. [8, 4, 8]).

    The bottleneck is the SMALLEST value in `hidden` -- encoder goes
    input_dim -> ... -> bottleneck, decoder mirrors it back out to
    input_dim. ReLU between all layers except the final output layer
    (linear, since inputs are standardized and can be negative).
    """
    import torch.nn as nn

    bottleneck_idx = hidden.index(min(hidden))
    encoder_dims = [input_dim] + hidden[: bottleneck_idx + 1]
    decoder_dims = hidden[bottleneck_idx:] + [input_dim]

    encoder_layers = []
    for i in range(len(encoder_dims) - 1):
        encoder_layers.append(nn.Linear(encoder_dims[i], encoder_dims[i + 1]))
        encoder_layers.append(nn.ReLU())
    encoder = nn.Sequential(*encoder_layers[:-1])  # drop trailing ReLU on bottleneck

    decoder_layers = []
    for i in range(len(decoder_dims) - 1):
        decoder_layers.append(nn.Linear(decoder_dims[i], decoder_dims[i + 1]))
        if i < len(decoder_dims) - 2:
            decoder_layers.append(nn.ReLU())
    decoder = nn.Sequential(*decoder_layers)

    class Autoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder

        def forward(self, x):
            return self.decoder(self.encoder(x))

    return Autoencoder()


@dataclass
class TrainResult:
    model: object
    scaler: StandardScaler
    loss_history: list = field(default_factory=list)
    input_cols: list = field(default_factory=list)


def train_autoencoder(
    train_features: pd.DataFrame,
    hidden: list[int],
    epochs: int,
    learning_rate: float,
    seed: int = 42,
    early_stop_features: pd.DataFrame = None,
    patience: int = 20,
) -> TrainResult:
    """
    Trains on train-split features (the bulk of clean data -- correct and
    necessary for fitting model weights).

    early_stop_features: OPTIONAL, should be calib-split features (NOT
    train, NOT test_clean -- keep test_clean fully held out for the final
    baseline comparison). If given, tracks reconstruction loss on this
    held-out set each epoch and stops once it hasn't improved for
    `patience` epochs, restoring the best-seen weights. This replaces a
    fixed epoch count -- config.yaml's epochs=200 was verified (this
    session) to be well short of convergence for this feature scheme
    (loss still dropping sharply at 200: 1.10->0.41, vs 0.12 at 1000) --
    rather than just raising the fixed count arbitrarily (risking the
    same overfitting pattern already diagnosed in the twin, Layer 2
    Section 4), early stopping on genuinely held-out data stops at
    convergence, whatever that takes.

    If early_stop_features is None, falls back to the fixed `epochs`
    count (original behavior, kept for callers/tests that don't have a
    calib split handy).

    Returns the loss history alongside the model -- ALWAYS inspect this
    before trusting reconstruction error scores. A flat or
    non-decreasing TRAIN loss is the LSTM-bug signature (see module
    docstring); a validation loss that starts INCREASING while train
    loss keeps falling is classic overfitting, which early stopping
    exists specifically to catch.
    """
    import torch

    torch.manual_seed(seed)

    scaler = fit_scaler(train_features)
    X = torch.tensor(scaler.transform(train_features.values), dtype=torch.float32)

    model = build_model(input_dim=X.shape[1], hidden=hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    X_val = None
    if early_stop_features is not None:
        X_val = torch.tensor(
            scaler.transform(early_stop_features[train_features.columns].values), dtype=torch.float32
        )

    loss_history = []
    val_loss_history = []
    best_val_loss = float("inf")
    best_state = None
    epochs_since_improvement = 0
    max_epochs = epochs if X_val is None else 100000  # effectively unbounded, patience controls stop

    model.train()
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        recon = model(X)
        loss = loss_fn(recon, X)
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))

        if X_val is not None:
            model.eval()
            with torch.no_grad():
                val_recon = model(X_val)
                val_loss = float(loss_fn(val_recon, X_val).item())
            model.train()
            val_loss_history.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainResult(
        model=model, scaler=scaler, loss_history=loss_history,
        input_cols=list(train_features.columns),
    )


def compute_reconstruction_error(
    features: pd.DataFrame, result: TrainResult
) -> pd.Series:
    """
    Per-timestep reconstruction error (mean squared error across the
    standardized feature vector), using the SAME scaler fit on train
    data -- never re-fit on the data being scored.
    """
    import torch

    features = features[result.input_cols]  # enforce column order match
    X = torch.tensor(result.scaler.transform(features.values), dtype=torch.float32)

    result.model.eval()
    with torch.no_grad():
        recon = result.model(X)
        error = ((recon - X) ** 2).mean(dim=1).numpy()

    return pd.Series(error, index=features.index, name="reconstruction_error")


def _longest_over_threshold_run(error: pd.Series, threshold: float) -> int:
    """Longest run of consecutive (in the given, time-ordered index)
    values exceeding threshold. 0 if none exceed it."""
    over = error > threshold
    if not over.any():
        return 0
    run_id = (~over).cumsum()
    run_sizes = over.groupby(run_id).transform("sum") * over
    return int(run_sizes.max())


def train_and_calibrate(
    ts_table: pd.DataFrame,
    twin_feature_cols: list[str],
    train_mask: pd.Series,
    calib_dates,
    test_clean_dates,
    hidden: list[int],
    epochs: int,
    learning_rate: float,
    threshold_percentile: float = 0.95,
) -> tuple[TrainResult, float, int]:
    """
    THE canonical training+calibration path for this module -- trains on
    `ts_table[train_mask]`, then calibrates BOTH quantities this project's
    escalation logic needs from clean data alone (calib+test_clean),
    never from injected/labeled data:
      - `threshold`: 95th percentile of pooled clean reconstruction error
        (same convention as aggregate.py's calibrate_divergence_cap).
      - `run_length`: the longest run of consecutive over-threshold
        timesteps ever observed on that SAME clean pool, plus one. A day
        is only escalated by a run at least this long -- one shorter than
        anything real clean data itself produces (src/evaluation/
        _experiment_ae_runlength.py's investigation: plant_1's own clean
        data produces a run as long as 28, so a naively eyeballed cutoff
        that ignored this would not actually sit above the clean noise
        floor).
    Single source of truth for both src/evaluation/ablations.py's ablation
    (f) and the real production trainer (src/layer3_anomaly/
    train_autoencoder_model.py) -- do not reimplement this a third time.
    """
    feats = build_autoencoder_features(ts_table, twin_feature_cols)
    split_dates = pd.Series(ts_table.index.date, index=ts_table.index).loc[feats.index]
    is_daylight = ts_table["is_daylight"].astype(bool).loc[feats.index]

    train_feats = feats[train_mask.reindex(feats.index, fill_value=False)]
    clean_dates = calib_dates.union(test_clean_dates)
    clean_feats = feats[split_dates.isin(clean_dates)]
    calib_feats = feats[split_dates.isin(calib_dates)]

    result = train_autoencoder(
        train_feats, hidden=hidden, epochs=epochs, learning_rate=learning_rate,
        early_stop_features=calib_feats if len(calib_feats) else None,
    )

    # Threshold: pooled clean data, day+night (matches src/evaluation/
    # ablations.py's original train_plant_autoencoder -- establishes what
    # "normal reconstruction error" looks like overall).
    clean_error = compute_reconstruction_error(clean_feats, result)
    threshold = float(clean_error.quantile(threshold_percentile))

    # Run-length: DAYLIGHT ONLY -- a run spanning near-zero-power night
    # rows doesn't reflect the same signal a daylight run does, and this
    # must match score_replicate_autoencoder()'s day-level flagging, which
    # is also daylight-only. Restricting to clean days AND daylight rows,
    # in that order, mirrors src/evaluation/_experiment_ae_runlength.py's
    # investigation exactly (plant_1's own clean daylight data alone
    # produces a run as long as 28 -- night rows were never part of that
    # finding and must not silently change it here).
    clean_daylight_error = compute_reconstruction_error(feats[split_dates.isin(clean_dates) & is_daylight], result)
    run_length = _longest_over_threshold_run(clean_daylight_error, threshold) + 1

    return result, threshold, run_length


def compute_daily_ae_flag(
    ts_table: pd.DataFrame, twin_feature_cols: list[str], result: TrainResult,
    threshold: float, run_length: int,
) -> pd.Series:
    """
    Day-level autoencoder flag using a RUN of `run_length` consecutive
    over-threshold DAYLIGHT timesteps -- not the earlier "any single
    timestep" rule (still used, deliberately unchanged, by src/evaluation/
    ablations.py's ablation (f), which is reporting on that original rule
    specifically). This run-length version is what src/evaluation/
    _experiment_ae_runlength.py validated as a real improvement (see
    docs/TASK7_HANDOFF.md and docs/PROJECT_RESULTS_SUMMARY.md) and is the
    one used by the production escalation path
    (src/layer3_anomaly/escalation.py).
    """
    feats = build_autoencoder_features(ts_table, twin_feature_cols)
    error = compute_reconstruction_error(feats, result)

    daylight_idx = ts_table.index[ts_table["is_daylight"].astype(bool)]
    error_daylight = error[error.index.isin(daylight_idx)].sort_index()
    over = error_daylight > threshold

    run_id = (~over).cumsum()
    run_sizes = over.groupby(run_id).transform("sum") * over
    qualifies = run_sizes >= run_length

    dates = pd.Series(error_daylight.index.date, index=error_daylight.index)
    return qualifies.groupby(dates).any()


def save_autoencoder(
    result: TrainResult, hidden: list[int], threshold: float, run_length: int, out_dir: str,
) -> None:
    """
    Persists a trained autoencoder for reuse across runs -- until this
    function existed, the autoencoder was retrained from scratch every
    single time it was needed (confirmed via grep before this was added;
    see docs/TASK7_HANDOFF.md). Follows this project's established
    text-format-over-pickle convention (LAYER2_HANDOFF.md SS1: LightGBM
    boosters are saved via .save_model(), never pickled, for version
    stability across machines) -- the scaler's fitted mean/scale are
    written out as plain JSON floats rather than a pickled StandardScaler
    object, for the same reason.

    `threshold` and `run_length` are NOT properties of the model itself --
    they're the two calibrated quantities computed alongside it (95th
    percentile of pooled calib+test_clean reconstruction error, and the
    max over-threshold run length observed on that same clean pool + 1,
    see src/evaluation/ablations.py / _experiment_ae_runlength.py) -- both
    persisted here so a caller never has to re-derive them.
    """
    import json
    import torch

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(result.model.state_dict(), out / "autoencoder.pt")
    with open(out / "autoencoder_meta.json", "w") as f:
        json.dump(
            {
                "input_cols": result.input_cols,
                "hidden": hidden,
                "threshold": threshold,
                "run_length": run_length,
                "scaler_mean": result.scaler.mean_.tolist(),
                "scaler_scale": result.scaler.scale_.tolist(),
            },
            f,
            indent=2,
        )


def load_autoencoder(model_dir: str) -> tuple[TrainResult, float, int]:
    """Loads what save_autoencoder() wrote. Returns (TrainResult usable
    directly with compute_reconstruction_error(), threshold, run_length)."""
    import json

    import numpy as np
    import torch

    d = Path(model_dir)
    with open(d / "autoencoder_meta.json") as f:
        meta = json.load(f)

    scaler = StandardScaler()
    scaler.mean_ = np.array(meta["scaler_mean"])
    scaler.scale_ = np.array(meta["scaler_scale"])
    scaler.var_ = scaler.scale_**2
    scaler.n_features_in_ = len(meta["input_cols"])

    model = build_model(input_dim=len(meta["input_cols"]), hidden=meta["hidden"])
    model.load_state_dict(torch.load(d / "autoencoder.pt", weights_only=True))
    model.eval()

    result = TrainResult(model=model, scaler=scaler, loss_history=[], input_cols=meta["input_cols"])
    return result, meta["threshold"], meta["run_length"]
