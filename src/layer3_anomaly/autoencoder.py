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
