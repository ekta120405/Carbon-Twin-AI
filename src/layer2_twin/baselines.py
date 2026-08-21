"""
src/layer2_twin/baselines.py

Benchmark comparators from Task 2 §2.5 (items 3 and 4): Ridge and LSTM.
Both trained on the SAME twin_* feature set the PI-GBQR twin uses -- a fair
comparison isolates the MODEL, not the features.

Both baselines predict ac_power DIRECTLY (not the physics-residual target)
-- that's deliberate: Ridge tests whether non-linearity is actually needed
at all (Task 2 §2.5 item 3), and LSTM connects to references [3]/[8] in the
literature review, which train directly on generation, not a physics-informed
residual. Comparing "PI-GBQR on residual" vs "LSTM on raw target" is exactly
the ablation that argues for the physics-informed design (Overview v2 §9.5a).

CRITICAL: the LSTM baseline respects the SAME R1 invariant as the twin --
it sees only twin_* independent weather features in its input sequence,
NEVER lagged claimed generation. An LSTM that autoregresses on its own
target would not be a fair or valid comparator under this project's rules;
it would also not be auditable, since it could simply learn to echo
whatever the plant claims.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def fit_ridge_baseline(X_train: pd.DataFrame, y_train: pd.Series, alpha: float = 1.0):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(X_train, y_train)
    return model


def build_sequences(
    X: pd.DataFrame, y: pd.Series, seq_len: int = 8
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Windows the twin_* feature matrix into overlapping sequences of length
    `seq_len` (default 8 x 15-min = 2 hours of weather context) for the LSTM.
    Only contiguous runs (no gaps) produce a valid window -- a gap (missing
    timestamp) breaks the sequence, matching the QC-driven exclusion of dirty
    rows rather than silently bridging across a discontinuity.

    Returns (X_seq [n, seq_len, n_features], y_seq [n], valid_end_index).
    """
    assert X.index.equals(y.index), "X and y must share the same index"
    values = X.values
    target = y.values
    index = X.index

    expected_step = pd.Timedelta("15min")
    gaps = index.to_series().diff().fillna(expected_step) != expected_step

    X_seqs, y_seqs, end_idx = [], [], []
    for end in range(seq_len - 1, len(X)):
        start = end - seq_len + 1
        window_gap = gaps.iloc[start + 1:end + 1]  # any gap WITHIN the window invalidates it
        if window_gap.any():
            continue
        X_seqs.append(values[start:end + 1])
        y_seqs.append(target[end])
        end_idx.append(index[end])

    return (np.array(X_seqs), np.array(y_seqs), pd.DatetimeIndex(end_idx))


class LSTMBaseline:
    """
    Simple 2-layer LSTM -> linear head, point-estimate output (q50 only --
    this baseline does NOT have quantiles, per Task 2 SS2.2's R2 failure
    mode for LSTM; that's the point being demonstrated, not a limitation
    to hide).

    Normalizes both features and target internally before training, and
    inverts the target transform on predict() -- without this, raw-scale
    kW targets (tens of thousands) combined with raw-scale weather
    features produce a loss surface where the network fails to learn
    anything at all (confirmed empirically: flat MSE, catastrophic
    negative R2 on TRAIN data too, not just test -- a training failure,
    not overfitting). This does not change what's being predicted --
    ac_power directly, per the deliberate design -- only how it's trained.
    """

    def __init__(self, n_features: int, hidden_size: int = 32, num_layers: int = 2):
        import torch
        import torch.nn as nn

        self.torch = torch
        self.x_mean_ = None
        self.x_std_ = None
        self.y_mean_ = None
        self.y_std_ = None

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True)
                self.head = nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :]).squeeze(-1)

        self.net = Net()

    def fit(self, X_seq: np.ndarray, y_seq: np.ndarray, epochs: int = 60,
            lr: float = 1e-3, batch_size: int = 64) -> "LSTMBaseline":
        torch = self.torch

        # Normalize features across (samples, timesteps), per-feature
        self.x_mean_ = X_seq.mean(axis=(0, 1), keepdims=True)
        self.x_std_ = X_seq.std(axis=(0, 1), keepdims=True)
        self.x_std_[self.x_std_ < 1e-8] = 1.0
        X_norm = (X_seq - self.x_mean_) / self.x_std_

        # Normalize target
        self.y_mean_ = y_seq.mean()
        self.y_std_ = y_seq.std() if y_seq.std() > 1e-8 else 1.0
        y_norm = (y_seq - self.y_mean_) / self.y_std_

        X_t = torch.tensor(X_norm, dtype=torch.float32)
        y_t = torch.tensor(y_norm, dtype=torch.float32)

        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()

        n = len(X_t)
        for epoch in range(epochs):
            perm = torch.randperm(n)
            total_loss = 0.0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad()
                pred = self.net(X_t[idx])
                loss = loss_fn(pred, y_t[idx])
                loss.backward()
                opt.step()
                total_loss += loss.item() * len(idx)
            if (epoch + 1) % 20 == 0:
                print(f"  epoch {epoch+1}/{epochs}  mse(normalized)={total_loss/n:.4f}")
        return self

    def predict(self, X_seq: np.ndarray) -> np.ndarray:
        torch = self.torch
        X_norm = (X_seq - self.x_mean_) / self.x_std_
        with torch.no_grad():
            X_t = torch.tensor(X_norm, dtype=torch.float32)
            pred_norm = self.net(X_t).numpy()
        return pred_norm * self.y_std_ + self.y_mean_

