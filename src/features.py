"""Causal (leak-free) feature engineering.

The SAME class is used for offline training and online serving, so there is no
train/serve skew. Every feature for transaction i is computed from transactions
strictly BEFORE i (state is updated only after features are emitted).

Note: the Kaggle dataset has no card/account ID, so behavioural features are
computed over the global stream (velocity, amount deviation vs. history).
"""
import math
from collections import deque

V_COLS = [f"V{i}" for i in range(1, 29)]
BEHAVIOR_COLS = [
    "log_amount", "hour", "secs_since_prev",
    "tx_count_60s", "tx_count_600s",
    "amount_vs_past_mean", "amount_zscore",
]
FEATURE_COLS = V_COLS + BEHAVIOR_COLS


class StreamingFeaturizer:
    def __init__(self):
        self.d60, self.d600 = deque(), deque()
        self.last_t = None
        self.n, self.mean, self.m2 = 0, 0.0, 0.0  # Welford on log1p(amount)

    def transform_and_update(self, t: float, amount: float) -> dict:
        while self.d60 and t - self.d60[0] > 60:
            self.d60.popleft()
        while self.d600 and t - self.d600[0] > 600:
            self.d600.popleft()

        la = math.log1p(max(amount, 0.0))
        if self.n > 1:
            std = math.sqrt(self.m2 / (self.n - 1))
            z = (la - self.mean) / (std + 1e-6)
        else:
            z = 0.0
        feats = {
            "log_amount": la,
            "hour": (t // 3600) % 24,
            "secs_since_prev": 0.0 if self.last_t is None else max(t - self.last_t, 0.0),
            "tx_count_60s": len(self.d60),
            "tx_count_600s": len(self.d600),
            "amount_vs_past_mean": la - self.mean if self.n else 0.0,
            "amount_zscore": max(min(z, 50.0), -50.0),
        }
        # --- update state AFTER computing features (no leakage) ---
        self.d60.append(t)
        self.d600.append(t)
        self.last_t = t
        self.n += 1
        delta = la - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (la - self.mean)
        return feats
