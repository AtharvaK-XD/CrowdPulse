"""Model definition: pooled Ridge regression on next-day return.

All tickers share one model (ticker as a one-hot category) because 10 days of posts
is only ~7-8 trading days per stock. Target = next bar's return; the predicted
close is today's close * (1 + predicted return).

Predicted returns are capped at +/- MAX_MOVE_SD x the stock's 20-day return std
(vol_20d, known at the close), so a noisy input cannot produce an implausible
one-day move. The cap applies in walk-forward evaluation and live prediction alike.

Two variants are trained identically on the same rows:
  "model"         price features + sentiment features
  "no_sentiment"  price features only
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PRICE_FEATURES = ["ret_1d", "mom_5d", "vol_5d"]
SENTIMENT_FEATURES = ["sent_w", "sent_change", "mentions_ratio", "news_sent"]
# Unknown sentiment means "no signal": neutral score, mentions at their usual level.
FILL_VALUES = {"sent_w": 0.0, "sent_change": 0.0, "mentions_ratio": 1.0, "news_sent": 0.0,
               "ret_1d": 0.0, "mom_5d": 0.0, "vol_5d": 0.0}
RIDGE_ALPHA = 1.0
MAX_MOVE_SD = 3.0
FALLBACK_DAILY_SD = 0.02  # when vol_20d is unknown
VARIANTS = {"model": PRICE_FEATURES + SENTIMENT_FEATURES, "no_sentiment": PRICE_FEATURES}


@dataclass
class FittedModel:
    pipeline: Pipeline
    features: list[str]

    def predict_returns(self, rows: pd.DataFrame) -> np.ndarray:
        return cap_returns(self.pipeline.predict(prepare(rows, self.features)), rows)


def cap_returns(pred: np.ndarray, rows: pd.DataFrame) -> np.ndarray:
    """Clip each predicted return to +/- MAX_MOVE_SD x that row's vol_20d."""
    sd = pd.to_numeric(rows["vol_20d"], errors="coerce") if "vol_20d" in rows else pd.Series(np.nan, rows.index)
    limit = MAX_MOVE_SD * sd.fillna(FALLBACK_DAILY_SD).to_numpy(float)
    return np.clip(np.asarray(pred, dtype=float), -limit, limit)


def prepare(rows: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    x = rows[features + ["ticker"]].copy()
    for f in features:
        x[f] = pd.to_numeric(x[f], errors="coerce").fillna(FILL_VALUES.get(f, 0.0))
    return x


def fit(rows: pd.DataFrame, features: list[str], alpha: float = RIDGE_ALPHA) -> FittedModel:
    pre = ColumnTransformer([
        ("num", StandardScaler(), features),
        ("ticker", OneHotEncoder(handle_unknown="ignore"), ["ticker"]),
    ])
    pipe = Pipeline([("pre", pre), ("ridge", Ridge(alpha=alpha))])
    pipe.fit(prepare(rows, features), rows["target_ret"].astype(float))
    return FittedModel(pipe, features)


def training_rows(features: pd.DataFrame) -> pd.DataFrame:
    """Rows inside the sentiment window whose next-day return is known."""
    rows = features[features["in_window"] & features["target_ret"].notna() & features["close"].notna()]
    return rows.dropna(subset=["ret_1d"])
