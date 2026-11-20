"""Expanding-window walk-forward validation and metrics.

For each test date d (in order, never shuffled): train on every row whose target
was already realised by d's close (its next bar is on or before d), predict the
rows dated d, then move on. Compared against:
  naive         tomorrow's close = today's close
  no_sentiment  the same Ridge without sentiment features

Metrics per entry: mae, rmse, mape (percent), directional_accuracy, n_predictions.
Per ticker, mae/rmse are in that ticker's currency. Overall pools different
currencies, so overall mae/rmse are in return units (fraction of price). The naive
baseline never predicts a direction, so its directional_accuracy is null.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from crowdpulse.model.train import VARIANTS, fit, training_rows

logger = logging.getLogger(__name__)

MIN_TRAIN_DATES = 3
MIN_TRAIN_ROWS = 15


def _target_dates(features: pd.DataFrame) -> pd.Series:
    """Each row's next bar date per ticker (when its target_ret is realised)."""
    return features.sort_values(["ticker", "date"]).groupby("ticker")["date"].shift(-1)


def walk_forward(features: pd.DataFrame) -> pd.DataFrame:
    """Out-of-sample predictions: date, ticker, close, actual_ret, pred_<variant> (returns)."""
    feats = features.copy()
    feats["target_date"] = _target_dates(feats)
    rows = training_rows(feats)
    dates = sorted(rows["date"].unique())
    out = []
    for i, d in enumerate(dates):
        if i < MIN_TRAIN_DATES:
            continue
        train = rows[rows["target_date"] <= d]
        test = rows[rows["date"] == d]
        if len(train) < MIN_TRAIN_ROWS or test.empty:
            continue
        rec = test[["date", "ticker", "close", "target_ret"]].rename(columns={"target_ret": "actual_ret"}).copy()
        for name, feats_list in VARIANTS.items():
            rec[f"pred_{name}"] = fit(train, feats_list).predict_returns(test)
        rec["pred_naive"] = 0.0
        out.append(rec)
    if not out:
        logger.warning("Walk-forward produced no predictions (need > %d dates with targets)", MIN_TRAIN_DATES)
        return pd.DataFrame(columns=["date", "ticker", "close", "actual_ret", "pred_model", "pred_no_sentiment",
                                     "pred_naive"])
    return pd.concat(out, ignore_index=True)


def _metrics(close: np.ndarray, actual_ret: np.ndarray, pred_ret: np.ndarray, *, price_units: bool,
             directional: bool) -> dict[str, float | int | None]:
    n = len(actual_ret)
    if n == 0:
        return {"mae": None, "rmse": None, "mape": None, "directional_accuracy": None, "n_predictions": 0}
    actual_close = close * (1 + actual_ret)
    pred_close = close * (1 + pred_ret)
    err = pred_close - actual_close if price_units else pred_ret - actual_ret
    mape = float(np.mean(np.abs(pred_close - actual_close) / actual_close) * 100)
    da = None
    if directional:
        da = float(np.mean(np.sign(pred_ret) == np.sign(actual_ret)))
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(math.sqrt(np.mean(err ** 2))),
            "mape": mape, "directional_accuracy": da, "n_predictions": int(n)}


def compute_metrics(wf: pd.DataFrame, tickers: list[str]) -> dict:
    """metrics.json payload: {"overall": {...}, "per_ticker": {...}} for model / naive / no_sentiment."""
    def entry(df: pd.DataFrame, price_units: bool) -> dict:
        c, a = df["close"].to_numpy(float), df["actual_ret"].to_numpy(float)
        return {
            "model": _metrics(c, a, df["pred_model"].to_numpy(float), price_units=price_units, directional=True),
            "naive": _metrics(c, a, df["pred_naive"].to_numpy(float), price_units=price_units, directional=False),
            "no_sentiment": _metrics(c, a, df["pred_no_sentiment"].to_numpy(float), price_units=price_units,
                                     directional=True),
        }

    overall = entry(wf, price_units=False)
    overall["units"] = {"mae": "return (fraction of price)", "rmse": "return (fraction of price)", "mape": "percent"}
    per_ticker = {}
    for t in tickers:
        e = entry(wf[wf["ticker"] == t], price_units=True)
        e["units"] = {"mae": "price", "rmse": "price", "mape": "percent"}
        per_ticker[t] = e
    model_mae, naive_mae, nos_mae = (overall[k]["mae"] for k in ("model", "naive", "no_sentiment"))
    helps = None
    if None not in (model_mae, naive_mae, nos_mae):
        helps = bool(model_mae < naive_mae and model_mae < nos_mae)
    return {"overall": overall, "per_ticker": per_ticker,
            "sentiment_helps": helps,
            "validation": "expanding-window walk-forward, pooled Ridge on next-day return"}


def residual_quantiles(wf: pd.DataFrame, q_low: float = 0.1, q_high: float = 0.9) -> tuple[float, float] | None:
    """Spread of past walk-forward return errors (actual - predicted) for the prediction range."""
    if len(wf) < 5:
        return None
    resid = (wf["actual_ret"] - wf["pred_model"]).to_numpy(float)
    return float(np.quantile(resid, q_low)), float(np.quantile(resid, q_high))
