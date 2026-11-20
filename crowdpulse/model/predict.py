"""Live n+1 prediction per ticker and the predictions log.

The final model trains on every row with a known target, then predicts from each
ticker's latest closed bar. The low/high range adds the 10th/90th percentile of
past walk-forward errors; with too few errors it falls back to +/-1.28 x the
ticker's recent daily-return std (an 80% band).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from crowdpulse.features.calendar import TradingCalendar
from crowdpulse.model.train import VARIANTS, fit, training_rows
from crowdpulse.schema import PREDICTIONS_LOG_COLUMNS

logger = logging.getLogger(__name__)
MIN_FINAL_TRAIN_ROWS = 15


def _round_price(x: float, ref: float) -> float:
    return round(float(x), 0 if ref >= 1000 else 2 if ref >= 1 else 4)


def predict_next(features: pd.DataFrame, calendars: dict[str, TradingCalendar], exchange_of: dict[str, str],
                 resid_q: tuple[float, float] | None) -> dict[str, dict]:
    """ticker -> {target_date, predicted_close, low, high, predicted_return, base_close}. Nulls if no model."""
    rows = training_rows(features)
    model = fit(rows, VARIANTS["model"]) if len(rows) >= MIN_FINAL_TRAIN_ROWS else None
    if model is None:
        logger.warning("Only %d training rows; predictions will be null", len(rows))
    out: dict[str, dict] = {}
    for ticker, g in features.sort_values("date").groupby("ticker"):
        last = g.iloc[-1]
        last_date = pd.Timestamp(last["date"]).date()
        cal = calendars[exchange_of[ticker]]
        target_date = cal.next_day(last_date).isoformat()
        base = float(last["close"])
        if model is None or pd.isna(last.get("ret_1d")):
            out[ticker] = {"target_date": target_date, "predicted_close": None, "low": None, "high": None,
                           "predicted_return": None, "base_close": base}
            continue
        pred_ret = float(model.predict_returns(g.iloc[[-1]])[0])
        if resid_q is not None:
            lo_q, hi_q = resid_q
        else:
            sd = float(g["ret_1d"].tail(20).std()) if g["ret_1d"].notna().sum() > 2 else 0.02
            lo_q, hi_q = -1.28 * sd, 1.28 * sd
        out[ticker] = {
            "target_date": target_date,
            "predicted_close": _round_price(base * (1 + pred_ret), base),
            "low": _round_price(base * (1 + pred_ret + min(lo_q, 0.0)), base),
            "high": _round_price(base * (1 + pred_ret + max(hi_q, 0.0)), base),
            "predicted_return": pred_ret,
            "base_close": base,
        }
    return out


def append_log(log_path: Path, run_id: str, predictions: dict[str, dict]) -> pd.DataFrame:
    """Add this run's predictions to the cumulative log (replacing rows of the same run_id)."""
    new = pd.DataFrame([
        {"run_id": run_id, "ticker": t, "target_date": p["target_date"], "predicted_close": p["predicted_close"],
         "actual_close": None, "error": None, "error_pct": None, "direction_hit": None}
        for t, p in sorted(predictions.items()) if p.get("predicted_close") is not None
    ], columns=list(PREDICTIONS_LOG_COLUMNS))
    if log_path.is_file():
        old = pd.read_csv(log_path, dtype={"run_id": str, "ticker": str, "target_date": str})
        old = old[old["run_id"] != run_id]
        log = pd.concat([old, new], ignore_index=True) if not new.empty else old
    else:
        log = new
    log = log[list(PREDICTIONS_LOG_COLUMNS)].copy()
    log["direction_hit"] = pd.to_numeric(log["direction_hit"], errors="coerce").astype("Int64")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(log_path, index=False)
    return log
