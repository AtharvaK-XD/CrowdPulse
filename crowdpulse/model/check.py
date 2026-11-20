"""Fill actual closes into the predictions log once the target day has closed.

error = predicted_close - actual_close; error_pct = error / actual_close (a fraction);
direction_hit = 1 when predicted and actual moves from the previous close share a sign.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Callable

import numpy as np
import pandas as pd

from crowdpulse.config import Settings
from crowdpulse.features.calendar import session_close_utc

# (symbol, start, end) -> DataFrame with columns date (ISO str) and close
CloseFetcher = Callable[[str, date, date], pd.DataFrame]


def default_fetch_closes(symbol: str, start: date, end: date) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(symbol, start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
                     interval="1d", auto_adjust=False, progress=False, threads=False)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(symbol, axis=1, level=-1) if symbol in df.columns.get_level_values(-1) else df.droplevel(0, axis=1)
    out = df.reset_index()
    out = out.rename(columns={out.columns[0]: "date", "Close": "close"})
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    return out[["date", "close"]].dropna()


def fill_actuals(log: pd.DataFrame, settings: Settings, *, now: datetime | None = None,
                 fetch_closes: CloseFetcher = default_fetch_closes) -> tuple[pd.DataFrame, int]:
    """Return (updated log, number of rows filled)."""
    now = now or datetime.now(timezone.utc)
    log = log.copy()
    for col in ("actual_close", "error", "error_pct", "direction_hit"):
        log[col] = pd.to_numeric(log[col], errors="coerce").astype("float64")
    exchange_of = {t.symbol: settings.exchanges[t.exchange] for t in settings.tickers}
    pending = log[log["actual_close"].isna()]
    filled = 0
    for ticker, rows in pending.groupby("ticker"):
        ex = exchange_of.get(ticker)
        if ex is None:
            continue
        due = [d for d in rows["target_date"].unique()
               if session_close_utc(date.fromisoformat(d), ex) <= now]
        if not due:
            continue
        start = date.fromisoformat(min(due)) - timedelta(days=10)
        closes = fetch_closes(ticker, start, date.fromisoformat(max(due)))
        series = closes.set_index("date")["close"].astype(float).sort_index()
        for idx, row in rows.iterrows():
            d = row["target_date"]
            if d not in due or d not in series.index:
                continue  # not closed yet, or no bar that day (holiday) -> stays pending
            prev = series[series.index < d]
            if prev.empty:
                continue
            actual, base, pred = float(series[d]), float(prev.iloc[-1]), float(row["predicted_close"])
            log.at[idx, "actual_close"] = actual
            log.at[idx, "error"] = round(pred - actual, 6)
            log.at[idx, "error_pct"] = round((pred - actual) / actual, 6)
            log.at[idx, "direction_hit"] = int(np.sign(pred - base) == np.sign(actual - base))
            filled += 1
    return log, filled


def scorecard(log: pd.DataFrame) -> pd.DataFrame:
    done = log.dropna(subset=["actual_close"])
    if done.empty:
        return pd.DataFrame(columns=["ticker", "checked", "mean_abs_error_pct", "direction_hit_rate"])
    g = done.groupby("ticker")
    card = pd.DataFrame({
        "checked": g.size(),
        "mean_abs_error_pct": g["error_pct"].apply(lambda s: float(np.mean(np.abs(s)) * 100)),
        "direction_hit_rate": g["direction_hit"].mean(),
    }).reset_index()
    total = pd.DataFrame([{"ticker": "ALL", "checked": len(done),
                           "mean_abs_error_pct": float(np.mean(np.abs(done["error_pct"])) * 100),
                           "direction_hit_rate": float(done["direction_hit"].mean())}])
    return pd.concat([card, total], ignore_index=True)
