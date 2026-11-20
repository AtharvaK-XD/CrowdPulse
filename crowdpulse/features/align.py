"""Give every post and article a bar_date: the next trading bar that hasn't closed yet.

HKEX closes 08:00 UTC, KRX 06:30 UTC (from config.yaml). A post 1 minute before
the close counts toward that day; 1 minute after (or exactly at) the close counts
toward the next trading day. Weekends and holidays roll forward.
"""

from __future__ import annotations

import pandas as pd

from crowdpulse.config import Settings
from crowdpulse.features.calendar import TradingCalendar


def calendars(prices: pd.DataFrame, settings: Settings) -> dict[str, TradingCalendar]:
    return {code: TradingCalendar.from_prices(prices, settings, code) for code in settings.exchanges}


def assign_bar_dates(df: pd.DataFrame, time_col: str, settings: Settings,
                     cals: dict[str, TradingCalendar]) -> pd.DataFrame:
    """Return a copy with bar_date (ISO date string) filled from time_col and the ticker's exchange."""
    out = df.copy()
    if out.empty:
        out["bar_date"] = pd.Series(dtype="object")
        return out
    exchange_of = {t.symbol: t.exchange for t in settings.tickers}
    ts = pd.to_datetime(out[time_col], utc=True, errors="coerce")
    bar_dates = []
    for ticker, t in zip(out["ticker"], ts):
        ex = exchange_of.get(ticker)
        if ex is None or pd.isna(t):
            bar_dates.append(None)
        else:
            bar_dates.append(cals[ex].bar_for(t.to_pydatetime()).isoformat())
    out["bar_date"] = bar_dates
    return out
