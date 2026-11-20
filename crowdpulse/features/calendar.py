"""Trading calendars: the days that have a price bar, extended forward by weekdays.

Past trading days come from prices.csv (a day is a trading day if it has a bar).
Future days are not known from prices, so they are approximated as the next
weekday; an exchange holiday right after the last bar would be misdated by a day.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from crowdpulse.config import ExchangeConfig, Settings


def session_close_utc(day: date, exchange: ExchangeConfig) -> datetime:
    """UTC timestamp at which `day`'s session closes."""
    return datetime.combine(day, exchange.close_utc, tzinfo=timezone.utc)


def next_weekday(day: date) -> date:
    d = day + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


class TradingCalendar:
    """Sorted trading days for one exchange, with weekday extension past the last bar."""

    def __init__(self, exchange: ExchangeConfig, days: list[date]):
        self.exchange = exchange
        self.days = sorted(set(days))

    @classmethod
    def from_prices(cls, prices: pd.DataFrame, settings: Settings, exchange_code: str) -> "TradingCalendar":
        ex = settings.exchanges[exchange_code]
        symbols = {t.symbol for t in settings.tickers_for(exchange_code)}
        sub = prices[prices["ticker"].isin(symbols)]
        days = [pd.Timestamp(d).date() for d in sub["date"].unique()]
        return cls(ex, days)

    def next_day(self, day: date) -> date:
        """First trading day strictly after `day`."""
        i = bisect_right(self.days, day)
        if i < len(self.days):
            return self.days[i]
        return next_weekday(max(day, self.days[-1]) if self.days else day)

    def bar_for(self, ts_utc: datetime) -> date:
        """The first trading day whose session has not closed at ts_utc.

        A post exactly at the close counts toward the next day (that bar has closed).
        """
        ts = pd.Timestamp(ts_utc).to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        ts = ts.astimezone(timezone.utc)
        # Any day before ts's UTC date closed before ts (close times are within the UTC day).
        for d in self.days[bisect_left(self.days, ts.date()):]:
            if session_close_utc(d, self.exchange) > ts:
                return d
        d = max(self.days[-1], ts.date() - timedelta(days=1)) if self.days else ts.date() - timedelta(days=1)
        while True:
            d = next_weekday(d)
            if session_close_utc(d, self.exchange) > ts:
                return d
