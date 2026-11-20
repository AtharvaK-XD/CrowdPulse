from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd

from crowdpulse.features.align import assign_bar_dates, calendars
from crowdpulse.features.calendar import TradingCalendar, next_weekday, session_close_utc
from crowdpulse.ingest.prices import fetch_prices, normalise_bars

from .synth import business_days


def _records(days, close=100.0):
    return [{"date": d.isoformat(), "open": close, "high": close + 1, "low": close - 1, "close": close,
             "adj_close": close, "volume": 1000} for d in days]


def test_normalise_drops_unclosed_bar(settings):
    t = settings.ticker("0700.HK")  # HKEX closes 08:00 UTC
    days = business_days(date(2026, 11, 20), 5)
    during = datetime(2026, 11, 20, 7, 0, tzinfo=timezone.utc)
    after = datetime(2026, 11, 20, 8, 0, tzinfo=timezone.utc)
    assert normalise_bars(_records(days), t, settings, now=during)["date"].max() == "2026-11-19"
    assert normalise_bars(_records(days), t, settings, now=after)["date"].max() == "2026-11-20"


def test_normalise_keeps_last_n_and_currency(settings):
    t = settings.ticker("005930.KS")
    days = business_days(date(2026, 11, 20), 200)
    recs = _records(days) + [{"date": "2026-11-21", "open": None, "high": None, "low": None, "close": None,
                              "volume": None}]
    bars = normalise_bars(recs, t, settings, now=datetime(2026, 11, 22, tzinfo=timezone.utc))
    assert len(bars) == settings.price_history_days
    assert (bars["currency"] == "KRW").all()
    assert list(bars.columns) == ["ticker", "date", "open", "high", "low", "close", "volume", "currency"]


def test_fetch_prices_with_fake_downloader(settings):
    def download(symbol, start, end):
        days = business_days(end, 180)
        return pd.DataFrame({"Open": 1.0, "High": 2.0, "Low": 0.5, "Close": 1.5, "Adj Close": 1.5, "Volume": 10},
                            index=pd.DatetimeIndex(days, name="Date"))

    now = datetime(2026, 11, 20, 9, 0, tzinfo=timezone.utc)
    prices = fetch_prices(settings, tickers=["0700.HK", "005930.KS"], now=now, download=download)
    assert prices.groupby("ticker").size().to_dict() == {"0700.HK": 120, "005930.KS": 120}


def test_fetch_prices_one_failure_does_not_stop(settings):
    def download(symbol, start, end):
        if symbol == "0700.HK":
            raise RuntimeError("boom")
        days = business_days(end, 130)
        return pd.DataFrame({"Open": 1.0, "High": 2.0, "Low": 0.5, "Close": 1.5, "Volume": 10},
                            index=pd.DatetimeIndex(days, name="Date"))

    prices = fetch_prices(settings, tickers=["0700.HK", "005930.KS"],
                          now=datetime(2026, 11, 20, 9, tzinfo=timezone.utc), download=download)
    assert set(prices["ticker"]) == {"005930.KS"}


# ---------------------------------------------------------------- calendar / alignment


def _cal(settings, code, days):
    return TradingCalendar(settings.exchanges[code], days)


def test_post_one_minute_before_close_lands_same_day(settings):
    cal = _cal(settings, "HKEX", [date(2026, 11, 19), date(2026, 11, 20), date(2026, 11, 23)])
    close = session_close_utc(date(2026, 11, 20), settings.exchanges["HKEX"])
    assert cal.bar_for(close - timedelta(minutes=1)) == date(2026, 11, 20)


def test_post_one_minute_after_close_lands_next_day(settings):
    cal = _cal(settings, "HKEX", [date(2026, 11, 19), date(2026, 11, 20), date(2026, 11, 23)])
    close = session_close_utc(date(2026, 11, 20), settings.exchanges["HKEX"])
    assert cal.bar_for(close + timedelta(minutes=1)) == date(2026, 11, 23)  # Fri -> Mon
    assert cal.bar_for(close) == date(2026, 11, 23)  # exactly at the close: bar has closed


def test_krx_close_0630(settings):
    cal = _cal(settings, "KRX", [date(2026, 11, 19), date(2026, 11, 20)])
    assert cal.bar_for(datetime(2026, 11, 20, 6, 29, tzinfo=timezone.utc)) == date(2026, 11, 20)
    assert cal.bar_for(datetime(2026, 11, 20, 6, 31, tzinfo=timezone.utc)) == date(2026, 11, 23)


def test_weekend_and_holiday_roll_forward(settings):
    # 2026-11-18 (Wed) is a holiday: no bar.
    cal = _cal(settings, "KRX", [date(2026, 11, 16), date(2026, 11, 17), date(2026, 11, 19), date(2026, 11, 20)])
    assert cal.bar_for(datetime(2026, 11, 17, 12, tzinfo=timezone.utc)) == date(2026, 11, 19)
    assert cal.bar_for(datetime(2026, 11, 15, 3, tzinfo=timezone.utc)) == date(2026, 11, 16)  # Sunday


def test_next_day_beyond_calendar(settings):
    cal = _cal(settings, "HKEX", [date(2026, 11, 20)])
    assert cal.next_day(date(2026, 11, 20)) == date(2026, 11, 23)
    assert next_weekday(date(2026, 11, 21)) == date(2026, 11, 23)


def test_assign_bar_dates(settings):
    prices = pd.DataFrame({"ticker": ["0700.HK", "0700.HK", "005930.KS", "005930.KS"],
                           "date": ["2026-11-19", "2026-11-20", "2026-11-19", "2026-11-20"]})
    cals = calendars(prices, settings)
    posts = pd.DataFrame({"ticker": ["0700.HK", "005930.KS", "005930.KS"],
                          "created_at_utc": pd.to_datetime(["2026-11-20T07:59:00Z", "2026-11-20T07:00:00Z",
                                                            "2026-11-19T06:00:00Z"], utc=True)})
    out = assign_bar_dates(posts, "created_at_utc", settings, cals)
    assert out["bar_date"].tolist() == ["2026-11-20", "2026-11-23", "2026-11-19"]
