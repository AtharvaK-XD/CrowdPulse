"""Deterministic synthetic prices/posts/news for offline tests."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from crowdpulse.ingest.social_common import finalize_posts, make_row

NOW = datetime(2026, 11, 20, 9, 0, tzinfo=timezone.utc)  # after both closes


def business_days(end: date, n: int) -> list[date]:
    days, d = [], end
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    return sorted(days)


def make_prices(settings, n: int = 120, end: date = NOW.date(), seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i, t in enumerate(settings.tickers):
        ex = settings.exchanges[t.exchange]
        base = 100.0 * (i + 1) if ex.currency == "HKD" else 50_000.0 * (i + 1)
        rets = rng.normal(0, 0.015, n)
        close = base * np.cumprod(1 + rets)
        for d, c in zip(business_days(end, n), close):
            hi, lo = c * (1 + abs(rng.normal(0, 0.01))), c * (1 - abs(rng.normal(0, 0.01)))
            rows.append({"ticker": t.symbol, "date": d.isoformat(), "open": c, "high": hi, "low": lo,
                         "close": c, "volume": int(rng.integers(1e5, 1e6)), "currency": ex.currency})
    return pd.DataFrame(rows)


def make_posts(settings, per_day: int = 6, days: int = 10, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for t in settings.tickers:
        for d in range(days):
            for k in range(per_day):
                ts = NOW - timedelta(days=d, hours=float(rng.uniform(0, 23)))
                score = float(np.clip(rng.normal(0.1, 0.4), -1, 1))
                r = make_row(source="naver" if t.exchange == "KRX" else "futu", native_id=f"{t.code}-{d}-{k}",
                             exchange=t.exchange, ticker=t.symbol, created_at_utc=ts,
                             text=f"post {t.symbol} {d} {k} {rng.integers(1e9)}", url="https://example.com/p",
                             likes=int(rng.integers(0, 20)), views=int(rng.integers(0, 500)))
                r["sentiment_score"] = score
                rows.append(r)
    df = finalize_posts(rows, now=NOW, lookback_days=days)
    from crowdpulse.nlp.sentiment import label_for, post_weight
    df["sentiment_label"] = df["sentiment_score"].map(label_for)
    df["weight"] = [post_weight(*r) for r in df[["likes", "reposts", "views", "author_followers"]]
                    .itertuples(index=False, name=None)]
    return df


def make_news(settings, per_ticker: int = 4, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for t in settings.tickers[:-1]:  # last ticker gets no news (edge case)
        for k in range(per_ticker):
            rows.append({"ticker": t.symbol, "title": f"{t.name} news {k}", "url": f"https://news.example.com/{t.code}/{k}",
                         "published_at_utc": pd.Timestamp(NOW - timedelta(days=int(rng.integers(0, 9)), hours=3)),
                         "source_domain": "news.example.com", "summary": "s",
                         "sentiment_score": float(rng.uniform(-0.5, 0.5)), "bar_date": None})
    return pd.DataFrame(rows)
