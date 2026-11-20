"""daily.csv (one row per ticker per trading day in the sentiment window) + model features.

daily.csv columns follow the contract. Definitions:
  mentions              posts with this bar_date (0 if none)
  social_sent_mean      mean post score (null if no posts)
  social_sent_weighted  sum(score*weight)/sum(weight) (null if no posts)
  bullish_ratio         bullish posts / all posts (null if no posts)
  news_count            articles with this bar_date (0 if none)
  news_sent_mean        mean article score (null if none)
  close, return_1d      bar close and close/prev_close - 1 (null for the first bar)

Only bars with a known close appear (the not-yet-closed next bar has no close).

build_features() adds model inputs for every price date, each computed only from
data available at that day's close (no look-ahead):
  ret_1d, mom_5d (close/close[t-5]-1), vol_5d (std of last 5 daily returns),
  vol_20d (std of last 20 daily returns; caps predicted moves, not a model input),
  sent_w (weighted social sentiment; unknown when the bar has < MIN_SENT_POSTS
  posts, because a score from one or two posts is noise), sent_change (vs previous bar),
  mentions_ratio (mentions / mean of previous 3 bars), news_sent,
  target_ret = next bar's return (the label; NaN for the latest bar).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crowdpulse.nlp.sentiment import weighted_mean
from crowdpulse.schema import DAILY_COLUMNS

MIN_SENT_POSTS = 5  # bars with fewer posts get no sentiment feature (filled as neutral)


def _post_aggregates(posts: pd.DataFrame) -> pd.DataFrame:
    cols = ["ticker", "date", "mentions", "social_sent_mean", "social_sent_weighted", "bullish_ratio"]
    if posts.empty:
        return pd.DataFrame(columns=cols)
    p = posts.dropna(subset=["bar_date"])
    rows = []
    for (ticker, day), g in p.groupby(["ticker", "bar_date"]):
        scores = pd.to_numeric(g["sentiment_score"], errors="coerce")
        rows.append({
            "ticker": ticker,
            "date": day,
            "mentions": len(g),
            "social_sent_mean": float(scores.mean()) if scores.notna().any() else None,
            "social_sent_weighted": weighted_mean(scores.tolist(), pd.to_numeric(g["weight"], errors="coerce").tolist()),
            "bullish_ratio": float((g["sentiment_label"] == "bullish").mean()) if len(g) else None,
        })
    return pd.DataFrame(rows, columns=cols)


def _news_aggregates(news: pd.DataFrame) -> pd.DataFrame:
    cols = ["ticker", "date", "news_count", "news_sent_mean"]
    if news.empty:
        return pd.DataFrame(columns=cols)
    n = news.dropna(subset=["bar_date"])
    g = n.groupby(["ticker", "bar_date"])
    out = pd.DataFrame({"news_count": g.size(),
                        "news_sent_mean": g["sentiment_score"].apply(lambda s: pd.to_numeric(s, errors="coerce").mean())})
    return out.reset_index().rename(columns={"bar_date": "date"})[cols]


def build_daily_all(posts: pd.DataFrame, news: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Every price bar with sentiment aggregates joined (counts 0, unknown sentiment NaN)."""
    px = prices[["ticker", "date", "close"]].copy()
    px["date"] = px["date"].astype(str)
    px = px.sort_values(["ticker", "date"])
    px["return_1d"] = px.groupby("ticker")["close"].pct_change()
    df = px.merge(_post_aggregates(posts), on=["ticker", "date"], how="left")
    df = df.merge(_news_aggregates(news), on=["ticker", "date"], how="left")
    df["mentions"] = pd.to_numeric(df["mentions"], errors="coerce").fillna(0).astype("int64")
    df["news_count"] = pd.to_numeric(df["news_count"], errors="coerce").fillna(0).astype("int64")
    for c in ("social_sent_mean", "social_sent_weighted", "bullish_ratio", "news_sent_mean"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df.reset_index(drop=True)


def build_daily(posts: pd.DataFrame, news: pd.DataFrame, prices: pd.DataFrame, window_start: str) -> pd.DataFrame:
    """daily.csv rows: trading days from window_start (ISO date) onward."""
    df = build_daily_all(posts, news, prices)
    df = df[df["date"] >= window_start]
    return df[list(DAILY_COLUMNS)].sort_values(["ticker", "date"]).reset_index(drop=True)


def build_features(daily_all: pd.DataFrame, window_start: str) -> pd.DataFrame:
    """Model features per ticker per bar; only uses data up to that bar's close."""
    frames = []
    for ticker, g in daily_all.sort_values(["ticker", "date"]).groupby("ticker"):
        g = g.copy()
        g["ret_1d"] = g["close"].pct_change()
        g["mom_5d"] = g["close"] / g["close"].shift(5) - 1
        g["vol_5d"] = g["ret_1d"].rolling(5, min_periods=3).std()
        g["vol_20d"] = g["ret_1d"].rolling(20, min_periods=5).std()
        in_window = g["date"] >= window_start
        # Sentiment features exist only inside the scraped window; outside it they are unknown.
        enough = g["mentions"].fillna(0).astype(float) >= MIN_SENT_POSTS
        g["sent_w"] = g["social_sent_weighted"].where(in_window & enough)
        g["sent_change"] = g["sent_w"] - g["sent_w"].shift(1)
        m = g["mentions"].astype(float).where(in_window)
        prev3 = m.shift(1).rolling(3, min_periods=1).mean()
        g["mentions_ratio"] = np.where(prev3 > 0, m / prev3, np.nan)
        g["news_sent"] = g["news_sent_mean"].where(in_window)
        g["target_ret"] = g["close"].shift(-1) / g["close"] - 1
        g["in_window"] = in_window
        frames.append(g)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
