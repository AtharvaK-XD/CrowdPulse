from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crowdpulse.analytics.levels import compute_levels, merge_levels, pivot_points, swing_points
from crowdpulse.analytics.positions import AVOID_NOTE, confidence, suggest_position
from crowdpulse.analytics.signal import sentiment_shift
from crowdpulse.features.daily import build_daily, build_daily_all
from crowdpulse.schema import DAILY_COLUMNS


def test_daily_aggregates():
    prices = pd.DataFrame({"ticker": "T.KS", "date": ["2026-11-18", "2026-11-19", "2026-11-20"],
                           "close": [100.0, 110.0, 99.0]})
    posts = pd.DataFrame({"ticker": "T.KS", "bar_date": ["2026-11-19", "2026-11-19", "2026-11-20"],
                          "sentiment_score": [0.5, -0.5, 0.2], "sentiment_label": ["bullish", "bearish", "bullish"],
                          "weight": [3.0, 1.0, 1.0]})
    news = pd.DataFrame({"ticker": "T.KS", "bar_date": ["2026-11-20"], "sentiment_score": [0.4]})
    d = build_daily(posts, news, prices, "2026-11-18")
    assert list(d.columns) == list(DAILY_COLUMNS)
    r18, r19, r20 = (d[d["date"] == x].iloc[0] for x in ("2026-11-18", "2026-11-19", "2026-11-20"))
    assert r18["mentions"] == 0 and pd.isna(r18["social_sent_mean"]) and pd.isna(r18["return_1d"])
    assert r19["mentions"] == 2
    assert r19["social_sent_mean"] == pytest.approx(0.0)
    assert r19["social_sent_weighted"] == pytest.approx(0.25)
    assert r19["bullish_ratio"] == pytest.approx(0.5)
    assert r19["return_1d"] == pytest.approx(0.1)
    assert r20["news_count"] == 1 and r20["news_sent_mean"] == pytest.approx(0.4)
    assert r19["news_count"] == 0


def test_daily_window_filter():
    prices = pd.DataFrame({"ticker": "T.KS", "date": ["2026-11-10", "2026-11-20"], "close": [1.0, 2.0]})
    empty = pd.DataFrame(columns=["ticker", "bar_date", "sentiment_score", "sentiment_label", "weight"])
    assert build_daily(empty, empty, prices, "2026-11-15")["date"].tolist() == ["2026-11-20"]
    assert len(build_daily_all(empty, empty, prices)) == 2


# ------------------------------------------------------------------------- levels


def test_swing_points():
    high = np.array([1, 2, 3, 10, 3, 2, 1, 2, 3, 4], dtype=float)
    low = np.array([5, 4, 3, 2, 3, 4, 5, 4, 3, 4], dtype=float)
    sh, sl = swing_points(high, low)
    assert sh == [10.0]
    assert sl == [2.0]


def test_merge_levels_within_tolerance():
    levels = merge_levels([100.0, 101.0, 100.5, 120.0])
    assert levels[0].touches == 3 and levels[0].price == pytest.approx(100.5)
    assert levels[1].price == 120.0


def test_pivots():
    p = pivot_points(110, 90, 100)
    assert p["P"] == pytest.approx(100)
    assert p["S1"] < p["P"] < p["R1"] and p["S2"] < p["S1"] and p["R2"] > p["R1"]


def test_compute_levels_orders_nearest_first():
    rng = np.random.default_rng(3)
    close = 100 + np.cumsum(rng.normal(0, 2, 120))
    bars = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=120).strftime("%Y-%m-%d"),
                         "high": close + 1, "low": close - 1, "close": close})
    lv = compute_levels(bars)
    cur = close[-1]
    assert len(lv.support) == 2 and len(lv.resistance) == 2
    s = [x for x in lv.support if x is not None]
    r = [x for x in lv.resistance if x is not None]
    assert all(x < cur for x in s) and s == sorted(s, reverse=True)
    assert all(x > cur for x in r) and r == sorted(r)


def test_compute_levels_pivot_fallback():
    bars = pd.DataFrame({"date": ["2026-11-19", "2026-11-20"], "high": [11.0, 12.0], "low": [9.0, 10.0],
                         "close": [10.0, 11.0]})
    lv = compute_levels(bars)
    assert lv.method in ("pivot", "swing+pivot")
    assert lv.support[0] is not None and lv.support[0] < 11 and lv.resistance[0] > 11


# ----------------------------------------------------------------------- positions


def test_long_position():
    pos = suggest_position("bullish", 0.7, 100.0, 95.0, [96.0, 90.0], [110.0, 120.0])
    assert pos["side"] == "long"
    assert pos["entry"] == 100.0 and pos["stop"] == pytest.approx(95.52) and pos["target"] == 110.0
    assert pos["risk_reward"] == pytest.approx(round(10 / (100 - 95.52), 2))


def test_bearish_is_avoid_hedge_short():
    pos = suggest_position("bearish", 0.7, 100.0, 105.0, [90.0, 85.0], [104.0, 110.0])
    assert pos["side"] == "short" and pos["note"] == AVOID_NOTE
    assert pos["target"] == 90.0 and pos["stop"] > 104.0


def test_no_position_cases():
    assert suggest_position("bullish", 0.5, 100, 95, [96, 90], [110, 120])["side"] == "none"   # low confidence
    assert suggest_position("bullish", 0.7, 100, 105, [96, 90], [110, 120])["side"] == "none"  # below SMA20
    assert suggest_position("neutral", 0.9, 100, 95, [96, 90], [110, 120])["side"] == "none"
    assert suggest_position("bullish", 0.9, 100, 95, [None, None], [110, 120])["side"] == "none"
    none = suggest_position("bearish", 0.7, 100, 95, [96, 90], [110, 120])
    assert none == {"side": "none", "entry": None, "stop": None, "target": None, "risk_reward": None}


def test_no_position_when_model_disagrees():
    args = (0.7, 100.0, 105.0, [90.0, 85.0], [104.0, 110.0])
    assert suggest_position("bearish", *args, predicted_return=0.06)["side"] == "none"
    assert suggest_position("bearish", *args, predicted_return=-0.01)["side"] == "short"
    assert suggest_position("bearish", *args, predicted_return=None)["side"] == "short"


def test_confidence_blend():
    hi = confidence(1, 1.0, 500, 0.3, 0.01)
    assert hi == pytest.approx(1.0)
    lo = confidence(1, 1.0, 500, -0.3, -0.01)
    assert lo == pytest.approx(0.6)
    assert confidence(0, 0.9, 0, None, None) == pytest.approx(0.2)   # no direction: strength ignored
    assert 0 <= confidence(-1, 0.4, 10, None, 0.02) <= 1


# -------------------------------------------------------------------- sentiment shift


def _shift_posts(base_scores, recent_scores):
    rows = [{"bar_date": "2026-11-18", "sentiment_score": s, "weight": 1.0} for s in base_scores]
    rows += [{"bar_date": d, "sentiment_score": s, "weight": 1.0}
             for s, d in zip(recent_scores, ["2026-11-20", "2026-11-23"] * len(recent_scores))]
    return pd.DataFrame(rows)


def test_uniformly_negative_forum_is_neutral_not_bearish():
    rng = np.random.default_rng(0)
    base, recent = rng.normal(-0.27, 0.5, 400).clip(-1, 1), rng.normal(-0.27, 0.5, 200).clip(-1, 1)
    sh = sentiment_shift(_shift_posts(base, recent), "2026-11-20")
    assert sh.recent < -0.15 and sh.baseline < -0.15        # raw level would say "bearish"
    assert sh.direction == "neutral" and sh.strength == 0.0


def test_shift_up_is_bullish_and_down_is_bearish():
    rng = np.random.default_rng(1)
    base = rng.normal(-0.3, 0.4, 400).clip(-1, 1)
    up = sentiment_shift(_shift_posts(base, rng.normal(0.0, 0.4, 200).clip(-1, 1)), "2026-11-20")
    assert up.direction == "bullish" and up.n_recent == 200 and 0 < up.strength <= 1
    down = sentiment_shift(_shift_posts(base, rng.normal(-0.6, 0.4, 200).clip(-1, 1)), "2026-11-20")
    assert down.direction == "bearish" and down.sign == -1


def test_shift_needs_enough_posts():
    sh = sentiment_shift(_shift_posts([0.0] * 10, [0.9] * 3), "2026-11-20")
    assert sh.n_recent == 3 and sh.z is None and sh.direction == "neutral"
    assert sentiment_shift(pd.DataFrame(), "2026-11-20").direction == "neutral"
