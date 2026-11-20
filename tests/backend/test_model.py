from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

import crowdpulse.model.evaluate as ev
from crowdpulse.features.align import assign_bar_dates, calendars
from crowdpulse.features.daily import build_daily_all, build_features
from crowdpulse.model.evaluate import compute_metrics, residual_quantiles, walk_forward
from crowdpulse.model.predict import append_log, predict_next
from crowdpulse.model.train import PRICE_FEATURES, SENTIMENT_FEATURES

from .synth import NOW, make_news, make_posts, make_prices

WINDOW_START = (NOW - timedelta(days=10)).date().isoformat()
FEATURE_COLS = PRICE_FEATURES + SENTIMENT_FEATURES


def _features(settings, prices, posts, news):
    cals = calendars(prices, settings)
    posts = assign_bar_dates(posts, "created_at_utc", settings, cals)
    news = assign_bar_dates(news, "published_at_utc", settings, cals)
    return build_features(build_daily_all(posts, news, prices), WINDOW_START), cals


@pytest.fixture
def data(settings):
    return make_prices(settings), make_posts(settings), make_news(settings)


def test_no_feature_uses_future_data(settings, data):
    """Changing everything after day t must not change any feature on or before t."""
    prices, posts, news = data
    base, _ = _features(settings, prices, posts, news)
    dates = sorted(base.loc[base["in_window"], "date"].unique())
    cut = dates[len(dates) // 2]

    p2 = prices.copy()
    future_px = p2["date"] > cut
    p2.loc[future_px, ["open", "high", "low", "close"]] *= 3.0
    cut_ts = pd.Timestamp(cut + "T00:00:00Z") + timedelta(days=1)
    po2 = posts.copy()
    po2.loc[po2["created_at_utc"] >= cut_ts, "sentiment_score"] = -1.0
    po2.loc[po2["created_at_utc"] >= cut_ts, "weight"] = 50.0
    n2 = news.copy()
    n2.loc[n2["published_at_utc"] >= cut_ts, "sentiment_score"] = 1.0
    changed, _ = _features(settings, p2, po2, n2)

    a = base[base["date"] <= cut].set_index(["ticker", "date"])[FEATURE_COLS].sort_index()
    b = changed[changed["date"] <= cut].set_index(["ticker", "date"])[FEATURE_COLS].sort_index()
    pd.testing.assert_frame_equal(a, b)
    # Sanity: the perturbation did change later features.
    assert not changed[changed["date"] > cut]["ret_1d"].equals(base[base["date"] > cut]["ret_1d"])


def test_target_is_next_day_return(settings, data):
    prices, posts, news = data
    feats, _ = _features(settings, prices, posts, news)
    g = feats[feats["ticker"] == "0700.HK"].sort_values("date")
    assert g["target_ret"].iloc[0] == pytest.approx(g["close"].iloc[1] / g["close"].iloc[0] - 1)
    assert pd.isna(g["target_ret"].iloc[-1])


def test_walk_forward_never_trains_on_unrealised_targets(settings, data, monkeypatch):
    prices, posts, news = data
    feats, _ = _features(settings, prices, posts, news)
    seen = []
    real_fit = ev.fit

    def spy(rows, features):
        seen.append(rows[["date", "target_date"]].copy())
        return real_fit(rows, features)

    monkeypatch.setattr(ev, "fit", spy)
    wf = walk_forward(feats)
    assert not wf.empty
    test_dates = sorted(wf["date"].unique())
    # Two variants per test date; each train set's targets were realised by the test date.
    for train, d in zip(seen[::2], test_dates):
        assert (train["target_date"] <= d).all()
        assert (train["date"] < d).all()
    # Expanding window: train sets only grow.
    sizes = [len(s) for s in seen[::2]]
    assert sizes == sorted(sizes)


def test_metrics_shape_and_baselines(settings, data):
    prices, posts, news = data
    feats, _ = _features(settings, prices, posts, news)
    wf = walk_forward(feats)
    tickers = [t.symbol for t in settings.tickers]
    m = compute_metrics(wf, tickers)
    for name in ("model", "naive", "no_sentiment"):
        e = m["overall"][name]
        assert set(e) == {"mae", "rmse", "mape", "directional_accuracy", "n_predictions"}
        assert e["n_predictions"] == len(wf)
    assert m["overall"]["naive"]["directional_accuracy"] is None
    assert set(m["per_ticker"]) == set(tickers)
    # Naive predicts zero return: its overall MAE is the mean absolute actual return.
    assert m["overall"]["naive"]["mae"] == pytest.approx(float(np.mean(np.abs(wf["actual_ret"]))))


def test_predict_next_and_log(settings, data, tmp_path):
    prices, posts, news = data
    feats, cals = _features(settings, prices, posts, news)
    wf = walk_forward(feats)
    exchange_of = {t.symbol: t.exchange for t in settings.tickers}
    preds = predict_next(feats, cals, exchange_of, residual_quantiles(wf))
    assert set(preds) == {t.symbol for t in settings.tickers}
    p = preds["0700.HK"]
    assert p["target_date"] == "2026-11-23"  # Friday's bar -> Monday
    assert p["low"] <= p["predicted_close"] <= p["high"]

    log_path = tmp_path / "predictions_log.csv"
    append_log(log_path, "run_a", preds)
    append_log(log_path, "run_a", preds)  # same run again replaces, not duplicates
    log = append_log(log_path, "run_b", preds)
    assert len(log) == 20
    assert log["actual_close"].isna().all()


def test_predicted_returns_are_capped():
    from crowdpulse.model.train import FALLBACK_DAILY_SD, MAX_MOVE_SD, cap_returns
    rows = pd.DataFrame({"vol_20d": [0.01, 0.01, np.nan]})
    out = cap_returns(np.array([0.10, -0.005, -0.5]), rows)
    assert out[0] == pytest.approx(MAX_MOVE_SD * 0.01)
    assert out[1] == pytest.approx(-0.005)                       # within the cap: unchanged
    assert out[2] == pytest.approx(-MAX_MOVE_SD * FALLBACK_DAILY_SD)


def test_sentiment_feature_needs_enough_posts():
    from crowdpulse.features.daily import MIN_SENT_POSTS
    daily = pd.DataFrame({"ticker": "T.HK", "date": ["2026-11-18", "2026-11-19", "2026-11-20"],
                          "close": [10.0, 10.5, 10.2], "mentions": [MIN_SENT_POSTS, 1, 0],
                          "social_sent_weighted": [0.4, -0.9, np.nan], "news_sent_mean": np.nan})
    f = build_features(daily, "2026-11-18")
    assert f["sent_w"].iloc[0] == pytest.approx(0.4)
    assert f["sent_w"].iloc[1:].isna().all()                     # one post is not a signal
    assert "vol_20d" in f


def test_predict_null_when_too_little_data(settings, data):
    prices, posts, news = data
    feats, cals = _features(settings, prices, posts, news)
    feats = feats.assign(in_window=False)
    preds = predict_next(feats, cals, {t.symbol: t.exchange for t in settings.tickers}, None)
    assert all(p["predicted_close"] is None for p in preds.values())
