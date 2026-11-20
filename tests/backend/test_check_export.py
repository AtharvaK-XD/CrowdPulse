from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from crowdpulse.export import build_summary, make_run_id, validate_outputs, write_outputs
from crowdpulse.features.align import assign_bar_dates, calendars
from crowdpulse.features.daily import build_daily, build_daily_all, build_features
from crowdpulse.model.check import fill_actuals, scorecard
from crowdpulse.model.evaluate import compute_metrics, residual_quantiles, walk_forward
from crowdpulse.model.predict import append_log, predict_next

from .synth import NOW, make_news, make_posts, make_prices


def test_fill_actuals(settings):
    log = pd.DataFrame([
        {"run_id": "r", "ticker": "0700.HK", "target_date": "2026-11-20", "predicted_close": 105.0,
         "actual_close": None, "error": None, "error_pct": None, "direction_hit": None},
        {"run_id": "r", "ticker": "0700.HK", "target_date": "2026-11-23", "predicted_close": 99.0,
         "actual_close": None, "error": None, "error_pct": None, "direction_hit": None},
    ])

    def closes(symbol, start, end):
        return pd.DataFrame({"date": ["2026-11-19", "2026-11-20"], "close": [100.0, 104.0]})

    out, n = fill_actuals(log, settings, now=datetime(2026, 11, 20, 9, tzinfo=timezone.utc), fetch_closes=closes)
    assert n == 1
    r = out.iloc[0]
    assert r["actual_close"] == 104.0
    assert r["error"] == pytest.approx(1.0)               # predicted - actual
    assert r["error_pct"] == pytest.approx(1.0 / 104.0, abs=1e-6)
    assert r["direction_hit"] == 1                        # both up from 100
    assert pd.isna(out.iloc[1]["actual_close"])           # target not closed yet
    card = scorecard(out)
    assert card.iloc[-1]["ticker"] == "ALL" and card.iloc[-1]["checked"] == 1


def test_fill_actuals_waits_for_close(settings):
    log = pd.DataFrame([{"run_id": "r", "ticker": "005930.KS", "target_date": "2026-11-20", "predicted_close": 1.0,
                         "actual_close": None, "error": None, "error_pct": None, "direction_hit": None}])
    called = []

    def closes(*a):
        called.append(a)
        return pd.DataFrame(columns=["date", "close"])

    _, n = fill_actuals(log, settings, now=datetime(2026, 11, 20, 6, 0, tzinfo=timezone.utc), fetch_closes=closes)
    assert n == 0 and called == []  # KRX closes 06:30 UTC


def _full_run(settings, out_root):
    prices, posts, news = make_prices(settings), make_posts(settings), make_news(settings)
    cals = calendars(prices, settings)
    posts = assign_bar_dates(posts, "created_at_utc", settings, cals)
    news = assign_bar_dates(news, "published_at_utc", settings, cals)
    window = (NOW - timedelta(days=10)).date().isoformat()
    daily = build_daily(posts, news, prices, window)
    feats = build_features(build_daily_all(posts, news, prices), window)
    wf = walk_forward(feats)
    tickers = [t.symbol for t in settings.tickers]
    metrics = compute_metrics(wf, tickers)
    preds = predict_next(feats, cals, {t.symbol: t.exchange for t in settings.tickers}, residual_quantiles(wf))
    run_id = make_run_id(NOW)
    log = append_log(out_root / "predictions_log.csv", run_id, preds)
    summary = build_summary(settings, posts=posts, news=news, prices=prices, daily=daily, predictions=preds,
                            tickers=tickers)
    run_dir = write_outputs(out_root, run_id, posts=posts, news=news, prices=prices, daily=daily,
                            summary=summary, metrics=metrics, predictions_log=log)
    return run_dir, tickers


def test_export_writes_valid_contract_files(settings, tmp_path):
    run_dir, tickers = _full_run(settings, tmp_path)
    assert run_dir.name == "run_20261120_090000"
    assert validate_outputs(run_dir, tickers) == []
    latest = tmp_path / "latest"
    assert validate_outputs(latest, tickers) == []
    summary = json.loads((latest / "summary.json").read_text(encoding="utf-8"))
    assert len(summary) == 10
    item = summary[0]
    for key in ("exchange", "ticker", "name", "currency", "current_price", "as_of", "mentions_10d",
                "mentions_change_pct", "sentiment_score", "news_sentiment", "direction", "confidence",
                "support", "resistance", "position", "prediction", "top_posts", "top_news"):
        assert key in item
    no_news = [s for s in summary if s["ticker"] == settings.tickers[-1].symbol][0]
    assert no_news["top_news"] == [] and no_news["news_sentiment"] is None
    posts = pd.read_csv(latest / "posts.csv")
    assert posts["created_at_utc"].str.endswith("Z").all()
    assert posts["bar_date"].notna().all()


def test_validator_catches_problems(settings, tmp_path):
    run_dir, tickers = _full_run(settings, tmp_path)
    daily = pd.read_csv(run_dir / "daily.csv")
    daily.drop(columns=["bullish_ratio"]).to_csv(run_dir / "daily.csv", index=False)
    posts = pd.read_csv(run_dir / "posts.csv")
    posts.loc[0, "sentiment_score"] = 3.0
    posts.to_csv(run_dir / "posts.csv", index=False)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    summary[0]["position"]["side"] = "maybe"
    del summary[1]["prediction"]
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    errors = validate_outputs(run_dir, tickers)
    text = "\n".join(errors)
    assert "daily.csv: header" in text
    assert "posts.sentiment_score" in text
    assert ".position: invalid" in text
    assert ".prediction: invalid" in text


def test_clean_json_writes_whole_numbers_as_integers():
    from crowdpulse.export import _clean_json
    out = _clean_json({"price": 183100.0, "px": 27.64, "zero": 0.0, "bad": float("nan"), "lv": [344.0, None]})
    assert out == {"price": 183100, "px": 27.64, "zero": 0.0, "bad": None, "lv": [344, None]}
    assert isinstance(out["price"], int)
