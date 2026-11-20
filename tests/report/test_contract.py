"""Contract verification tests for CrowdPulse output directories.

Validates that any run output directory (e.g., data/outputs/sample/ or data/outputs/latest/)
strictly follows the schema, column names, JSON structures, and data types
defined in docs/CONTRACT.md.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import pytest

REQUIRED_POSTS_COLUMNS = [
    "post_id", "exchange", "ticker", "source", "created_at_utc", "text",
    "likes", "reposts", "views", "author_followers", "url",
    "sentiment_score", "sentiment_label", "weight", "bar_date"
]

REQUIRED_NEWS_COLUMNS = [
    "ticker", "title", "url", "published_at_utc", "source_domain", "summary",
    "sentiment_score", "bar_date"
]

REQUIRED_PRICES_COLUMNS = [
    "ticker", "date", "open", "high", "low", "close", "volume", "currency"
]

REQUIRED_DAILY_COLUMNS = [
    "ticker", "date", "mentions", "social_sent_mean", "social_sent_weighted",
    "bullish_ratio", "news_count", "news_sent_mean", "close", "return_1d"
]

REQUIRED_PREDICTIONS_LOG_COLUMNS = [
    "run_id", "ticker", "target_date", "predicted_close", "actual_close",
    "error", "error_pct", "direction_hit"
]

EXPECTED_TICKERS = {
    "0700.HK", "9988.HK", "3690.HK", "1810.HK", "1211.HK",
    "005930.KS", "000660.KS", "035420.KS", "035720.KS", "005380.KS"
}

VALID_DIRECTIONS = {"bullish", "bearish", "neutral"}
VALID_SIDES = {"long", "short", "none"}


@pytest.fixture
def sample_dir() -> Path:
    # Default path relative to repo root
    repo_root = Path(__file__).resolve().parent.parent.parent
    p = repo_root / "data" / "outputs" / "sample"
    if not p.exists():
        pytest.fail(f"Sample data folder does not exist at {p}. Run fixtures/make_fixtures.py first.")
    return p


def test_required_files_exist(sample_dir: Path) -> None:
    expected_files = [
        "posts.csv",
        "news.csv",
        "prices.csv",
        "daily.csv",
        "summary.json",
        "metrics.json",
        "predictions_log.csv",
    ]
    for filename in expected_files:
        f = sample_dir / filename
        assert f.exists(), f"Missing required contract file: {filename}"
        assert f.stat().st_size > 0, f"Contract file is empty: {filename}"


def test_posts_csv_schema_and_types(sample_dir: Path) -> None:
    posts_file = sample_dir / "posts.csv"
    with open(posts_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == REQUIRED_POSTS_COLUMNS, f"posts.csv header mismatch: {reader.fieldnames}"
        rows = list(reader)

    assert len(rows) > 0, "posts.csv has no data rows"
    tickers_found = set()

    for idx, row in enumerate(rows):
        tickers_found.add(row["ticker"])
        assert row["exchange"] in ("HKEX", "KRX"), f"Row {idx}: invalid exchange {row['exchange']}"
        assert row["ticker"] in EXPECTED_TICKERS, f"Row {idx}: unexpected ticker {row['ticker']}"
        assert row["source"], f"Row {idx}: empty source"
        assert row["created_at_utc"], f"Row {idx}: empty created_at_utc"
        assert len(row["text"]) > 0, f"Row {idx}: empty text"

        # Numerical fields
        likes = int(row["likes"])
        reposts = int(row["reposts"])
        views = int(row["views"])
        followers = int(row["author_followers"])
        assert likes >= 0 and reposts >= 0 and views >= 0 and followers >= 0, f"Row {idx}: negative count"

        score = float(row["sentiment_score"])
        assert -1.0 <= score <= 1.0, f"Row {idx}: sentiment_score out of range: {score}"

        label = row["sentiment_label"]
        assert label in VALID_DIRECTIONS, f"Row {idx}: invalid sentiment_label: {label}"

        # Direction rule check
        if score > 0.15:
            assert label == "bullish", f"Row {idx}: score {score} should be bullish"
        elif score < -0.15:
            assert label == "bearish", f"Row {idx}: score {score} should be bearish"
        else:
            assert label == "neutral", f"Row {idx}: score {score} should be neutral"

        weight = float(row["weight"])
        assert weight > 0, f"Row {idx}: weight must be positive"
        assert row["bar_date"], f"Row {idx}: empty bar_date"

    assert len(tickers_found) == 10, f"Expected all 10 tickers in posts.csv, found {len(tickers_found)}"


def test_news_csv_schema_and_types(sample_dir: Path) -> None:
    news_file = sample_dir / "news.csv"
    with open(news_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == REQUIRED_NEWS_COLUMNS, f"news.csv header mismatch: {reader.fieldnames}"
        rows = list(reader)

    assert len(rows) > 0, "news.csv has no data rows"
    for idx, row in enumerate(rows):
        assert row["ticker"] in EXPECTED_TICKERS, f"Row {idx}: unexpected ticker {row['ticker']}"
        assert row["title"], f"Row {idx}: missing title"
        assert row["url"].startswith("http"), f"Row {idx}: invalid url {row['url']}"
        assert row["source_domain"], f"Row {idx}: missing source_domain"
        score = float(row["sentiment_score"])
        assert -1.0 <= score <= 1.0, f"Row {idx}: sentiment_score out of range: {score}"
        assert row["bar_date"], f"Row {idx}: missing bar_date"


def test_prices_csv_schema_and_history(sample_dir: Path) -> None:
    prices_file = sample_dir / "prices.csv"
    with open(prices_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == REQUIRED_PRICES_COLUMNS, f"prices.csv header mismatch: {reader.fieldnames}"
        rows = list(reader)

    assert len(rows) > 0, "prices.csv has no data rows"
    ticker_counts: dict[str, int] = {}
    for idx, row in enumerate(rows):
        ticker = row["ticker"]
        ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1
        curr = row["currency"]
        assert curr in ("HKD", "KRW"), f"Row {idx}: invalid currency {curr}"

        op = float(row["open"])
        hi = float(row["high"])
        lo = float(row["low"])
        cl = float(row["close"])
        vol = float(row["volume"])
        assert op > 0 and hi > 0 and lo > 0 and cl > 0 and vol >= 0, f"Row {idx}: non-positive price/volume"
        assert hi >= lo, f"Row {idx}: high {hi} < low {lo}"

    for t in EXPECTED_TICKERS:
        count = ticker_counts.get(t, 0)
        assert count >= 90, f"Ticker {t} has only {count} price rows (< 90 trading days required)"


def test_daily_csv_schema_and_types(sample_dir: Path) -> None:
    daily_file = sample_dir / "daily.csv"
    with open(daily_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == REQUIRED_DAILY_COLUMNS, f"daily.csv header mismatch: {reader.fieldnames}"
        rows = list(reader)

    assert len(rows) > 0, "daily.csv has no rows"
    for idx, row in enumerate(rows):
        assert row["ticker"] in EXPECTED_TICKERS, f"Row {idx}: unexpected ticker {row['ticker']}"
        mentions = int(row["mentions"])
        assert mentions >= 0, f"Row {idx}: negative mentions"
        b_ratio = float(row["bullish_ratio"])
        assert 0.0 <= b_ratio <= 1.0, f"Row {idx}: bullish_ratio out of [0, 1]"
        news_count = int(row["news_count"])
        assert news_count >= 0, f"Row {idx}: negative news_count"
        close_p = float(row["close"])
        assert close_p > 0, f"Row {idx}: non-positive close"


def test_summary_json_schema_and_types(sample_dir: Path) -> None:
    summary_file = sample_dir / "summary.json"
    with open(summary_file, encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list), "summary.json must contain a JSON list"
    assert len(data) == 10, f"summary.json must contain exactly 10 ticker items, got {len(data)}"

    found_tickers = set()
    found_edge_cases = {
        "zero_news": False,
        "null_prediction": False,
        "no_position": False,
    }

    for item in data:
        ticker = item.get("ticker")
        found_tickers.add(ticker)
        assert ticker in EXPECTED_TICKERS, f"Unknown ticker in summary.json: {ticker}"
        assert item.get("exchange") in ("HKEX", "KRX")
        assert isinstance(item.get("name"), str) and len(item["name"]) > 0
        assert item.get("currency") in ("HKD", "KRW")
        assert float(item.get("current_price")) > 0
        assert isinstance(item.get("as_of"), str)
        assert int(item.get("mentions_10d")) >= 0
        assert isinstance(item.get("mentions_change_pct"), (int, float))
        assert -1.0 <= float(item.get("sentiment_score")) <= 1.0
        if item.get("news_sentiment") is not None:
            assert -1.0 <= float(item.get("news_sentiment")) <= 1.0
        assert item.get("direction") in VALID_DIRECTIONS
        assert 0.0 <= float(item.get("confidence")) <= 1.0

        # Support & Resistance
        support = item.get("support")
        resistance = item.get("resistance")
        assert isinstance(support, list) and len(support) == 2, f"{ticker}: support must be list of 2"
        assert isinstance(resistance, list) and len(resistance) == 2, f"{ticker}: resistance must be list of 2"
        for s_lvl in support:
            if s_lvl is not None:
                assert float(s_lvl) > 0, f"{ticker}: support level must be positive"
        for r_lvl in resistance:
            if r_lvl is not None:
                assert float(r_lvl) > 0, f"{ticker}: resistance level must be positive"
        if support[0] is not None:
            assert support[0] <= item["current_price"] * 1.05  # reasonable level

        # Position
        pos = item.get("position")
        assert isinstance(pos, dict), f"{ticker}: position must be a dict"
        assert pos.get("side") in VALID_SIDES
        if pos.get("side") == "none":
            found_edge_cases["no_position"] = True
        else:
            assert pos.get("entry") is not None
            assert pos.get("stop") is not None
            assert pos.get("target") is not None
            assert pos.get("risk_reward") is not None

        # Prediction
        pred = item.get("prediction")
        assert isinstance(pred, dict), f"{ticker}: prediction must be a dict"
        assert "target_date" in pred
        if pred.get("predicted_close") is None:
            found_edge_cases["null_prediction"] = True
            assert pred.get("low") is None and pred.get("high") is None
        else:
            assert float(pred["predicted_close"]) > 0
            if pred.get("low") is not None and pred.get("high") is not None:
                assert float(pred["low"]) <= float(pred["high"])

        # Top posts and top news
        top_posts = item.get("top_posts")
        assert isinstance(top_posts, list)
        for tp in top_posts:
            assert "source" in tp and "text" in tp and "sentiment_score" in tp and "url" in tp

        top_news = item.get("top_news")
        assert isinstance(top_news, list)
        if len(top_news) == 0:
            found_edge_cases["zero_news"] = True
        for tn in top_news:
            assert "title" in tn and "url" in tn and "sentiment_score" in tn

    assert found_tickers == EXPECTED_TICKERS, "summary.json did not cover all 10 tickers"
    assert found_edge_cases["zero_news"], "Missing edge case: ticker with zero news"
    assert found_edge_cases["null_prediction"], "Missing edge case: ticker with null prediction"
    assert found_edge_cases["no_position"], "Missing edge case: ticker with no position"


def test_metrics_json_schema(sample_dir: Path) -> None:
    metrics_file = sample_dir / "metrics.json"
    with open(metrics_file, encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, dict), "metrics.json must be a dict"
    assert "overall" in data, "metrics.json missing 'overall'"
    assert "per_ticker" in data, "metrics.json missing 'per_ticker'"

    metric_keys = ["mae", "rmse", "mape", "directional_accuracy", "n_predictions"]

    for model_key in ["model", "naive", "no_sentiment"]:
        assert model_key in data["overall"], f"metrics.json overall missing {model_key}"
        for mk in metric_keys:
            assert mk in data["overall"][model_key], f"Missing {mk} in overall.{model_key}"

    per_ticker = data["per_ticker"]
    for t in EXPECTED_TICKERS:
        assert t in per_ticker, f"metrics.json per_ticker missing {t}"
        for model_key in ["model", "naive", "no_sentiment"]:
            assert model_key in per_ticker[t], f"Missing {model_key} in per_ticker[{t}]"
            for mk in metric_keys:
                assert mk in per_ticker[t][model_key], f"Missing {mk} in per_ticker[{t}][{model_key}]"


def test_predictions_log_csv_schema(sample_dir: Path) -> None:
    pred_file = sample_dir / "predictions_log.csv"
    with open(pred_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == REQUIRED_PREDICTIONS_LOG_COLUMNS, f"predictions_log.csv header mismatch: {reader.fieldnames}"
        rows = list(reader)

    assert len(rows) > 0, "predictions_log.csv has no rows"
    for idx, row in enumerate(rows):
        assert row["run_id"], f"Row {idx}: missing run_id"
        assert row["ticker"] in EXPECTED_TICKERS, f"Row {idx}: unexpected ticker {row['ticker']}"
        assert row["target_date"], f"Row {idx}: missing target_date"
        pred = float(row["predicted_close"])
        act = float(row["actual_close"])
        err = float(row["error"])
        assert abs(pred - act - err) < 0.1, f"Row {idx}: predicted - actual != error"
        assert int(row["direction_hit"]) in (0, 1), f"Row {idx}: direction_hit must be 0 or 1"
