"""Build summary.json and write every contract file to data/outputs/<run_id>/ and latest/.

validate_outputs() checks each file against docs/CONTRACT.md: exact CSV headers,
value types and ranges, and every summary.json / metrics.json key. The pipeline
runs it at the end of every run; if it passes, the report can read the files.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from crowdpulse.analytics.levels import compute_levels
from crowdpulse.analytics.positions import confidence, suggest_position
from crowdpulse.analytics.signal import sentiment_shift
from crowdpulse.config import Settings
from crowdpulse.nlp.sentiment import weighted_mean
from crowdpulse.schema import (
    DAILY_COLUMNS, NEWS_COLUMNS, POSTS_COLUMNS, PREDICTIONS_LOG_COLUMNS, PRICES_COLUMNS,
)

logger = logging.getLogger(__name__)

TOP_N = 5
METRIC_KEYS = ("mae", "rmse", "mape", "directional_accuracy", "n_predictions")
METRIC_MODELS = ("model", "naive", "no_sentiment")


def make_run_id(now: datetime) -> str:
    return "run_" + now.astimezone(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _num(x: Any, digits: int | None = None) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return round(v, digits) if digits is not None else v


# ----------------------------------------------------------------------- summary


def _mentions_change_pct(daily_t: pd.DataFrame) -> float | None:
    """Mentions in the latest half of the window's trading days vs the earlier half, in %."""
    m = daily_t.sort_values("date")["mentions"].astype(float).to_numpy()
    if len(m) < 2:
        return None
    half = len(m) // 2
    prev, recent = m[:half].sum(), m[half:].sum()
    # Halves can differ by one day; compare per-day averages.
    prev_avg, recent_avg = prev / half, recent / (len(m) - half)
    if prev_avg == 0:
        return None
    return round((recent_avg / prev_avg - 1) * 100, 1)


def build_summary(settings: Settings, *, posts: pd.DataFrame, news: pd.DataFrame, prices: pd.DataFrame,
                  daily: pd.DataFrame, predictions: dict[str, dict], tickers: list[str]) -> list[dict]:
    items = []
    for t in settings.tickers:
        if t.symbol not in tickers:
            continue
        ex = settings.exchanges[t.exchange]
        px = prices[prices["ticker"] == t.symbol].sort_values("date")
        if px.empty:
            logger.warning("%s: no prices; skipping summary item", t.symbol)
            continue
        current = float(px["close"].iloc[-1])
        sma20 = float(px["close"].tail(20).mean()) if len(px) >= 20 else None
        tp = posts[posts["ticker"] == t.symbol]
        tn = news[news["ticker"] == t.symbol]
        sent = weighted_mean(tp["sentiment_score"].astype(float).tolist(), tp["weight"].astype(float).tolist()) \
            if not tp.empty else None
        news_sent = _num(pd.to_numeric(tn["sentiment_score"], errors="coerce").mean()) if not tn.empty else None
        pred = predictions.get(t.symbol, {})
        # Direction = recent sentiment vs the stock's own baseline, not the raw level
        # (forum tone biases the level; see analytics/signal.py).
        shift = sentiment_shift(tp, str(px["date"].iloc[-1]))
        direction = shift.direction
        conf = confidence(shift.sign, shift.strength, len(tp), news_sent, pred.get("predicted_return"))
        lv = compute_levels(px, current)
        position = suggest_position(direction, conf, current, sma20, lv.support, lv.resistance,
                                    predicted_return=pred.get("predicted_return"))

        top_posts = (tp.assign(_rank=tp["weight"].astype(float) * (0.5 + tp["sentiment_score"].abs().astype(float)))
                     .sort_values("_rank", ascending=False).head(TOP_N))
        top_news = tn.sort_values("published_at_utc", ascending=False).head(TOP_N)
        items.append({
            "exchange": t.exchange,
            "ticker": t.symbol,
            "name": t.name,
            "currency": ex.currency,
            "current_price": _num(current, 4),
            "as_of": str(px["date"].iloc[-1]),
            "mentions_10d": int(len(tp)),
            "mentions_change_pct": _mentions_change_pct(daily[daily["ticker"] == t.symbol]),
            "sentiment_score": _num(sent, 4),
            "news_sentiment": _num(news_sent, 4),
            "direction": direction,
            "confidence": _num(conf, 3),
            "support": lv.support,
            "resistance": lv.resistance,
            "position": position,
            "prediction": {k: pred.get(k) for k in ("target_date", "predicted_close", "low", "high")},
            "top_posts": [{"source": r.source, "text": str(r.text)[:280], "sentiment_score": _num(r.sentiment_score, 3),
                           "url": r.url} for r in top_posts.itertuples(index=False)],
            "top_news": [{"title": r.title, "url": r.url, "sentiment_score": _num(r.sentiment_score, 3)}
                         for r in top_news.itertuples(index=False)],
        })
    return items


# ------------------------------------------------------------------------- writing


def _iso_utc(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    return ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ").where(ts.notna(), None)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return _num(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    return str(o)


def _clean_json(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: _clean_json(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean_json(v) for v in o]
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


def write_outputs(out_root: Path, run_id: str, *, posts: pd.DataFrame, news: pd.DataFrame, prices: pd.DataFrame,
                  daily: pd.DataFrame, summary: list[dict], metrics: dict, predictions_log: pd.DataFrame) -> Path:
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    p = posts.copy()
    p["created_at_utc"] = _iso_utc(p["created_at_utc"])
    p[list(POSTS_COLUMNS)].to_csv(run_dir / "posts.csv", index=False, encoding="utf-8")
    n = news.copy()
    n["published_at_utc"] = _iso_utc(n["published_at_utc"])
    n[list(NEWS_COLUMNS)].to_csv(run_dir / "news.csv", index=False, encoding="utf-8")
    prices[list(PRICES_COLUMNS)].to_csv(run_dir / "prices.csv", index=False, encoding="utf-8")
    daily[list(DAILY_COLUMNS)].to_csv(run_dir / "daily.csv", index=False, encoding="utf-8")
    predictions_log[list(PREDICTIONS_LOG_COLUMNS)].to_csv(run_dir / "predictions_log.csv", index=False)
    (run_dir / "summary.json").write_text(
        json.dumps(_clean_json(summary), ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps(_clean_json(metrics), ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")

    latest = out_root / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    for f in run_dir.iterdir():
        shutil.copy2(f, latest / f.name)
    (latest / "RUN_ID").write_text(run_id + "\n", encoding="utf-8")
    logger.info("Wrote outputs to %s and %s", run_dir, latest)
    return run_dir


# ----------------------------------------------------------------------- validator


def _is_num_or_null(v: Any) -> bool:
    return v is None or (isinstance(v, (int, float)) and not isinstance(v, bool))


def _check_csv(path: Path, columns: tuple[str, ...], errors: list[str]) -> pd.DataFrame | None:
    if not path.is_file():
        errors.append(f"{path.name}: missing")
        return None
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if tuple(df.columns) != columns:
        errors.append(f"{path.name}: header {list(df.columns)} != contract {list(columns)}")
        return None
    return df


def _numeric(df: pd.DataFrame, col: str, name: str, errors: list[str], *, nullable: bool = True,
             lo: float | None = None, hi: float | None = None, integer: bool = False) -> None:
    raw = df[col]
    empty = raw == ""
    if not nullable and empty.any():
        errors.append(f"{name}.{col}: {int(empty.sum())} empty value(s)")
    vals = pd.to_numeric(raw[~empty], errors="coerce")
    if vals.isna().any():
        errors.append(f"{name}.{col}: {int(vals.isna().sum())} non-numeric value(s)")
    vals = vals.dropna()
    if integer and (vals % 1 != 0).any():
        errors.append(f"{name}.{col}: non-integer value(s)")
    if lo is not None and (vals < lo).any():
        errors.append(f"{name}.{col}: value(s) below {lo}")
    if hi is not None and (vals > hi).any():
        errors.append(f"{name}.{col}: value(s) above {hi}")


def _dates(df: pd.DataFrame, col: str, name: str, errors: list[str], *, nullable: bool = False) -> None:
    raw = df[col]
    empty = raw == ""
    if not nullable and empty.any():
        errors.append(f"{name}.{col}: {int(empty.sum())} empty value(s)")
    bad = pd.to_datetime(raw[~empty], errors="coerce", utc=True).isna()
    if bad.any():
        errors.append(f"{name}.{col}: {int(bad.sum())} unparseable date(s)")


def validate_outputs(run_dir: Path, expected_tickers: list[str] | None = None) -> list[str]:
    """Return a list of contract violations (empty list = valid)."""
    errors: list[str] = []
    labels = {"bullish", "bearish", "neutral"}

    posts = _check_csv(run_dir / "posts.csv", POSTS_COLUMNS, errors)
    if posts is not None:
        for c in ("likes", "reposts", "views", "author_followers"):
            _numeric(posts, c, "posts", errors, nullable=False, lo=0, integer=True)
        _numeric(posts, "sentiment_score", "posts", errors, lo=-1, hi=1)
        _numeric(posts, "weight", "posts", errors, lo=0)
        _dates(posts, "created_at_utc", "posts", errors)
        _dates(posts, "bar_date", "posts", errors)
        bad = ~posts["sentiment_label"].isin(labels | {""})
        if bad.any():
            errors.append(f"posts.sentiment_label: {int(bad.sum())} invalid label(s)")
        if (posts["text"] == "").any():
            errors.append("posts.text: empty text")
        if posts["post_id"].duplicated().any():
            errors.append("posts.post_id: duplicates")

    news = _check_csv(run_dir / "news.csv", NEWS_COLUMNS, errors)
    if news is not None:
        _numeric(news, "sentiment_score", "news", errors, lo=-1, hi=1)
        _dates(news, "published_at_utc", "news", errors)
        _dates(news, "bar_date", "news", errors)

    prices = _check_csv(run_dir / "prices.csv", PRICES_COLUMNS, errors)
    if prices is not None:
        for c in ("open", "high", "low", "close"):
            _numeric(prices, c, "prices", errors, nullable=False, lo=0)
        _numeric(prices, "volume", "prices", errors, nullable=False, lo=0, integer=True)
        _dates(prices, "date", "prices", errors)
        counts = prices.groupby("ticker").size()
        for t in (expected_tickers or counts.index.tolist()):
            if counts.get(t, 0) < 90:
                errors.append(f"prices: {t} has {counts.get(t, 0)} bars (< 90)")

    daily = _check_csv(run_dir / "daily.csv", DAILY_COLUMNS, errors)
    if daily is not None:
        _numeric(daily, "mentions", "daily", errors, nullable=False, lo=0, integer=True)
        _numeric(daily, "news_count", "daily", errors, nullable=False, lo=0, integer=True)
        _numeric(daily, "bullish_ratio", "daily", errors, lo=0, hi=1)
        for c in ("social_sent_mean", "social_sent_weighted", "news_sent_mean"):
            _numeric(daily, c, "daily", errors, lo=-1, hi=1)
        _numeric(daily, "close", "daily", errors, nullable=False, lo=0)
        _numeric(daily, "return_1d", "daily", errors)
        _dates(daily, "date", "daily", errors)

    log = _check_csv(run_dir / "predictions_log.csv", PREDICTIONS_LOG_COLUMNS, errors)
    if log is not None:
        for c in ("predicted_close", "actual_close", "error", "error_pct"):
            _numeric(log, c, "predictions_log", errors)
        _numeric(log, "direction_hit", "predictions_log", errors, lo=0, hi=1, integer=True)
        _dates(log, "target_date", "predictions_log", errors)

    errors += _validate_summary(run_dir / "summary.json", expected_tickers)
    errors += _validate_metrics(run_dir / "metrics.json", expected_tickers)
    return errors


def _validate_summary(path: Path, expected: list[str] | None) -> list[str]:
    errs: list[str] = []
    if not path.is_file():
        return ["summary.json: missing"]
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return ["summary.json: not a list"]
    seen = set()
    for item in data:
        t = item.get("ticker", "?")
        seen.add(t)
        where = f"summary[{t}]"
        for k in ("exchange", "ticker", "name", "currency", "as_of", "direction"):
            if not isinstance(item.get(k), str):
                errs.append(f"{where}.{k}: expected string")
        for k in ("current_price", "mentions_change_pct", "sentiment_score", "news_sentiment", "confidence"):
            if k not in item or not _is_num_or_null(item[k]):
                errs.append(f"{where}.{k}: expected number or null")
        if not isinstance(item.get("mentions_10d"), int):
            errs.append(f"{where}.mentions_10d: expected integer")
        if item.get("direction") not in ("bullish", "bearish", "neutral"):
            errs.append(f"{where}.direction: invalid")
        for k in ("support", "resistance"):
            v = item.get(k)
            if not (isinstance(v, list) and len(v) == 2 and all(_is_num_or_null(x) for x in v)):
                errs.append(f"{where}.{k}: expected list of 2 numbers")
        pos = item.get("position")
        if not isinstance(pos, dict) or pos.get("side") not in ("long", "short", "none") or \
                not all(k in pos and _is_num_or_null(pos[k]) for k in ("entry", "stop", "target", "risk_reward")):
            errs.append(f"{where}.position: invalid")
        pred = item.get("prediction")
        if not isinstance(pred, dict) or "target_date" not in pred or \
                not all(k in pred and _is_num_or_null(pred[k]) for k in ("predicted_close", "low", "high")):
            errs.append(f"{where}.prediction: invalid")
        for tp in item.get("top_posts", None) or []:
            if not all(k in tp for k in ("source", "text", "sentiment_score", "url")):
                errs.append(f"{where}.top_posts: missing keys")
                break
        for tn in item.get("top_news", None) or []:
            if not all(k in tn for k in ("title", "url", "sentiment_score")):
                errs.append(f"{where}.top_news: missing keys")
                break
        if not isinstance(item.get("top_posts"), list) or not isinstance(item.get("top_news"), list):
            errs.append(f"{where}: top_posts/top_news must be lists")
    for t in expected or []:
        if t not in seen:
            errs.append(f"summary.json: missing ticker {t}")
    return errs


def _validate_metrics(path: Path, expected: list[str] | None) -> list[str]:
    errs: list[str] = []
    if not path.is_file():
        return ["metrics.json: missing"]
    data = json.loads(path.read_text(encoding="utf-8"))

    def check(entry: Any, where: str) -> None:
        for m in METRIC_MODELS:
            e = entry.get(m) if isinstance(entry, dict) else None
            if not isinstance(e, dict):
                errs.append(f"{where}.{m}: missing")
                continue
            for k in METRIC_KEYS:
                if k not in e or not _is_num_or_null(e[k]):
                    errs.append(f"{where}.{m}.{k}: expected number or null")

    if not isinstance(data, dict) or "overall" not in data or "per_ticker" not in data:
        return ["metrics.json: needs 'overall' and 'per_ticker'"]
    check(data["overall"], "metrics.overall")
    for t, entry in data["per_ticker"].items():
        check(entry, f"metrics.per_ticker[{t}]")
    for t in expected or []:
        if t not in data["per_ticker"]:
            errs.append(f"metrics.json: per_ticker missing {t}")
    return errs
