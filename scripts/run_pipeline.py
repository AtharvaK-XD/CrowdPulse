"""CrowdPulse pipeline: scrape -> sentiment -> align -> features -> model -> export -> validate.

    python scripts/run_pipeline.py                       # full live run, all tickers
    python scripts/run_pipeline.py --exchange KRX --cache
    python scripts/run_pipeline.py --skip-scrape         # reuse data/outputs/latest posts + news

Options:
  --exchange HKEX|KRX|all   which tickers to process (default all)
  --cache                   reuse today's raw API responses instead of calling APIs again
  --skip-scrape             no scraping: reuse posts.csv/news.csv from data/outputs/latest
                            (prices are still refreshed unless --cache)
  --llm-sentiment           score posts with Claude (needs `anthropic` + ANTHROPIC_API_KEY)
  --per-day-cap N           max Naver posts per stock per day (default 500)

Exit code 0 when the contract validator passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crowdpulse.config import load_config  # noqa: E402
from crowdpulse.export import build_summary, make_run_id, validate_outputs, write_outputs  # noqa: E402
from crowdpulse.features.align import assign_bar_dates, calendars  # noqa: E402
from crowdpulse.features.daily import build_daily, build_daily_all, build_features  # noqa: E402
from crowdpulse.ingest.news_exa import fetch_news, normalise_news  # noqa: E402
from crowdpulse.ingest.prices import fetch_prices  # noqa: E402
from crowdpulse.ingest.social import collect_posts, source_summary  # noqa: E402
from crowdpulse.ingest.social_common import finalize_posts  # noqa: E402
from crowdpulse.ingest.social_kr import DEFAULT_PER_DAY_CAP  # noqa: E402
from crowdpulse.model.evaluate import compute_metrics, residual_quantiles, walk_forward  # noqa: E402
from crowdpulse.model.predict import append_log, predict_next  # noqa: E402
from crowdpulse.nlp.sentiment import default_scorer, score_news, score_posts  # noqa: E402
from crowdpulse.schema import NEWS_COLUMNS  # noqa: E402

logger = logging.getLogger("crowdpulse.pipeline")


def _load_previous(out_root: Path, now: datetime, lookback_days: int, tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    latest = out_root / "latest"
    if not (latest / "posts.csv").is_file():
        raise SystemExit("--skip-scrape needs a previous run in data/outputs/latest/")
    posts = pd.read_csv(latest / "posts.csv", dtype={"post_id": str, "ticker": str})
    posts = posts[posts["ticker"].isin(tickers)]
    posts = finalize_posts(posts.to_dict("records"), now=now, lookback_days=lookback_days)
    news = pd.read_csv(latest / "news.csv", dtype={"ticker": str})
    news = news[news["ticker"].isin(tickers)]
    news["published_at_utc"] = pd.to_datetime(news["published_at_utc"], utc=True, errors="coerce")
    return posts, news


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exchange", choices=["HKEX", "KRX", "all"], default="all")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--skip-scrape", action="store_true")
    parser.add_argument("--llm-sentiment", action="store_true")
    parser.add_argument("--per-day-cap", type=int, default=DEFAULT_PER_DAY_CAP)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    settings = load_config(args.config)
    now = datetime.now(timezone.utc)
    run_id = make_run_id(now)
    exchanges = None if args.exchange == "all" else [args.exchange]
    tickers = [t.symbol for t in settings.tickers if exchanges is None or t.exchange in exchanges]
    out_root = settings.data_dir / "outputs"
    logger.info("Run %s for %d tickers", run_id, len(tickers))

    # 1. Prices (also the trading calendars)
    prices = fetch_prices(settings, tickers=tickers, use_cache=args.cache, now=now)
    if prices.empty:
        logger.error("No prices downloaded; stopping")
        return 1
    cals = calendars(prices, settings)

    # 2. Posts and news
    if args.skip_scrape:
        posts, news = _load_previous(out_root, now, settings.lookback_days, tickers)
    else:
        posts = collect_posts(settings, exchanges=exchanges, use_cache=args.cache, now=now,
                              per_day_cap=args.per_day_cap)
        frames = []
        for sym in tickers:
            try:
                raw = fetch_news(sym, settings=settings, use_cache=args.cache, now=now)
                frames.append(normalise_news(raw, sym, now=now, lookback_days=settings.lookback_days))
            except Exception as exc:  # noqa: BLE001 - one ticker's news failing must not stop the run
                logger.error("News for %s failed: %s", sym, exc)
        news = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(NEWS_COLUMNS))

    # 3. Sentiment and weights
    scorer = default_scorer(settings.data_dir, llm=args.llm_sentiment)
    posts = score_posts(posts, scorer)
    news = score_news(news, scorer)

    # 4. Alignment to trading bars
    posts = assign_bar_dates(posts, "created_at_utc", settings, cals)
    news = assign_bar_dates(news, "published_at_utc", settings, cals)

    # 5. Daily table and model features
    window_start = (now - timedelta(days=settings.lookback_days)).date().isoformat()
    daily = build_daily(posts, news, prices, window_start)
    features = build_features(build_daily_all(posts, news, prices), window_start)

    # 6. Model: walk-forward validation, then live prediction
    wf = walk_forward(features)
    metrics = compute_metrics(wf, tickers)
    exchange_of = {t.symbol: t.exchange for t in settings.tickers}
    predictions = predict_next(features, cals, exchange_of, residual_quantiles(wf))
    log = append_log(out_root / "predictions_log.csv", run_id, predictions)

    # 7. Summary, export, validation
    summary = build_summary(settings, posts=posts, news=news, prices=prices, daily=daily,
                            predictions=predictions, tickers=tickers)
    run_dir = write_outputs(out_root, run_id, posts=posts, news=news, prices=prices, daily=daily,
                            summary=summary, metrics=metrics, predictions_log=log)
    errors = validate_outputs(run_dir, expected_tickers=tickers)

    # 8. Report
    print(f"\n=== {run_id} ===  outputs: {run_dir}")
    print("\nPosts per ticker per source:")
    print(source_summary(posts)[["ticker", "source", "posts"]].to_string(index=False) if not posts.empty else "  none")
    print("\nNews per ticker:")
    print(news.groupby("ticker").size().rename("articles").to_string() if not news.empty else "  none")
    o = metrics["overall"]
    print(f"\nWalk-forward ({o['model']['n_predictions']} predictions):")
    for name in ("model", "no_sentiment", "naive"):
        e = o[name]
        fmt = lambda v, s="{:.4f}": "n/a" if v is None else s.format(v)  # noqa: E731
        print(f"  {name:>12}: MAE {fmt(e['mae'])}  RMSE {fmt(e['rmse'])}  MAPE {fmt(e['mape'], '{:.2f}%')}  "
              f"dir.acc {fmt(e['directional_accuracy'], '{:.0%}')}")
    print(f"  sentiment beats both baselines: {metrics.get('sentiment_helps')}")
    print("\nNext-day predictions:")
    for item in summary:
        p = item["prediction"]
        print(f"  {item['ticker']:>10} {item['direction']:>8} conf {item['confidence']:.2f}  "
              f"close {item['current_price']} -> {p['predicted_close']} [{p['low']}, {p['high']}] on {p['target_date']}"
              f"  position {item['position']['side']}")
    if errors:
        print(f"\nContract validator: FAILED ({len(errors)} problem(s))")
        for e in errors[:40]:
            print("  -", e)
        return 1
    print("\nContract validator: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
