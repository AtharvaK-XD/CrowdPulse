"""Collect posts from every configured social source, with fallback to backups.

For each exchange, the sources in config.yaml run first. A source with no
scraper (e.g. aastocks) or one that returns nothing for a ticker is replaced by
the exchange's backup_sources, so each ticker aims for two working sources.
Switching sources is a config.yaml change, not a code change.

Live check for one ticker (Phase 2 gate):
    python -m crowdpulse.ingest.social --ticker 005930.KS
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd

from crowdpulse.config import Settings, load_config
from crowdpulse.ingest.social_common import finalize_posts
from crowdpulse.ingest.social_hk import fetch_futu, fetch_lihkg
from crowdpulse.ingest.social_kr import DEFAULT_PER_DAY_CAP, fetch_daum, fetch_naver

logger = logging.getLogger(__name__)

Fetcher = Callable[..., list[dict[str, Any]]]
SOURCES: dict[str, Fetcher] = {
    "naver": fetch_naver,
    "daum": fetch_daum,
    "futu": fetch_futu,
    "lihkg": fetch_lihkg,
}
TARGET_SOURCES_PER_TICKER = 2


def collect_posts(
    settings: Settings,
    *,
    exchanges: list[str] | None = None,
    tickers: list[str] | None = None,
    use_cache: bool = False,
    now: datetime | None = None,
    per_day_cap: int = DEFAULT_PER_DAY_CAP,
    fetchers: dict[str, Fetcher] | None = None,
) -> pd.DataFrame:
    """Scrape all sources for the selected exchanges/tickers -> posts DataFrame (no sentiment yet)."""
    now = now or datetime.now(timezone.utc)
    registry = fetchers if fetchers is not None else SOURCES
    rows: list[dict[str, Any]] = []
    for ex_code, ex in settings.exchanges.items():
        if exchanges and ex_code not in exchanges:
            continue
        ex_tickers = [t for t in settings.tickers_for(ex_code) if not tickers or t.symbol in tickers]
        if not ex_tickers:
            continue
        sources_seen: dict[str, set[str]] = {t.symbol: set() for t in ex_tickers}
        candidates = list(ex.sources) + [b for b in ex.backup_sources if b not in ex.sources]
        for name in candidates:
            is_backup = name not in ex.sources
            need = [t for t in ex_tickers if len(sources_seen[t.symbol]) < TARGET_SOURCES_PER_TICKER]
            if not need:
                break
            if is_backup:
                logger.info("%s: using backup source %s for %s", ex_code, name, [t.symbol for t in need])
            fetcher = registry.get(name)
            if fetcher is None:
                logger.warning("No scraper for source %r; skipping", name)
                continue
            try:
                got = fetcher(need, settings, use_cache=use_cache, now=now, per_day_cap=per_day_cap)
            except Exception as exc:  # noqa: BLE001
                logger.error("Source %s failed: %s", name, exc)
                continue
            for r in got:
                sources_seen.setdefault(r["ticker"], set()).add(name)
            rows.extend(got)
        for sym, found in sources_seen.items():
            if len(found) < TARGET_SOURCES_PER_TICKER:
                logger.warning("%s has posts from %d source(s): %s", sym, len(found), sorted(found))
    return finalize_posts(rows, now=now, lookback_days=settings.lookback_days)


def source_summary(posts: pd.DataFrame) -> pd.DataFrame:
    """Posts per ticker per source with first/last timestamp."""
    if posts.empty:
        return pd.DataFrame(columns=["ticker", "source", "posts", "first_utc", "last_utc"])
    g = posts.groupby(["ticker", "source"])["created_at_utc"]
    return pd.DataFrame({"posts": g.size(), "first_utc": g.min(), "last_utc": g.max()}).reset_index()


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scrape social posts and print samples.")
    parser.add_argument("--ticker", action="append", help="repeatable; default: all")
    parser.add_argument("--exchange", choices=["HKEX", "KRX"], action="append")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--per-day-cap", type=int, default=DEFAULT_PER_DAY_CAP)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    settings = load_config()
    now = datetime.now(timezone.utc)
    posts = collect_posts(settings, exchanges=args.exchange, tickers=args.ticker,
                          use_cache=args.cache, now=now, per_day_cap=args.per_day_cap)
    print("\n" + source_summary(posts).to_string(index=False))
    future = int((posts["created_at_utc"] > pd.Timestamp(now)).sum()) if not posts.empty else 0
    print(f"\nTotal posts: {len(posts)}; future timestamps: {future}")
    cols = ["source", "ticker", "created_at_utc", "likes", "views", "text", "url"]
    for (ticker, source), grp in posts.groupby(["ticker", "source"]):
        print(f"\n--- {ticker} / {source}: {args.samples} sample rows ---")
        sample = grp.head(args.samples)[cols].copy()
        sample["text"] = sample["text"].str.slice(0, 80)
        print(sample.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
