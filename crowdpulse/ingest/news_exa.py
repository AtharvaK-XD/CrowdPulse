"""News per ticker from Exa: English + local-language queries over the lookback window.

Run live for one ticker:
    python -m crowdpulse.ingest.news_exa --ticker 0700.HK [--cache]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time as _time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

import pandas as pd

from crowdpulse.config import Settings, TickerConfig, get_secret, load_config
from crowdpulse.ingest.cache import load_raw, raw_path, save_raw
from crowdpulse.schema import NEWS_COLUMNS

logger = logging.getLogger(__name__)

SOURCE = "exa"
_QUERY_RETRIES = 2
_RESULT_FIELDS = ("id", "url", "title", "published_date", "author", "score", "text", "summary")


class NewsFetchError(RuntimeError):
    """Every Exa query for a ticker failed."""


def _default_client_factory(api_key: str) -> Any:
    from exa_py import Exa  # lazy: offline tests don't need exa-py

    return Exa(api_key)


def build_queries(ticker: TickerConfig) -> list[tuple[str, str]]:
    """(language, query) pairs: English first, then local language. Duplicates removed."""
    english = list(ticker.keywords_en) or [f"{ticker.name} {ticker.symbol} stock"]
    pairs = [("en", q) for q in english] + [("local", q) for q in ticker.keywords_local]
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for lang, q in pairs:
        q = q.strip()
        if q and q not in seen:
            seen.add(q)
            out.append((lang, q))
    return out


def _iso_z(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _result_to_dict(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return {k: result.get(k) for k in _RESULT_FIELDS}
    return {k: getattr(result, k, None) for k in _RESULT_FIELDS}


def _search(client: Any, query: str, *, num_results: int, start: datetime, end: datetime,
            max_chars: int) -> list[dict[str, Any]]:
    response = client.search(
        query,
        num_results=num_results,
        start_published_date=_iso_z(start),
        end_published_date=_iso_z(end),
        contents={"text": {"max_characters": max_chars}, "summary": True},
        type="auto",
    )
    results = response.get("results") if isinstance(response, dict) else getattr(response, "results", None)
    return [_result_to_dict(r) for r in (results or [])]


def fetch_news(
    symbol: str,
    *,
    settings: Settings | None = None,
    use_cache: bool = False,
    client: Any = None,
    client_factory: Callable[[str], Any] = _default_client_factory,
    now: datetime | None = None,
    sleep: Callable[[float], None] = _time.sleep,
) -> list[dict[str, Any]]:
    """Raw Exa results for one ticker, each tagged with its query and language.

    Saved to data/raw/exa/<day>/<symbol>.json. A run where some queries failed is
    saved as incomplete, so --cache will not reuse it.
    """
    settings = settings or load_config()
    ticker = settings.ticker(symbol)
    now = now or datetime.now(timezone.utc)
    start = now - timedelta(days=settings.lookback_days)
    queries = build_queries(ticker)
    path = raw_path(settings.data_dir, SOURCE, symbol, now.date())
    params = {
        "queries": [q for _, q in queries],
        "lookback_days": settings.lookback_days,
        "num_results": settings.news.results_per_query,
    }

    if use_cache:
        cached = load_raw(path, params=params)
        if cached is not None:
            return cached["items"]

    if client is None:
        client = client_factory(get_secret("EXA_API_KEY"))

    items: list[dict[str, Any]] = []
    failed: list[str] = []
    for lang, query in queries:
        results = _search_with_retry(client, query, settings, start, now, sleep)
        if results is None:
            failed.append(query)
            continue
        for r in results:
            items.append({**r, "query": query, "query_lang": lang})

    if failed and len(failed) == len(queries):
        raise NewsFetchError(f"All {len(queries)} Exa queries failed for {symbol}")
    if failed:
        logger.warning("%s: %d of %d Exa queries failed: %s", symbol, len(failed), len(queries), failed)

    save_raw(path, items, source=SOURCE, key=symbol, params=params, complete=not failed,
             extra={"failed_queries": failed, "window_start_utc": _iso_z(start),
                    "window_end_utc": _iso_z(now)})
    return items


def _search_with_retry(client: Any, query: str, settings: Settings, start: datetime, end: datetime,
                       sleep: Callable[[float], None]) -> list[dict[str, Any]] | None:
    for attempt in range(_QUERY_RETRIES + 1):
        try:
            return _search(client, query, num_results=settings.news.results_per_query,
                           start=start, end=end, max_chars=settings.news.max_text_chars)
        except Exception as exc:  # noqa: BLE001 - network/API errors all retried
            if attempt == _QUERY_RETRIES:
                logger.error("Exa query %r failed: %s", query, exc)
                return None
            delay = 2.0 * 2 ** attempt
            logger.warning("Exa query %r failed (%s); retrying in %.0fs", query, exc, delay)
            sleep(delay)
    return None


def _canonical_url(url: str) -> str:
    parts = urlsplit(url.strip())
    path = parts.path.rstrip("/") or "/"
    host = parts.netloc.lower()
    host = host[4:] if host.startswith("www.") else host
    scheme = "https" if parts.scheme.lower() in ("http", "https") else parts.scheme.lower()
    return urlunsplit((scheme, host, path, parts.query, ""))


def _domain(url: str) -> str | None:
    host = urlsplit(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host or None


def _clean_str(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def normalise_news(
    items: list[dict[str, Any]],
    symbol: str,
    *,
    now: datetime,
    lookback_days: int,
    include_text: bool = False,
) -> pd.DataFrame:
    """Raw Exa items -> rows in the news.csv columns.

    Drops items with no URL, no publish date, or a date outside [now - lookback, now].
    sentiment_score and bar_date are left null (filled in Phases 4 and 5).
    """
    start = now - timedelta(days=lookback_days)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    dropped = {"no_url": 0, "no_date": 0, "out_of_window": 0, "duplicate": 0}

    for item in items:
        url = _clean_str(item.get("url"))
        if not url:
            dropped["no_url"] += 1
            continue
        key = _canonical_url(url)
        if key in seen:
            dropped["duplicate"] += 1
            continue
        published = pd.to_datetime(item.get("published_date"), utc=True, errors="coerce")
        if pd.isna(published):
            dropped["no_date"] += 1
            continue
        if not (start <= published.to_pydatetime() <= now):
            dropped["out_of_window"] += 1
            continue
        seen.add(key)
        row = {
            "ticker": symbol,
            "title": _clean_str(item.get("title")),
            "url": url,
            "published_at_utc": published,
            "source_domain": _domain(url),
            "summary": _clean_str(item.get("summary")),
            "sentiment_score": None,
            "bar_date": None,
        }
        if include_text:
            row["text"] = _clean_str(item.get("text"))
        rows.append(row)

    if any(dropped.values()):
        logger.info("%s news: kept %d, dropped %s", symbol, len(rows), dropped)

    columns = list(NEWS_COLUMNS) + (["text"] if include_text else [])
    df = pd.DataFrame(rows, columns=columns)
    df["published_at_utc"] = pd.to_datetime(df["published_at_utc"], utc=True)
    df["sentiment_score"] = df["sentiment_score"].astype("float64")
    return df.sort_values("published_at_utc", ascending=False, kind="stable").reset_index(drop=True)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch Exa news for one or all tickers.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ticker", help="e.g. 0700.HK")
    group.add_argument("--all", action="store_true", help="every ticker in config.yaml")
    parser.add_argument("--cache", action="store_true", help="reuse today's raw JSON if present")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    settings = load_config(args.config)
    symbols = [t.symbol for t in settings.tickers] if args.all else [args.ticker]
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(days=settings.lookback_days)
    ok = True
    for symbol in symbols:
        raw = fetch_news(symbol, settings=settings, use_cache=args.cache, now=now)
        df = normalise_news(raw, symbol, now=now, lookback_days=settings.lookback_days)
        print(f"\n{symbol}: {len(raw)} raw results -> {len(df)} articles")
        if df.empty:
            ok = False
            continue
        lo, hi = df["published_at_utc"].min(), df["published_at_utc"].max()
        inside = bool(lo >= window_start and hi <= now)
        ok = ok and inside
        print(f"  published range: {lo:%Y-%m-%d %H:%M} .. {hi:%Y-%m-%d %H:%M} UTC "
              f"(window {window_start:%Y-%m-%d} .. {now:%Y-%m-%d}) inside={inside}")
        for _, r in df.head(5).iterrows():
            print(f"  {r['published_at_utc']:%Y-%m-%d} {r['source_domain']}: {r['title']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
