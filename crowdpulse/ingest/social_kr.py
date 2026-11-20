"""KRX social sources: Naver Finance discussion + Daum Finance talks.

Both sites serve their boards from open JSON endpoints, so these are fetched
directly (polite rate limit, retries) instead of through a paid Apify run:
  Naver: m.stock.naver.com/front-api/discussion/list  (offset paging, KST times)
  Daum:  finance.daum.net/content/debates/A<code>      (page paging, KST times)

Very busy boards (Samsung can see 10k+ posts a day) are sampled with a per-day
cap: once a day has `per_day_cap` posts, the crawler jumps to the previous day
using Naver's global post-id sequence, so all 10 days get covered.
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import date, datetime, time as dtime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

from crowdpulse.config import Settings, TickerConfig
from crowdpulse.ingest.cache import load_raw, raw_path, save_raw
from crowdpulse.ingest.http import GetJson, make_get_json
from crowdpulse.ingest.social_common import join_title_body, make_row, parse_local_time

logger = logging.getLogger(__name__)

KST = ZoneInfo("Asia/Seoul")
NAVER_LIST_URL = "https://m.stock.naver.com/front-api/discussion/list"
NAVER_POST_URL = "https://m.stock.naver.com/domestic/stock/{code}/discussion/{id}"
DAUM_LIST_URL = "https://finance.daum.net/content/debates/A{code}"
DAUM_POST_URL = "https://finance.daum.net/domestic/stocks/{code}/talks/{id}"

DEFAULT_PER_DAY_CAP = 500
NAVER_MAX_PAGES = 200
DAUM_MAX_PAGES = 60
# Fallback for Naver's global id rate (~55k posts/day across all stocks).
_NAVER_IDS_PER_SEC = 0.64


# ------------------------------------------------------------------------- Naver


def _naver_url(code: str, offset: str | None, page_size: int = 100) -> str:
    q = {
        "discussionType": "domesticStock", "itemCode": code, "pageSize": page_size,
        "isHolderOnly": "false", "excludesItemNews": "false",
        "isItemNewsOnly": "false", "isCleanbotPassedOnly": "false",
    }
    if offset is not None:
        q["offset"] = offset
    return f"{NAVER_LIST_URL}?{urlencode(q)}"


def _estimate_naver_id(anchors: list[tuple[int, datetime]], target: datetime) -> int | None:
    """Estimate the global post id at `target` from (id, time) pairs seen so far."""
    if not anchors:
        return None
    newest = max(anchors, key=lambda a: a[0])
    oldest = min(anchors, key=lambda a: a[0])
    span = (newest[1] - oldest[1]).total_seconds()
    rate = (newest[0] - oldest[0]) / span if span >= 600 else _NAVER_IDS_PER_SEC
    rate = max(rate, 0.05)
    est = oldest[0] - rate * (oldest[1] - target).total_seconds()
    return int(est) if est > 0 else None


def crawl_naver(
    code: str, *, start_utc: datetime, now_utc: datetime, get_json: GetJson,
    per_day_cap: int = DEFAULT_PER_DAY_CAP, max_pages: int = NAVER_MAX_PAGES,
) -> list[dict[str, Any]]:
    """Raw Naver posts for one stock from now back to start_utc, at most per_day_cap per KST day."""
    kept: dict[str, dict[str, Any]] = {}
    per_day: Counter[date] = Counter()
    anchors: list[tuple[int, datetime]] = []
    offset: str | None = None
    for page_no in range(1, max_pages + 1):
        data = get_json(_naver_url(code, offset))
        result = (data or {}).get("result") or {}
        batch = result.get("posts") or []
        if not batch:
            break
        times: list[datetime] = []
        for p in batch:
            t = parse_local_time(p.get("writtenAt"), KST, now_utc)
            if t is None:
                continue
            times.append(t)
            try:
                anchors.append((int(p["id"]), t))
            except (KeyError, TypeError, ValueError):
                pass
            if t < start_utc or str(p.get("id")) in kept:
                continue
            d = t.astimezone(KST).date()
            if per_day[d] >= per_day_cap:
                continue
            kept[str(p["id"])] = p
            per_day[d] += 1
        if not times:
            break
        oldest_t = min(times)
        last_offset = result.get("lastOffset")
        if oldest_t < start_utc or not last_offset:
            break
        oldest_day = oldest_t.astimezone(KST).date()
        offset = str(last_offset)
        if per_day[oldest_day] >= per_day_cap:
            # Day is full: jump to just before local midnight of that day.
            target = datetime.combine(oldest_day, dtime(0, 0), tzinfo=KST).astimezone(timezone.utc)
            if target <= start_utc:
                break
            est = _estimate_naver_id(anchors, target)
            current_id = min(a[0] for a in anchors)
            if est is not None and est < current_id:
                offset = str(-est)
                logger.debug("naver %s: day %s full, jumping to id ~%d", code, oldest_day, est)
    else:
        logger.warning("naver %s: hit max_pages=%d before reaching %s", code, max_pages, start_utc)
    logger.info("naver %s: %d posts over %d days (cap %d/day)", code, len(kept), len(per_day), per_day_cap)
    return list(kept.values())


def naver_rows(raw: list[dict[str, Any]], ticker: TickerConfig, *, scraped_at: datetime) -> list[dict[str, Any]]:
    rows = []
    for p in raw:
        pid = p.get("id")
        if pid is None:
            continue
        rows.append(make_row(
            source="naver", native_id=pid, exchange=ticker.exchange, ticker=ticker.symbol,
            created_at_utc=parse_local_time(p.get("writtenAt"), KST, scraped_at),
            text=join_title_body(p.get("title"), p.get("contentSwReplacedButImg") or p.get("contentSwReplaced")),
            url=NAVER_POST_URL.format(code=ticker.code, id=pid),
            likes=p.get("recommendCount"), views=p.get("viewCount"),
        ))
    return rows


# -------------------------------------------------------------------------- Daum


def crawl_daum(
    code: str, *, start_utc: datetime, now_utc: datetime, get_json: GetJson,
    max_pages: int = DAUM_MAX_PAGES, per_page: int = 100,
) -> list[dict[str, Any]]:
    """Raw Daum talk posts for one stock from now back to start_utc."""
    kept: dict[str, dict[str, Any]] = {}
    for page in range(1, max_pages + 1):
        data = get_json(f"{DAUM_LIST_URL.format(code=code)}?{urlencode({'page': page, 'perPage': per_page})}")
        batch = ((data or {}).get("data") or {}).get("posts") or []
        if not batch:
            break
        oldest: datetime | None = None
        for p in batch:
            t = parse_local_time(p.get("createdAt"), KST, now_utc)
            if t is None:
                continue
            oldest = t if oldest is None or t < oldest else oldest
            if t >= start_utc:
                kept.setdefault(str(p.get("id")), p)
        total_pages = (data or {}).get("totalPages")
        if oldest is None or oldest < start_utc or (total_pages and page >= int(total_pages)):
            break
    logger.info("daum %s: %d posts", code, len(kept))
    return list(kept.values())


def daum_rows(raw: list[dict[str, Any]], ticker: TickerConfig, *, scraped_at: datetime) -> list[dict[str, Any]]:
    rows = []
    for p in raw:
        pid = p.get("id")
        if pid is None:
            continue
        rows.append(make_row(
            source="daum", native_id=pid, exchange=ticker.exchange, ticker=ticker.symbol,
            created_at_utc=parse_local_time(p.get("createdAt"), KST, scraped_at),
            text=join_title_body(p.get("title"), p.get("content")),
            url=DAUM_POST_URL.format(code=ticker.code, id=pid),
            likes=p.get("agreeCount"), views=p.get("readCount"),
        ))
    return rows


# --------------------------------------------------------------------- entry points


def _fetch(
    source: str, crawler, to_rows, tickers: list[TickerConfig], settings: Settings, *,
    use_cache: bool, now: datetime, get_json: GetJson | None, **crawl_kw: Any,
) -> list[dict[str, Any]]:
    start = now - timedelta(days=settings.lookback_days)
    get_json = get_json or make_get_json(headers={"Referer": "https://finance.daum.net/"} if source == "daum" else None)
    rows: list[dict[str, Any]] = []
    for t in tickers:
        path = raw_path(settings.data_dir, source, t.symbol, now.date())
        params = {"code": t.code, "lookback_days": settings.lookback_days, **crawl_kw}
        cached = load_raw(path, params=params) if use_cache else None
        if cached is not None:
            raw = cached["items"]
            scraped_at = datetime.fromisoformat(cached["fetched_at_utc"])
        else:
            try:
                raw = crawler(t.code, start_utc=start, now_utc=now, get_json=get_json, **crawl_kw)
            except Exception as exc:  # noqa: BLE001 - one ticker failing must not stop the run
                logger.error("%s %s failed: %s", source, t.symbol, exc)
                continue
            save_raw(path, raw, source=source, key=t.symbol, params=params)
            scraped_at = now
        rows.extend(to_rows(raw, t, scraped_at=scraped_at))
    return rows


def fetch_naver(tickers: list[TickerConfig], settings: Settings, *, use_cache: bool = False,
                now: datetime | None = None, get_json: GetJson | None = None,
                per_day_cap: int = DEFAULT_PER_DAY_CAP) -> list[dict[str, Any]]:
    return _fetch("naver", crawl_naver, naver_rows, tickers, settings, use_cache=use_cache,
                  now=now or datetime.now(timezone.utc), get_json=get_json, per_day_cap=per_day_cap)


def fetch_daum(tickers: list[TickerConfig], settings: Settings, *, use_cache: bool = False,
               now: datetime | None = None, get_json: GetJson | None = None,
               per_day_cap: int = DEFAULT_PER_DAY_CAP) -> list[dict[str, Any]]:
    # Daum volumes are low; per_day_cap is accepted for a uniform signature but not needed.
    return _fetch("daum", crawl_daum, daum_rows, tickers, settings, use_cache=use_cache,
                  now=now or datetime.now(timezone.utc), get_json=get_json)
