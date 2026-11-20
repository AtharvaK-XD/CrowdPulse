"""HKEX social sources: Futu/Moomoo community + LIHKG, both via Apify.

Futu/Moomoo: futunn.com and moomoo.com serve a JavaScript bot check, and the full
community feed needs a login (visitors get "not support visitor"). The stock page
does render its latest discussion posts for visitors, so apify/playwright-scraper
opens https://www.moomoo.com/hk/stock/<code>-HK and returns that section's HTML,
which is parsed here. Coverage is therefore the most recent posts only (roughly
the last day), not the full 10 days.

LIHKG: Cloudflare blocks direct access, so the dedicated actor
claude_code_reviewer/lihkg-scraper runs in keyword-search mode (thread titles,
UTC timestamps, like counts) over the lookback window.

AAStocks no longer has a retail forum, so there is no AAStocks scraper; the
collector falls back to the configured backup source instead.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from lxml import html as lxml_html

from crowdpulse.config import Settings, TickerConfig
from crowdpulse.ingest.apify_client import proxy_input, run_actor
from crowdpulse.ingest.social_common import clean_text, join_title_body, make_row, parse_local_time

logger = logging.getLogger(__name__)

HKT = ZoneInfo("Asia/Hong_Kong")
PLAYWRIGHT_ACTOR = "apify/playwright-scraper"
LIHKG_ACTOR = "claude_code_reviewer/lihkg-scraper"
MOOMOO_STOCK_URL = "https://www.moomoo.com/hk/stock/{code}-HK"
LIHKG_MAX_ITEMS_PER_KEYWORD = 40
LIHKG_MAX_CHARGE_USD = 1.0

# Runs inside the browser on Apify. Returns the discussion section's HTML.
MOOMOO_PAGE_FUNCTION = r"""
async function pageFunction(context) {
  const { page, request } = context;
  await page.waitForTimeout(8000);
  for (let i = 0; i < 6; i++) { await page.mouse.wheel(0, 3000); await page.waitForTimeout(1500); }
  const html = await page.evaluate(() => {
    const h = document.querySelector('h2#comments');
    const card = h ? h.closest('.card-item') : null;
    if (card) return card.outerHTML;
    const list = document.querySelector('.nnq-feed-list');
    return list ? list.outerHTML : null;
  });
  return { ticker: request.userData.ticker, url: request.url, finalUrl: page.url(),
           scrapedAt: new Date().toISOString(), html };
}
"""


def moomoo_code(ticker: TickerConfig) -> str:
    """'0700.HK' -> '00700' (Moomoo uses 5-digit HK codes)."""
    return ticker.code.zfill(5)


def _first_text(node: Any, xpath: str) -> str:
    vals = node.xpath(xpath)
    return clean_text(" ".join(v if isinstance(v, str) else v.text_content() for v in vals)) if vals else ""


def parse_moomoo_section(html: str, ticker: TickerConfig, *, scraped_at: datetime) -> list[dict[str, Any]]:
    """Parse the Moomoo '股市討論區' section into posts rows."""
    if not html:
        return []
    doc = lxml_html.fromstring(html)
    rows = []
    for item in doc.xpath('//div[contains(concat(" ", normalize-space(@class), " "), " nnq-list-item ") and @fid]'):
        fid = item.get("fid")
        time_text = _first_text(item, './/span[contains(@class, "publish-time-text")]/text()')
        title = _first_text(item, './/div[contains(@class, "feed-title")]//a//text()')
        wrappers = item.xpath('.//div[contains(@class, "list-page__feed__wrapper")]')
        body = clean_text(wrappers[0].text_content()) if wrappers else ""
        likes = _first_text(item, './/div[contains(@class, "func-like")]//span[contains(@class, "num")]/text()')
        links = item.xpath('.//div[contains(@class, "publish-time")]//a/@href') or \
            item.xpath('.//div[contains(@class, "feed-title")]//a/@href')
        rows.append(make_row(
            source="futu", native_id=fid, exchange=ticker.exchange, ticker=ticker.symbol,
            created_at_utc=parse_local_time(time_text, HKT, scraped_at),
            text=join_title_body(title, body),
            url=links[0] if links else None,
            likes=likes or None,
        ))
    return rows


def fetch_futu(tickers: list[TickerConfig], settings: Settings, *, use_cache: bool = False,
               now: datetime | None = None, client: Any = None, **_: Any) -> list[dict[str, Any]]:
    now = now or datetime.now(timezone.utc)
    if not tickers:
        return []
    run_input = {
        "startUrls": [{"url": MOOMOO_STOCK_URL.format(code=moomoo_code(t)), "userData": {"ticker": t.symbol}}
                      for t in tickers],
        "pageFunction": MOOMOO_PAGE_FUNCTION,
        "proxyConfiguration": proxy_input(settings),
        "maxRequestsPerCrawl": len(tickers),
        "maxConcurrency": 2,
        "headless": True,
        "waitUntil": "domcontentloaded",
        "pageLoadTimeoutSecs": 90,
        "pageFunctionTimeoutSecs": 150,
    }
    items = run_actor(PLAYWRIGHT_ACTOR, run_input, source="futu", key="+".join(t.symbol for t in tickers),
                      settings=settings, use_cache=use_cache, client=client, day=now.date(),
                      memory_mbytes=2048)
    by_symbol = {t.symbol: t for t in tickers}
    rows: list[dict[str, Any]] = []
    for it in items:
        t = by_symbol.get(it.get("ticker"))
        if t is None:
            continue
        scraped = parse_local_time(it.get("scrapedAt"), HKT, now) or now
        parsed = parse_moomoo_section(it.get("html") or "", t, scraped_at=scraped)
        if not parsed:
            logger.warning("futu %s: no posts parsed (page %s)", t.symbol, it.get("finalUrl"))
        rows.extend(parsed)
    return rows


def lihkg_keywords(ticker: TickerConfig) -> list[str]:
    """Short company names for LIHKG title search: first word of each local keyword."""
    out: list[str] = []
    for kw in ticker.keywords_local:
        word = kw.split()[0].strip() if kw.strip() else ""
        if word and word not in out:
            out.append(word)
    return out


def lihkg_rows(items: list[dict[str, Any]], tickers: list[TickerConfig], *, scraped_at: datetime) -> list[dict[str, Any]]:
    by_kw: dict[str, TickerConfig] = {}
    for t in tickers:
        for kw in lihkg_keywords(t):
            by_kw.setdefault(kw, t)
    rows = []
    for it in items:
        t = by_kw.get(it.get("source_query") or "")
        tid = it.get("thread_id")
        if t is None or tid is None:
            continue
        rows.append(make_row(
            # A thread can match several stocks; keep post_id unique per ticker.
            source="lihkg", native_id=f"{tid}@{t.symbol}", exchange=t.exchange, ticker=t.symbol,
            created_at_utc=parse_local_time(it.get("created_at"), HKT, scraped_at),
            text=join_title_body(it.get("title"), it.get("own_text") or it.get("content")),
            url=it.get("thread_url") or f"https://lihkg.com/thread/{tid}/page/1",
            likes=it.get("like_count"),
        ))
    return rows


def fetch_lihkg(tickers: list[TickerConfig], settings: Settings, *, use_cache: bool = False,
                now: datetime | None = None, client: Any = None, **_: Any) -> list[dict[str, Any]]:
    now = now or datetime.now(timezone.utc)
    keywords = [kw for t in tickers for kw in lihkg_keywords(t)]
    if not keywords:
        return []
    run_input = {
        "mode": "search",
        "keywords": keywords,
        "maxItems": LIHKG_MAX_ITEMS_PER_KEYWORD,
        "order": "desc_create_time",
        "dateFrom": f"{settings.lookback_days}d",
        "proxyConfiguration": {"useApifyProxy": True},
    }
    items = run_actor(LIHKG_ACTOR, run_input, source="lihkg", key="+".join(t.symbol for t in tickers),
                      settings=settings, use_cache=use_cache, client=client, day=now.date(),
                      max_charge_usd=LIHKG_MAX_CHARGE_USD)
    start = now - timedelta(days=settings.lookback_days)
    rows = lihkg_rows(items, tickers, scraped_at=now)
    return [r for r in rows if r["created_at_utc"] is None or r["created_at_utc"] >= start]
