from __future__ import annotations

from datetime import datetime, timedelta, timezone

from crowdpulse.ingest.social import collect_posts
from crowdpulse.ingest.social_hk import (
    fetch_futu, fetch_lihkg, lihkg_keywords, lihkg_rows, moomoo_code, parse_moomoo_section,
)

from .conftest import SAMPLES, load_sample

NOW = datetime(2026, 11, 20, 8, 23, 4, tzinfo=timezone.utc)
OK_RUN = {"id": "r", "status": "SUCCEEDED", "defaultDatasetId": "ds"}


class FakeApify:
    def __init__(self, items):
        self.items = items
        self.inputs = []

    def actor(self, actor_id):
        outer = self

        class A:
            def call(self, *, run_input, timeout_secs, memory_mbytes=None):
                outer.inputs.append((actor_id, run_input))
                return OK_RUN

        return A()

    def dataset(self, _):
        outer = self

        class D:
            def iterate_items(self):
                yield from outer.items

        return D()


def test_moomoo_code(settings):
    assert moomoo_code(settings.ticker("0700.HK")) == "00700"


def test_parse_moomoo_sample(settings):
    html = (SAMPLES / "moomoo_section.html").read_text(encoding="utf-8")
    rows = parse_moomoo_section(html, settings.ticker("0700.HK"), scraped_at=NOW)
    assert len(rows) == 5
    first = rows[0]
    assert first["post_id"] == "futu:117647490875797"
    assert "RSI" in first["text"] and "清倉" in first["text"]
    assert first["likes"] == 5
    assert first["created_at_utc"] == NOW - timedelta(hours=1)  # "1小時前"
    assert first["url"].startswith("https://www.moomoo.com/")
    assert all(r["created_at_utc"] <= NOW for r in rows)


def test_parse_moomoo_empty(settings):
    assert parse_moomoo_section("", settings.ticker("0700.HK"), scraped_at=NOW) == []


def test_fetch_futu_with_fake_apify(settings):
    html = (SAMPLES / "moomoo_section.html").read_text(encoding="utf-8")
    fake = FakeApify([{"ticker": "0700.HK", "html": html, "scrapedAt": NOW.isoformat()}])
    rows = fetch_futu([settings.ticker("0700.HK")], settings, now=NOW, client=fake)
    assert len(rows) == 5
    actor_id, run_input = fake.inputs[0]
    assert actor_id == "apify/playwright-scraper"
    assert run_input["startUrls"][0]["url"].endswith("/stock/00700-HK")


def test_lihkg_keywords(settings):
    assert lihkg_keywords(settings.ticker("0700.HK")) == ["騰訊", "騰訊控股"]


def test_lihkg_rows(settings):
    items = load_sample("lihkg_items.json")
    rows = lihkg_rows(items, [settings.ticker("0700.HK")], scraped_at=NOW)
    assert len(rows) == len(items)
    assert rows[0]["post_id"] == f"lihkg:{items[0]['thread_id']}@0700.HK"
    assert rows[0]["created_at_utc"] == datetime(2026, 11, 20, 5, 29, 54, tzinfo=timezone.utc)
    assert rows[0]["likes"] == items[0]["like_count"]


def test_fetch_lihkg_filters_window(settings):
    items = load_sample("lihkg_items.json") + [{"thread_id": "1", "title": "old", "created_at": "2026-10-01T00:00:00Z",
                                                "source_query": "騰訊", "like_count": 0}]
    fake = FakeApify(items)
    rows = fetch_lihkg([settings.ticker("0700.HK")], settings, now=NOW, client=fake)
    assert all(r["created_at_utc"] >= NOW - timedelta(days=10) for r in rows)
    assert fake.inputs[0][1]["mode"] == "search"


def test_collect_falls_back_to_backup(settings):
    """HKEX sources are [futu, aastocks]; aastocks has no scraper -> backup lihkg is used."""
    calls = []

    def fake_source(name):
        def f(tickers, settings, **kw):
            calls.append((name, [t.symbol for t in tickers]))
            return [{"post_id": f"{name}:{t.symbol}", "exchange": t.exchange, "ticker": t.symbol, "source": name,
                     "created_at_utc": NOW - timedelta(hours=1), "text": f"{name} {t.symbol}", "likes": 0,
                     "reposts": 0, "views": 0, "author_followers": None, "url": "u", "sentiment_score": None,
                     "sentiment_label": None, "weight": None, "bar_date": None} for t in tickers]
        return f

    fetchers = {"futu": fake_source("futu"), "lihkg": fake_source("lihkg")}
    posts = collect_posts(settings, exchanges=["HKEX"], now=NOW, fetchers=fetchers)
    assert [c[0] for c in calls] == ["futu", "lihkg"]
    assert set(posts["source"]) == {"futu", "lihkg"}
    assert posts.groupby("ticker")["source"].nunique().min() == 2
