from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from crowdpulse.ingest.cache import raw_path
from crowdpulse.ingest.news_exa import (
    NewsFetchError,
    build_queries,
    fetch_news,
    normalise_news,
)
from crowdpulse.schema import NEWS_COLUMNS

from .conftest import load_sample

NOW = datetime(2026, 11, 20, 9, 0, tzinfo=timezone.utc)


class FakeExa:
    """Returns saved results per query; can fail a query N times first."""

    def __init__(self, by_query, fail_times=None):
        self.by_query = by_query
        self.fail_times = dict(fail_times or {})
        self.calls: list[dict] = []

    def search(self, query, **kwargs):
        self.calls.append({"query": query, **kwargs})
        if self.fail_times.get(query, 0) > 0:
            self.fail_times[query] -= 1
            raise ConnectionError("boom")
        return {"results": self.by_query.get(query, [])}


@pytest.fixture
def sample():
    data = load_sample("exa_0700HK.json")
    data.pop("_note")
    return data


def test_build_queries(settings):
    q = build_queries(settings.ticker("0700.HK"))
    assert q == [("en", "Tencent 0700.HK stock"), ("local", "騰訊 股價"), ("local", "騰訊控股")]
    kr = build_queries(settings.ticker("005930.KS"))
    assert ("local", "삼성전자 주가") in kr


def test_build_queries_fallback_when_no_english(settings):
    from dataclasses import replace

    t = replace(settings.ticker("0700.HK"), keywords_en=())
    assert build_queries(t)[0] == ("en", "Tencent 0700.HK stock")


def test_fetch_passes_window_and_limits(settings, sample):
    fake = FakeExa(sample)
    items = fetch_news("0700.HK", settings=settings, client=fake, now=NOW, sleep=lambda s: None)
    assert len(fake.calls) == 3
    call = fake.calls[0]
    assert call["start_published_date"] == "2026-11-10T09:00:00.000Z"
    assert call["end_published_date"] == "2026-11-20T09:00:00.000Z"
    assert call["num_results"] == settings.news.results_per_query
    assert call["contents"]["summary"] is True
    assert call["contents"]["text"] == {"max_characters": settings.news.max_text_chars}
    assert len(items) == 7
    assert {i["query_lang"] for i in items} == {"en", "local"}
    assert raw_path(settings.data_dir, "exa", "0700.HK", NOW.date()).is_file()


def test_cache_reuses_today(settings, sample):
    fetch_news("0700.HK", settings=settings, client=FakeExa(sample), now=NOW, sleep=lambda s: None)
    fake2 = FakeExa({})
    items = fetch_news("0700.HK", settings=settings, client=fake2, now=NOW, use_cache=True)
    assert len(items) == 7
    assert fake2.calls == []


def test_partial_failure_saved_incomplete(settings, sample):
    fake = FakeExa(sample, fail_times={"騰訊 股價": 99})
    items = fetch_news("0700.HK", settings=settings, client=fake, now=NOW, sleep=lambda s: None)
    assert len(items) == 4  # only the English query's results
    # Incomplete run must not be reused by --cache.
    fake2 = FakeExa(sample)
    fetch_news("0700.HK", settings=settings, client=fake2, now=NOW, use_cache=True, sleep=lambda s: None)
    assert len(fake2.calls) == 3


def test_retry_recovers(settings, sample):
    sleeps: list[float] = []
    fake = FakeExa(sample, fail_times={"Tencent 0700.HK stock": 1})
    items = fetch_news("0700.HK", settings=settings, client=fake, now=NOW, sleep=sleeps.append)
    assert len(items) == 7
    assert sleeps == [2.0]


def test_all_queries_fail_raises(settings, sample):
    fake = FakeExa(sample, fail_times={q: 99 for q in sample})
    with pytest.raises(NewsFetchError):
        fetch_news("0700.HK", settings=settings, client=fake, now=NOW, sleep=lambda s: None)


def test_normalise_filters_and_dedupes(settings, sample):
    items = fetch_news("0700.HK", settings=settings, client=FakeExa(sample), now=NOW, sleep=lambda s: None)
    df = normalise_news(items, "0700.HK", now=NOW, lookback_days=10)
    assert list(df.columns) == list(NEWS_COLUMNS)
    # Kept: a1, a2, b2. Dropped: old (a3), undated (a4), duplicate (b1), future (b3).
    assert len(df) == 3
    assert df["url"].str.contains("future|example-old|nodate").sum() == 0
    assert (df["published_at_utc"] >= pd.Timestamp("2026-11-10T09:00Z")).all()
    assert (df["published_at_utc"] <= pd.Timestamp(NOW)).all()
    # Newest first.
    assert df["published_at_utc"].is_monotonic_decreasing
    assert df.loc[0, "title"].startswith("騰訊股價突破")


def test_normalise_never_invents_values(settings, sample):
    items = fetch_news("0700.HK", settings=settings, client=FakeExa(sample), now=NOW, sleep=lambda s: None)
    df = normalise_news(items, "0700.HK", now=NOW, lookback_days=10)
    aastocks = df[df["source_domain"] == "aastocks.com"].iloc[0]
    assert pd.isna(aastocks["summary"])  # empty summary stays null, not copied from text
    assert df["sentiment_score"].isna().all()
    assert df["bar_date"].isna().all()
    assert set(df["source_domain"]) == {"reuters.com", "scmp.com", "aastocks.com"}
    assert str(df["published_at_utc"].dt.tz) == "UTC"


def test_normalise_empty():
    df = normalise_news([], "0700.HK", now=NOW, lookback_days=10)
    assert df.empty
    assert list(df.columns) == list(NEWS_COLUMNS)


def test_normalise_include_text(settings, sample):
    items = fetch_news("0700.HK", settings=settings, client=FakeExa(sample), now=NOW, sleep=lambda s: None)
    df = normalise_news(items, "0700.HK", now=NOW, lookback_days=10, include_text=True)
    assert df.columns[-1] == "text"
    assert df["text"].notna().all()


def test_object_style_results(settings):
    """exa-py returns Result objects, not dicts."""

    class R:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class Resp:
        results = [R(id="x", url="https://a.com/1", title="T", published_date="2026-11-19",
                     author=None, score=0.5, text="t", summary="s")]

    class ObjExa:
        def search(self, query, **kw):
            return Resp()

    items = fetch_news("0700.HK", settings=settings, client=ObjExa(), now=NOW)
    assert items[0]["url"] == "https://a.com/1"
    df = normalise_news(items, "0700.HK", now=NOW, lookback_days=10)
    assert len(df) == 1
