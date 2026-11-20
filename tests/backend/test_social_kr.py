from __future__ import annotations

from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlsplit

from crowdpulse.ingest.social_kr import (
    KST, crawl_daum, crawl_naver, daum_rows, fetch_daum, fetch_naver, naver_rows,
)

from .conftest import load_sample

NOW = datetime(2026, 11, 20, 8, 30, tzinfo=timezone.utc)


def test_naver_rows_from_sample(settings):
    raw = load_sample("naver_page.json")["result"]["posts"]
    rows = naver_rows(raw, settings.ticker("005930.KS"), scraped_at=NOW)
    assert len(rows) == len(raw)
    r = rows[0]
    assert r["post_id"] == f"naver:{raw[0]['id']}"
    assert r["source"] == "naver" and r["exchange"] == "KRX"
    # writtenAt is KST -> UTC is 9 hours earlier
    local = datetime.fromisoformat(raw[0]["writtenAt"]).replace(tzinfo=KST)
    assert r["created_at_utc"] == local.astimezone(timezone.utc)
    assert r["url"].endswith(f"/005930/discussion/{raw[0]['id']}")
    assert r["text"] and "<br>" not in r["text"]


def test_daum_rows_from_sample(settings):
    raw = load_sample("daum_page.json")["data"]["posts"]
    rows = daum_rows(raw, settings.ticker("005930.KS"), scraped_at=NOW)
    assert len(rows) == len(raw)
    assert rows[0]["likes"] == raw[0]["agreeCount"]
    assert rows[0]["views"] == raw[0]["readCount"]
    assert rows[0]["url"].endswith(f"/005930/talks/{raw[0]['id']}")


class FakeNaver:
    """Global id sequence: one post every 10 minutes for this stock, ids step by 600."""

    def __init__(self, now: datetime, days: int = 12, every_min: int = 10):
        self.posts = []
        t, pid = now, 10_000_000
        while t > now - timedelta(days=days):
            self.posts.append({"id": str(pid), "writtenAt": t.astimezone(KST).strftime("%Y-%m-%dT%H:%M:%S"),
                               "title": "t", "contentSwReplacedButImg": "c", "recommendCount": 0, "viewCount": 1})
            t -= timedelta(minutes=every_min)
            pid -= 600  # ~1 id/sec globally
        self.calls = 0

    def __call__(self, url: str):
        self.calls += 1
        q = parse_qs(urlsplit(url).query)
        size = int(q["pageSize"][0])
        offset = int(q["offset"][0]) if "offset" in q else None
        older = [p for p in self.posts if offset is None or int(p["id"]) < -offset]
        page = older[:size]
        return {"result": {"posts": page, "lastOffset": str(-int(page[-1]["id"])) if page else None}}


def test_crawl_naver_reaches_window_start():
    fake = FakeNaver(NOW)
    start = NOW - timedelta(days=10)
    raw = crawl_naver("005930", start_utc=start, now_utc=NOW, get_json=fake, per_day_cap=10_000)
    times = [datetime.fromisoformat(p["writtenAt"]).replace(tzinfo=KST) for p in raw]
    assert min(times) >= start
    assert min(times) - start < timedelta(minutes=10)
    assert len(raw) == len({p["id"] for p in raw})


def test_crawl_naver_per_day_cap_covers_all_days():
    fake = FakeNaver(NOW)
    start = NOW - timedelta(days=10)
    raw = crawl_naver("005930", start_utc=start, now_utc=NOW, get_json=fake, per_day_cap=20)
    days = {}
    for p in raw:
        d = datetime.fromisoformat(p["writtenAt"]).date()
        days[d] = days.get(d, 0) + 1
    assert max(days.values()) <= 20
    assert len(days) >= 10          # every day in the window got sampled
    assert fake.calls < 60          # jumping avoided paging through every post


def test_crawl_daum_stops_at_window():
    posts = [{"id": i, "createdAt": (NOW - timedelta(hours=6 * i)).astimezone(KST).strftime("%Y-%m-%d %H:%M:%S"),
              "title": "t", "content": "c"} for i in range(80)]

    def get_json(url):
        q = parse_qs(urlsplit(url).query)
        page, per = int(q["page"][0]), int(q["perPage"][0])
        chunk = posts[(page - 1) * per: page * per]
        return {"data": {"posts": chunk}, "totalPages": 8}

    raw = crawl_daum("005930", start_utc=NOW - timedelta(days=10), now_utc=NOW, get_json=get_json, per_page=10)
    assert len(raw) == 41  # hours 0..240 step 6
    assert all(datetime.strptime(p["createdAt"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST) >= NOW - timedelta(days=10)
               for p in raw)


def test_fetch_naver_uses_cache(settings):
    fake = FakeNaver(NOW, days=2)
    t = [settings.ticker("035720.KS")]
    rows1 = fetch_naver(t, settings, now=NOW, get_json=fake, use_cache=False)
    calls = fake.calls
    rows2 = fetch_naver(t, settings, now=NOW, get_json=fake, use_cache=True)
    assert fake.calls == calls
    assert len(rows1) == len(rows2) > 0


def test_fetch_daum_survives_errors(settings):
    def broken(url):
        raise RuntimeError("down")

    assert fetch_daum([settings.ticker("035720.KS")], settings, now=NOW, get_json=broken) == []
