from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from crowdpulse.ingest.social_common import (
    clean_text, finalize_posts, join_title_body, make_row, parse_local_time, to_int,
)

KST, HKT = ZoneInfo("Asia/Seoul"), ZoneInfo("Asia/Hong_Kong")
SCRAPED = datetime(2026, 11, 20, 8, 0, tzinfo=timezone.utc)  # 17:00 KST, 16:00 HKT


def test_clean_text_strips_html():
    assert clean_text("a<br>b &amp; <b>c</b>\n\n d") == "a b & c d"
    assert clean_text(None) == ""


def test_join_title_body():
    assert join_title_body("제목", "본문") == "제목 본문"
    assert join_title_body("제목", "제목 본문 더") == "제목 본문 더"
    assert join_title_body("", "본문") == "본문"


def test_kst_absolute_to_utc():
    assert parse_local_time("2026-11-20T16:57:59", KST, SCRAPED) == datetime(2026, 11, 20, 7, 57, 59, tzinfo=timezone.utc)
    assert parse_local_time("2026-11-20 14:37:48", KST, SCRAPED) == datetime(2026, 11, 20, 5, 37, 48, tzinfo=timezone.utc)
    assert parse_local_time("2026.11.19 14:05", KST, SCRAPED) == datetime(2026, 11, 19, 5, 5, tzinfo=timezone.utc)


def test_hkt_absolute_to_utc():
    assert parse_local_time("2026/11/19 09:30", HKT, SCRAPED) == datetime(2026, 11, 19, 1, 30, tzinfo=timezone.utc)


def test_relative_korean_and_chinese():
    assert parse_local_time("5분 전", KST, SCRAPED) == SCRAPED - timedelta(minutes=5)
    assert parse_local_time("2시간 전", KST, SCRAPED) == SCRAPED - timedelta(hours=2)
    assert parse_local_time("3小時前", HKT, SCRAPED) == SCRAPED - timedelta(hours=3)
    assert parse_local_time("10分鐘前", HKT, SCRAPED) == SCRAPED - timedelta(minutes=10)
    assert parse_local_time("2天前", HKT, SCRAPED) == SCRAPED - timedelta(days=2)
    assert parse_local_time("방금", KST, SCRAPED) == SCRAPED
    assert parse_local_time("剛剛", HKT, SCRAPED) == SCRAPED


def test_yesterday_and_month_day():
    # 昨天 14:05 HKT on 2026-11-19 -> 06:05 UTC
    assert parse_local_time("昨天 14:05", HKT, SCRAPED) == datetime(2026, 11, 19, 6, 5, tzinfo=timezone.utc)
    assert parse_local_time("11/18 14:05", HKT, SCRAPED) == datetime(2026, 11, 18, 6, 5, tzinfo=timezone.utc)
    # A month/day later than today belongs to last year, never the future.
    assert parse_local_time("12/25 10:00", HKT, SCRAPED).year == 2025


def test_time_only_is_today_or_yesterday():
    assert parse_local_time("15:09", HKT, SCRAPED) == datetime(2026, 11, 20, 7, 9, tzinfo=timezone.utc)
    assert parse_local_time("23:00", HKT, SCRAPED) == datetime(2026, 11, 19, 15, 0, tzinfo=timezone.utc)


def test_explicit_utc_kept():
    assert parse_local_time("2026-11-20T05:29:54.000Z", HKT, SCRAPED) == datetime(2026, 11, 20, 5, 29, 54, tzinfo=timezone.utc)


def test_unparseable_is_none():
    assert parse_local_time("someday", KST, SCRAPED) is None
    assert parse_local_time("", KST, SCRAPED) is None


def test_to_int():
    assert to_int("1,234") == 1234
    assert to_int("1.2萬") == 12000
    assert to_int("3K") == 3000
    assert to_int(" 5 ") == 5
    assert to_int("abc") is None
    assert to_int(None) is None


def _row(i, text, ts, ticker="005930.KS"):
    return make_row(source="naver", native_id=i, exchange="KRX", ticker=ticker, created_at_utc=ts,
                    text=text, url="u", likes=None, views="10")


def test_finalize_filters():
    now = SCRAPED
    rows = [
        _row(1, "좋다", now - timedelta(hours=1)),
        _row(1, "좋다 dup id", now - timedelta(hours=1)),        # duplicate id
        _row(2, "  ", now - timedelta(hours=1)),                 # empty
        _row(3, "future", now + timedelta(hours=1)),             # future
        _row(4, "old", now - timedelta(days=11)),                # outside 10 days
        _row(5, "no time", None),                                # unparseable time
        *[_row(100 + k, "광고 스팸 링크", now - timedelta(hours=k)) for k in range(5)],  # spam x5
        _row(200, "same text", now - timedelta(hours=2)),
        _row(201, "same  text", now - timedelta(hours=3)),     # repeat below spam threshold -> keep one
    ]
    df = finalize_posts(rows, now=now, lookback_days=10)
    assert sorted(df["post_id"]) == ["naver:1", "naver:200"]
    assert df["likes"].tolist() == [0, 0]            # missing count -> 0
    assert df["author_followers"].tolist() == [0, 0]
    assert (df["views"] == 10).all()
    assert str(df["created_at_utc"].dt.tz) == "UTC"


def test_finalize_empty():
    df = finalize_posts([], now=SCRAPED, lookback_days=10)
    assert df.empty and "post_id" in df.columns
