"""Shared helpers for social scrapers: text cleaning, local->UTC time parsing, final filtering.

Every scraper returns plain dict rows with the posts.csv columns (sentiment columns
empty). finalize_posts() turns them into a clean DataFrame.
"""

from __future__ import annotations

import html
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from crowdpulse.schema import POSTS_COLUMNS

logger = logging.getLogger(__name__)

SPAM_REPEAT_THRESHOLD = 5  # same text this many times (per ticker) -> all copies dropped
# Contract: "0 for counts". Follower counts are counts too, so a site that doesn't
# show them (most boards are anonymous) gives 0.
COUNT_COLUMNS = ("likes", "reposts", "views", "author_followers")

_TAG_RE = re.compile(r"<[^>]+>")
_BR_RE = re.compile(r"<\s*br\s*/?\s*>|</\s*p\s*>", re.I)
_WS_RE = re.compile(r"\s+")


def clean_text(value: Any) -> str:
    """Strip HTML tags and entities, collapse whitespace."""
    if value is None:
        return ""
    s = str(value)
    s = _BR_RE.sub(" ", s)
    s = _TAG_RE.sub(" ", s)
    s = html.unescape(s)
    s = s.replace("​", "").replace("\xa0", " ")
    return _WS_RE.sub(" ", s).strip()


def join_title_body(title: Any, body: Any) -> str:
    """Title + body, without repeating the body when it starts with the title."""
    t, b = clean_text(title), clean_text(body)
    if not b:
        return t
    if not t or b.startswith(t):
        return b
    return f"{t} {b}"


# ---------------------------------------------------------------- time parsing

_RELATIVE = [
    # (regex, unit) — Korean, Chinese (trad/simplified), English
    (re.compile(r"(\d+)\s*(?:초|秒)\s*(?:전|前)"), "seconds"),
    (re.compile(r"(\d+)\s*(?:분|分鐘|分钟|分)\s*(?:전|前)"), "minutes"),
    (re.compile(r"(\d+)\s*(?:시간|小時|小时)\s*(?:전|前)"), "hours"),
    (re.compile(r"(\d+)\s*(?:일|天|日)\s*(?:전|前)"), "days"),
    (re.compile(r"(\d+)\s*(?:sec|second)s?\s+ago", re.I), "seconds"),
    (re.compile(r"(\d+)\s*(?:min|minute)s?\s+ago", re.I), "minutes"),
    (re.compile(r"(\d+)\s*(?:hr|hour)s?\s+ago", re.I), "hours"),
    (re.compile(r"(\d+)\s*days?\s+ago", re.I), "days"),
]
_JUST_NOW = re.compile(r"^(방금|방금\s*전|剛剛|刚刚|just now)$", re.I)
_YESTERDAY = re.compile(r"^(?:어제|昨天|昨日|yesterday)\s*(\d{1,2}):(\d{2})$", re.I)
_DAY_BEFORE = re.compile(r"^(?:그제|前天)\s*(\d{1,2}):(\d{2})$")
_ABS_FORMATS = (
    "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
    "%Y.%m.%d %H:%M:%S", "%Y.%m.%d %H:%M", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M",
    "%Y-%m-%d", "%Y.%m.%d", "%Y/%m/%d",
)
_MONTH_DAY_TIME = re.compile(r"^(\d{1,2})[/.-](\d{1,2})\s+(\d{1,2}):(\d{2})$")
_MONTH_DAY = re.compile(r"^(\d{1,2})[/.-](\d{1,2})$")
_TIME_ONLY = re.compile(r"^(\d{1,2}):(\d{2})$")


def parse_local_time(value: Any, tz: ZoneInfo, scraped_at: datetime) -> datetime | None:
    """Parse a site timestamp shown in local time (or relative) into an aware UTC datetime.

    Handles ISO/dotted absolute times, 'MM/DD HH:MM' (year inferred, never in the
    future), bare 'HH:MM' (today, local), and relative times like '5분 전', '3小時前',
    '방금', '昨天 14:05'. Strings with an explicit offset/Z are honoured. Returns None
    when the value can't be parsed — the caller drops the row rather than guessing.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=tz)
        return dt.astimezone(timezone.utc)
    s = _WS_RE.sub(" ", str(value)).strip()
    if not s:
        return None
    scraped_utc = scraped_at.astimezone(timezone.utc)
    local_now = scraped_utc.astimezone(tz)

    if _JUST_NOW.match(s):
        return scraped_utc
    for rx, unit in _RELATIVE:
        m = rx.search(s)
        if m:
            return scraped_utc - timedelta(**{unit: int(m.group(1))})
    for rx, days in ((_YESTERDAY, 1), (_DAY_BEFORE, 2)):
        m = rx.match(s)
        if m:
            d = (local_now - timedelta(days=days)).date()
            return datetime(d.year, d.month, d.day, int(m.group(1)), int(m.group(2)),
                            tzinfo=tz).astimezone(timezone.utc)

    # Explicit offset or Z: trust it.
    if re.search(r"(Z|[+-]\d{2}:?\d{2})$", s) and "T" in s:
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
        except ValueError:
            pass
    for fmt in _ABS_FORMATS:
        try:
            naive = datetime.strptime(s, fmt)
        except ValueError:
            continue
        return naive.replace(tzinfo=tz).astimezone(timezone.utc)

    m = _MONTH_DAY_TIME.match(s) or _MONTH_DAY.match(s)
    if m:
        month, day = int(m.group(1)), int(m.group(2))
        hour = int(m.group(3)) if m.lastindex and m.lastindex >= 3 else 0
        minute = int(m.group(4)) if m.lastindex and m.lastindex >= 4 else 0
        for year in (local_now.year, local_now.year - 1):
            try:
                cand = datetime(year, month, day, hour, minute, tzinfo=tz)
            except ValueError:
                return None
            if cand <= local_now + timedelta(minutes=5):
                return cand.astimezone(timezone.utc)
        return None
    m = _TIME_ONLY.match(s)
    if m:
        cand = local_now.replace(hour=int(m.group(1)), minute=int(m.group(2)), second=0, microsecond=0)
        if cand > local_now + timedelta(minutes=5):
            cand -= timedelta(days=1)
        return cand.astimezone(timezone.utc)
    return None


def to_int(value: Any) -> int | None:
    """Parse counts like '1,234', '1.2萬', '3.4K', '5' -> int. None if not a number."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return None if pd.isna(value) else int(value)
    s = str(value).strip().replace(",", "")
    if not s:
        return None
    mult = 1
    for suffix, m in (("萬", 10_000), ("万", 10_000), ("億", 100_000_000), ("亿", 100_000_000),
                      ("만", 10_000), ("천", 1_000), ("K", 1_000), ("k", 1_000), ("M", 1_000_000)):
        if s.endswith(suffix):
            mult, s = m, s[: -len(suffix)]
            break
    try:
        return int(round(float(s) * mult))
    except ValueError:
        return None


def make_row(
    *, source: str, native_id: Any, exchange: str, ticker: str, created_at_utc: datetime | None,
    text: str, url: str | None, likes: Any = None, reposts: Any = None, views: Any = None,
    author_followers: Any = None,
) -> dict[str, Any]:
    """One posts.csv row. post_id is '<source>:<native id>' so it is unique across sources."""
    return {
        "post_id": f"{source}:{native_id}",
        "exchange": exchange,
        "ticker": ticker,
        "source": source,
        "created_at_utc": created_at_utc,
        "text": text,
        "likes": to_int(likes),
        "reposts": to_int(reposts),
        "views": to_int(views),
        "author_followers": to_int(author_followers),
        "url": url,
        "sentiment_score": None,
        "sentiment_label": None,
        "weight": None,
        "bar_date": None,
    }


def empty_posts() -> pd.DataFrame:
    return finalize_posts([], now=datetime.now(timezone.utc), lookback_days=1)


def finalize_posts(rows: Iterable[dict[str, Any]], *, now: datetime, lookback_days: int) -> pd.DataFrame:
    """Rows -> clean posts DataFrame.

    Drops: empty text, missing/unparseable time, future timestamps, posts older than
    the window, duplicate post_ids, and spam (identical text 5+ times per ticker).
    Counts (likes/reposts/views/author_followers) missing -> 0.
    """
    df = pd.DataFrame(list(rows), columns=list(POSTS_COLUMNS))
    n0 = len(df)
    start = now - timedelta(days=lookback_days)
    stats: dict[str, int] = {}

    df["text"] = df["text"].fillna("").astype(str).map(clean_text)
    mask = df["text"].str.len() > 0
    stats["empty"] = int((~mask).sum())
    df = df[mask]

    df["created_at_utc"] = pd.to_datetime(df["created_at_utc"], utc=True, errors="coerce")
    mask = df["created_at_utc"].notna()
    stats["no_time"] = int((~mask).sum())
    df = df[mask]
    mask = df["created_at_utc"] <= pd.Timestamp(now)
    stats["future"] = int((~mask).sum())
    df = df[mask]
    mask = df["created_at_utc"] >= pd.Timestamp(start)
    stats["too_old"] = int((~mask).sum())
    df = df[mask]

    before = len(df)
    df = df.drop_duplicates(subset="post_id", keep="first")
    stats["duplicate_id"] = before - len(df)

    norm = df["text"].str.lower().str.replace(r"\s+", "", regex=True)
    counts = norm.groupby([df["ticker"], norm]).transform("size")
    mask = counts < SPAM_REPEAT_THRESHOLD
    stats["spam"] = int((~mask).sum())
    df = df[mask]
    # Exact repeats below the spam threshold (same author re-posting): keep the first.
    before = len(df)
    df = df.loc[~df.assign(_n=norm).duplicated(subset=["ticker", "source", "_n"], keep="first")]
    stats["duplicate_text"] = before - len(df)

    for col in COUNT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce").astype("float64")
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").astype("float64")

    dropped = {k: v for k, v in stats.items() if v}
    if dropped:
        logger.info("Posts: kept %d of %d; dropped %s", len(df), n0, dropped)
    return df.sort_values(["ticker", "created_at_utc"], ascending=[True, False], kind="stable").reset_index(drop=True)
