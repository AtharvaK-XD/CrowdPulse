"""Column lists for the output files defined in docs/CONTRACT.md. Keep in sync with it."""

from __future__ import annotations

POSTS_COLUMNS: tuple[str, ...] = (
    "post_id", "exchange", "ticker", "source", "created_at_utc", "text",
    "likes", "reposts", "views", "author_followers", "url",
    "sentiment_score", "sentiment_label", "weight", "bar_date",
)

NEWS_COLUMNS: tuple[str, ...] = (
    "ticker", "title", "url", "published_at_utc", "source_domain", "summary",
    "sentiment_score", "bar_date",
)

PRICES_COLUMNS: tuple[str, ...] = (
    "ticker", "date", "open", "high", "low", "close", "volume", "currency",
)

DAILY_COLUMNS: tuple[str, ...] = (
    "ticker", "date", "mentions", "social_sent_mean", "social_sent_weighted",
    "bullish_ratio", "news_count", "news_sent_mean", "close", "return_1d",
)

PREDICTIONS_LOG_COLUMNS: tuple[str, ...] = (
    "run_id", "ticker", "target_date", "predicted_close", "actual_close",
    "error", "error_pct", "direction_hit",
)
