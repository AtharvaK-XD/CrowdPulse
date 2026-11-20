"""Daily price bars from yfinance -> prices.csv rows.

Keeps the last `price_history_days` trading bars per ticker (comfortably above the
contract's 90). A bar whose session has not closed yet (run during trading hours)
is dropped, so a half-finished day never leaks into features or checks.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

import pandas as pd

from crowdpulse.config import Settings, TickerConfig, load_config
from crowdpulse.features.calendar import session_close_utc
from crowdpulse.ingest.cache import load_raw, raw_path, save_raw
from crowdpulse.schema import PRICES_COLUMNS

logger = logging.getLogger(__name__)

SOURCE = "yfinance"
Downloader = Callable[[str, date, date], pd.DataFrame]


def _default_download(symbol: str, start: date, end: date) -> pd.DataFrame:
    import yfinance as yf  # lazy: tests inject a fake downloader

    df = yf.download(symbol, start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
                     interval="1d", auto_adjust=False, progress=False, threads=False)
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance >= 0.2.50 returns (Price, Ticker) columns even for one symbol.
        level = 1 if symbol in df.columns.get_level_values(-1) else 0
        df = df.xs(symbol, axis=1, level=-1) if level == 1 else df.droplevel(0, axis=1)
    return df


def _frame_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = df.reset_index()
    first = out.columns[0]
    out = out.rename(columns={first: "date"})
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.columns = [str(c).lower().replace(" ", "_") for c in out.columns]
    return out.to_dict(orient="records")


def normalise_bars(records: list[dict[str, Any]], ticker: TickerConfig, settings: Settings, *,
                   now: datetime) -> pd.DataFrame:
    """Raw yfinance records -> prices.csv rows: closed sessions only, last N bars, no NaN closes."""
    ex = settings.exchanges[ticker.exchange]
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=list(PRICES_COLUMNS))
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]
    closed = df["date"].map(lambda d: session_close_utc(d, ex) <= now)
    dropped = int((~closed).sum())
    if dropped:
        logger.info("%s: dropping %d bar(s) whose session has not closed yet", ticker.symbol, dropped)
    df = df[closed].drop_duplicates(subset="date", keep="last").sort_values("date")
    df = df.tail(settings.price_history_days)
    out = pd.DataFrame({
        "ticker": ticker.symbol,
        "date": df["date"].map(lambda d: d.isoformat()),
        "open": df["open"].astype(float),
        "high": df["high"].astype(float),
        "low": df["low"].astype(float),
        "close": df["close"].astype(float),
        "volume": pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64"),
        "currency": ex.currency,
    })
    return out[list(PRICES_COLUMNS)].reset_index(drop=True)


def fetch_prices(
    settings: Settings,
    *,
    tickers: list[str] | None = None,
    use_cache: bool = False,
    now: datetime | None = None,
    download: Downloader = _default_download,
) -> pd.DataFrame:
    """prices.csv rows for the selected tickers (all by default)."""
    now = now or datetime.now(timezone.utc)
    # ~1.6 calendar days per trading day, plus holidays margin.
    start = now.date() - timedelta(days=int(settings.price_history_days * 1.6) + 20)
    frames = []
    for t in settings.tickers:
        if tickers and t.symbol not in tickers:
            continue
        path = raw_path(settings.data_dir, SOURCE, t.symbol, now.date())
        params = {"start": start.isoformat(), "history_days": settings.price_history_days}
        cached = load_raw(path, params=params) if use_cache else None
        if cached is not None:
            records = cached["items"]
        else:
            try:
                records = _frame_to_records(download(t.symbol, start, now.date()))
            except Exception as exc:  # noqa: BLE001 - one ticker failing must not stop the run
                logger.error("Price download failed for %s: %s", t.symbol, exc)
                continue
            save_raw(path, records, source=SOURCE, key=t.symbol, params=params)
        bars = normalise_bars(records, t, settings, now=now)
        if len(bars) < 90:
            logger.warning("%s: only %d trading bars (contract wants >= 90)", t.symbol, len(bars))
        frames.append(bars)
    if not frames:
        return pd.DataFrame(columns=list(PRICES_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download daily prices.")
    parser.add_argument("--ticker", action="append")
    parser.add_argument("--cache", action="store_true")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    prices = fetch_prices(load_config(), tickers=args.ticker, use_cache=args.cache)
    summary = prices.groupby("ticker").agg(bars=("date", "size"), first=("date", "min"), last=("date", "max"),
                                           last_close=("close", "last"), currency=("currency", "first"))
    print(summary.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
