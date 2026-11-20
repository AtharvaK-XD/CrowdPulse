"""Typed settings loaded from config.yaml, plus secret lookup from .env."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_ENV_FILE = REPO_ROOT / ".env"

_SYMBOL_RE = re.compile(r"^[0-9A-Z]+\.[A-Z]{2}$")


class ConfigError(ValueError):
    """config.yaml is missing, malformed or inconsistent."""


class MissingSecretError(RuntimeError):
    """A required key is not set in the environment or .env."""


@dataclass(frozen=True)
class ExchangeConfig:
    code: str
    timezone: str
    close_utc: time
    currency: str
    sources: tuple[str, ...]
    backup_sources: tuple[str, ...] = ()

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)


@dataclass(frozen=True)
class TickerConfig:
    symbol: str
    exchange: str
    name: str
    keywords_en: tuple[str, ...] = ()
    keywords_local: tuple[str, ...] = ()
    cashtags: tuple[str, ...] = ()

    @property
    def code(self) -> str:
        """Exchange-local code without suffix, e.g. '0700' or '005930'."""
        return self.symbol.split(".")[0]


@dataclass(frozen=True)
class NewsConfig:
    results_per_query: int = 15
    max_text_chars: int = 2000


@dataclass(frozen=True)
class ApifyConfig:
    max_retries: int = 3
    timeout_secs: int = 900
    use_residential_proxy: bool = False


@dataclass(frozen=True)
class Settings:
    exchanges: dict[str, ExchangeConfig]
    tickers: tuple[TickerConfig, ...]
    lookback_days: int = 10
    price_history_days: int = 120
    news: NewsConfig = field(default_factory=NewsConfig)
    apify: ApifyConfig = field(default_factory=ApifyConfig)
    data_dir: Path = DEFAULT_DATA_DIR

    def ticker(self, symbol: str) -> TickerConfig:
        for t in self.tickers:
            if t.symbol == symbol:
                return t
        raise KeyError(f"Unknown ticker {symbol!r}; known: {[t.symbol for t in self.tickers]}")

    def tickers_for(self, exchange: str) -> tuple[TickerConfig, ...]:
        return tuple(t for t in self.tickers if t.exchange == exchange)

    def exchange_of(self, symbol: str) -> ExchangeConfig:
        return self.exchanges[self.ticker(symbol).exchange]


# ---------------------------------------------------------------- parsing helpers


def _require(mapping: dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping or mapping[key] is None:
        raise ConfigError(f"{where}: missing required key {key!r}")
    return mapping[key]


def _str_tuple(value: Any, where: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ConfigError(f"{where}: expected a list of strings, got {value!r}")
    return tuple(value)


def _parse_close(value: Any, where: str) -> time:
    # YAML 1.1 may turn an unquoted 8:00 into the integer 480 (minutes).
    if isinstance(value, int) and not isinstance(value, bool):
        hours, minutes = divmod(value, 60)
    elif isinstance(value, str) and re.fullmatch(r"\d{1,2}:\d{2}", value.strip()):
        hours, minutes = (int(p) for p in value.strip().split(":"))
    else:
        raise ConfigError(f"{where}: close_utc must look like 'HH:MM', got {value!r}")
    if not (0 <= hours < 24 and 0 <= minutes < 60):
        raise ConfigError(f"{where}: close_utc out of range: {value!r}")
    return time(hours, minutes)


def _positive_int(value: Any, where: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{where}: expected a positive integer, got {value!r}")
    return value


def _parse_exchange(code: str, raw: Any) -> ExchangeConfig:
    where = f"exchanges.{code}"
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: expected a mapping")
    tz = str(_require(raw, "timezone", where))
    try:
        ZoneInfo(tz)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ConfigError(f"{where}: unknown timezone {tz!r}") from exc
    sources = _str_tuple(_require(raw, "sources", where), f"{where}.sources")
    if not sources:
        raise ConfigError(f"{where}.sources: at least one source is required")
    return ExchangeConfig(
        code=code,
        timezone=tz,
        close_utc=_parse_close(_require(raw, "close_utc", where), where),
        currency=str(_require(raw, "currency", where)),
        sources=sources,
        backup_sources=_str_tuple(raw.get("backup_sources"), f"{where}.backup_sources"),
    )


def _parse_ticker(i: int, raw: Any, exchanges: dict[str, ExchangeConfig]) -> TickerConfig:
    where = f"tickers[{i}]"
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: expected a mapping")
    symbol = str(_require(raw, "symbol", where))
    if not _SYMBOL_RE.match(symbol):
        raise ConfigError(f"{where}: symbol {symbol!r} should look like '0700.HK'")
    exchange = str(_require(raw, "exchange", where))
    if exchange not in exchanges:
        raise ConfigError(f"{where}: exchange {exchange!r} is not defined under 'exchanges'")
    return TickerConfig(
        symbol=symbol,
        exchange=exchange,
        name=str(_require(raw, "name", where)),
        keywords_en=_str_tuple(raw.get("keywords_en"), f"{where}.keywords_en"),
        keywords_local=_str_tuple(raw.get("keywords_local"), f"{where}.keywords_local"),
        cashtags=_str_tuple(raw.get("cashtags"), f"{where}.cashtags"),
    )


def parse_config(raw: Any, *, data_dir: Path | None = None) -> Settings:
    """Build Settings from an already-parsed YAML document."""
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")

    raw_ex = _require(raw, "exchanges", "config")
    if not isinstance(raw_ex, dict) or not raw_ex:
        raise ConfigError("exchanges: expected a non-empty mapping")
    exchanges = {str(code): _parse_exchange(str(code), body) for code, body in raw_ex.items()}

    raw_tk = _require(raw, "tickers", "config")
    if not isinstance(raw_tk, list) or not raw_tk:
        raise ConfigError("tickers: expected a non-empty list")
    tickers = tuple(_parse_ticker(i, t, exchanges) for i, t in enumerate(raw_tk))
    seen: set[str] = set()
    for t in tickers:
        if t.symbol in seen:
            raise ConfigError(f"tickers: duplicate symbol {t.symbol!r}")
        seen.add(t.symbol)

    news_raw = raw.get("news") or {}
    news = NewsConfig(
        results_per_query=_positive_int(news_raw.get("results_per_query", 15), "news.results_per_query"),
        max_text_chars=_positive_int(news_raw.get("max_text_chars", 2000), "news.max_text_chars"),
    )
    if news.results_per_query > 100:
        raise ConfigError("news.results_per_query: Exa allows at most 100")

    apify_raw = raw.get("apify") or {}
    max_retries = apify_raw.get("max_retries", 3)
    if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 0:
        raise ConfigError(f"apify.max_retries: expected an integer >= 0, got {max_retries!r}")
    apify = ApifyConfig(
        max_retries=max_retries,
        timeout_secs=_positive_int(apify_raw.get("timeout_secs", 900), "apify.timeout_secs"),
        use_residential_proxy=bool(apify_raw.get("use_residential_proxy", False)),
    )

    return Settings(
        exchanges=exchanges,
        tickers=tickers,
        lookback_days=_positive_int(raw.get("lookback_days", 10), "lookback_days"),
        price_history_days=_positive_int(raw.get("price_history_days", 120), "price_history_days"),
        news=news,
        apify=apify,
        data_dir=Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR,
    )


def load_config(path: str | Path | None = None, *, data_dir: str | Path | None = None) -> Settings:
    """Read config.yaml (repo root by default) into a validated Settings object."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        raise ConfigError(f"Config file not found: {cfg_path}")
    with cfg_path.open(encoding="utf-8") as fh:
        try:
            raw = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise ConfigError(f"{cfg_path}: invalid YAML: {exc}") from exc
    return parse_config(raw, data_dir=Path(data_dir) if data_dir is not None else None)


_env_loaded = False


def get_secret(name: str, *, env_file: str | Path | None = None) -> str:
    """Return a secret from the environment, loading .env once. Raises if unset."""
    global _env_loaded
    if env_file is not None:
        load_dotenv(env_file, override=False)
    elif not _env_loaded:
        load_dotenv(DEFAULT_ENV_FILE, override=False)
        _env_loaded = True
    value = os.environ.get(name, "").strip()
    if not value:
        raise MissingSecretError(f"{name} is not set. Add it to .env (see .env.example).")
    return value
