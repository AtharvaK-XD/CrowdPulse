from __future__ import annotations

from datetime import time
from pathlib import Path

import pytest
import yaml

from crowdpulse.config import (
    ConfigError,
    MissingSecretError,
    get_secret,
    load_config,
    parse_config,
)

from .conftest import SAMPLES


def _raw() -> dict:
    with (SAMPLES / "config_sample.yaml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_loads_sample_config(settings):
    assert settings.lookback_days == 10
    assert settings.price_history_days == 120
    assert len(settings.tickers) == 10
    assert set(settings.exchanges) == {"HKEX", "KRX"}
    assert len(settings.tickers_for("HKEX")) == 5
    assert len(settings.tickers_for("KRX")) == 5


def test_exchange_fields(settings):
    hk = settings.exchanges["HKEX"]
    kr = settings.exchanges["KRX"]
    assert hk.close_utc == time(8, 0)
    assert kr.close_utc == time(6, 30)
    assert hk.currency == "HKD" and kr.currency == "KRW"
    assert hk.sources == ("futu", "aastocks")
    assert kr.backup_sources == ("blind",)
    assert str(kr.tz) == "Asia/Seoul"


def test_ticker_lookup(settings):
    t = settings.ticker("005930.KS")
    assert t.name == "Samsung Electronics"
    assert t.code == "005930"
    assert "삼성전자 주가" in t.keywords_local
    assert settings.exchange_of("0700.HK").code == "HKEX"
    with pytest.raises(KeyError):
        settings.ticker("AAPL.US")


def test_data_dir_override(tmp_path):
    s = load_config(SAMPLES / "config_sample.yaml", data_dir=tmp_path)
    assert s.data_dir == tmp_path


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_unknown_exchange_rejected():
    raw = _raw()
    raw["tickers"][0]["exchange"] = "NYSE"
    with pytest.raises(ConfigError, match="not defined"):
        parse_config(raw)


def test_duplicate_symbol_rejected():
    raw = _raw()
    raw["tickers"].append(dict(raw["tickers"][0]))
    with pytest.raises(ConfigError, match="duplicate"):
        parse_config(raw)


def test_bad_timezone_rejected():
    raw = _raw()
    raw["exchanges"]["HKEX"]["timezone"] = "Mars/Olympus"
    with pytest.raises(ConfigError, match="timezone"):
        parse_config(raw)


def test_bad_close_rejected():
    raw = _raw()
    raw["exchanges"]["KRX"]["close_utc"] = "6pm"
    with pytest.raises(ConfigError, match="close_utc"):
        parse_config(raw)


def test_unquoted_close_time_as_minutes():
    raw = _raw()
    raw["exchanges"]["HKEX"]["close_utc"] = 480  # YAML 1.1 reading of 8:00
    assert parse_config(raw).exchanges["HKEX"].close_utc == time(8, 0)


def test_bad_symbol_rejected():
    raw = _raw()
    raw["tickers"][0]["symbol"] = "tencent"
    with pytest.raises(ConfigError, match="symbol"):
        parse_config(raw)


def test_defaults_when_optional_sections_missing():
    raw = _raw()
    for key in ("news", "apify", "lookback_days", "price_history_days"):
        raw.pop(key)
    s = parse_config(raw)
    assert s.lookback_days == 10
    assert s.news.results_per_query == 15
    assert s.apify.max_retries == 3


def test_get_secret_reads_env_file(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("CP_TEST_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("CP_TEST_KEY=abc123\n", encoding="utf-8")
    assert get_secret("CP_TEST_KEY", env_file=env) == "abc123"


def test_get_secret_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("CP_MISSING_KEY", raising=False)
    with pytest.raises(MissingSecretError, match="CP_MISSING_KEY"):
        get_secret("CP_MISSING_KEY", env_file=tmp_path / "absent.env")
