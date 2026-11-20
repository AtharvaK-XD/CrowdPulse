from __future__ import annotations

import json
from datetime import date

import pytest

from crowdpulse.ingest.cache import load_raw, raw_path, safe_name, save_raw


def test_raw_path_layout(tmp_path):
    p = raw_path(tmp_path, "naver", "005930.KS", date(2026, 11, 20))
    assert p == tmp_path / "raw" / "naver" / "2026-11-20" / "005930.KS.json"


def test_safe_name():
    assert safe_name("apify/cheerio-scraper") == "apify_cheerio-scraper"
    with pytest.raises(ValueError):
        safe_name("///")


def test_save_then_load_roundtrip(tmp_path):
    p = raw_path(tmp_path, "exa", "0700.HK", date(2026, 11, 20))
    items = [{"title": "騰訊股價突破400元", "n": 1}]
    save_raw(p, items, source="exa", key="0700.HK", params={"q": 1})
    payload = load_raw(p, params={"q": 1})
    assert payload is not None
    assert payload["items"] == items
    assert payload["n_items"] == 1
    # Non-ASCII stored readably, not as \u escapes.
    assert "騰訊" in p.read_text(encoding="utf-8")
    # No temp files left behind.
    assert [f.name for f in p.parent.iterdir()] == ["0700.HK.json"]


def test_load_missing_returns_none(tmp_path):
    assert load_raw(tmp_path / "missing.json") is None


def test_incomplete_run_not_reused(tmp_path):
    p = tmp_path / "x.json"
    save_raw(p, [{"a": 1}], source="exa", key="x", complete=False)
    assert load_raw(p) is None


def test_corrupt_file_ignored(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    assert load_raw(p) is None
    p.write_text(json.dumps({"items": "nope"}), encoding="utf-8")
    assert load_raw(p) is None
