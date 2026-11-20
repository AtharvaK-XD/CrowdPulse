from __future__ import annotations

from datetime import date

import pytest

from crowdpulse.ingest.apify_client import ApifyRunError, proxy_input, run_actor
from crowdpulse.ingest.cache import raw_path

from .conftest import load_sample

DAY = date(2026, 11, 20)


class FakeApiError(Exception):
    def __init__(self, msg: str, status_code: int):
        super().__init__(msg)
        self.status_code = status_code


class FakeClient:
    """Mimics ApifyClient.actor(id).call(...) and .dataset(id).iterate_items()."""

    def __init__(self, outcomes, items):
        self.outcomes = list(outcomes)  # each: run dict, None, or Exception
        self.items = items
        self.calls: list[dict] = []

    def actor(self, actor_id):
        client = self

        class _Actor:
            def call(self, *, run_input, timeout_secs):
                client.calls.append({"actor_id": actor_id, "run_input": run_input, "timeout": timeout_secs})
                outcome = client.outcomes.pop(0)
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome

        return _Actor()

    def dataset(self, dataset_id):
        client = self

        class _Dataset:
            def iterate_items(self):
                assert dataset_id == "ds1"
                yield from client.items

        return _Dataset()


OK_RUN = {"id": "run1", "status": "SUCCEEDED", "defaultDatasetId": "ds1"}
FAILED_RUN = {"id": "run0", "status": "FAILED", "defaultDatasetId": "ds0"}


def _run(settings, client, **kw):
    sleeps: list[float] = []
    items = run_actor(
        "apify/cheerio-scraper", {"startUrls": [{"url": "https://x"}]},
        source="naver", key="005930.KS", settings=settings, client=client,
        day=DAY, sleep=sleeps.append, **kw,
    )
    return items, sleeps


def test_success_saves_raw(settings):
    sample = load_sample("apify_dataset_sample.json")
    client = FakeClient([OK_RUN], sample)
    items, sleeps = _run(settings, client)
    assert items == sample
    assert sleeps == []
    assert client.calls[0]["timeout"] == settings.apify.timeout_secs
    path = raw_path(settings.data_dir, "naver", "005930.KS", DAY)
    assert path.is_file()
    assert "삼성전자" in path.read_text(encoding="utf-8")


def test_retries_then_succeeds(settings):
    client = FakeClient([ConnectionError("reset"), FAILED_RUN, OK_RUN], [{"a": 1}])
    items, sleeps = _run(settings, client)
    assert items == [{"a": 1}]
    assert len(client.calls) == 3
    assert sleeps == [5.0, 10.0]  # exponential backoff


def test_gives_up_after_max_retries(settings):
    attempts = settings.apify.max_retries + 1
    client = FakeClient([FAILED_RUN] * attempts, [])
    with pytest.raises(ApifyRunError, match="failed after") as exc_info:
        _run(settings, client)
    assert exc_info.value.status == "FAILED"
    assert len(client.calls) == attempts
    assert not raw_path(settings.data_dir, "naver", "005930.KS", DAY).exists()


def test_auth_error_not_retried(settings):
    client = FakeClient([FakeApiError("bad token", 401)], [])
    with pytest.raises(FakeApiError):
        _run(settings, client)
    assert len(client.calls) == 1


def test_none_run_is_retried(settings):
    client = FakeClient([None, OK_RUN], [{"a": 1}])
    items, _ = _run(settings, client)
    assert items == [{"a": 1}]


def test_cache_hit_skips_api(settings):
    first = FakeClient([OK_RUN], [{"a": 1}])
    _run(settings, first)
    second = FakeClient([], [])  # would IndexError if called
    items, _ = _run(settings, second, use_cache=True)
    assert items == [{"a": 1}]
    assert second.calls == []


def test_without_cache_flag_calls_again(settings):
    _run(settings, FakeClient([OK_RUN], [{"a": 1}]))
    again = FakeClient([OK_RUN], [{"a": 2}])
    items, _ = _run(settings, again)
    assert items == [{"a": 2}]
    assert len(again.calls) == 1


def test_missing_token_raises_before_any_call(settings, monkeypatch):
    from crowdpulse.config import MissingSecretError
    import crowdpulse.ingest.apify_client as mod

    monkeypatch.delenv("APIFY_TOKEN", raising=False)
    monkeypatch.setattr("crowdpulse.config._env_loaded", True)  # don't read a real .env
    with pytest.raises(MissingSecretError, match="APIFY_TOKEN"):
        mod.run_actor("a/b", {}, source="naver", key="k", settings=settings, day=DAY)


def test_proxy_input(settings):
    assert proxy_input(settings) == {"useApifyProxy": True}
    assert proxy_input(settings, residential=True)["apifyProxyGroups"] == ["RESIDENTIAL"]


def test_new_client_signature_uses_run_timeout(settings):
    """apify-client >= 3 takes run_timeout (timedelta) and max_total_charge_usd, not timeout_secs."""
    from datetime import timedelta
    from decimal import Decimal

    seen = {}

    class V3Client(FakeClient):
        def actor(self, actor_id):
            client = self

            class _Actor:
                def call(self, *, run_input=None, memory_mbytes=None, run_timeout=None,
                         max_total_charge_usd=None, wait_duration=None):
                    seen.update(run_timeout=run_timeout, memory=memory_mbytes, charge=max_total_charge_usd)
                    return client.outcomes.pop(0)

            return _Actor()

    items, _ = _run(settings, V3Client([OK_RUN], [{"a": 1}]), memory_mbytes=2048, max_charge_usd=1.0)
    assert items == [{"a": 1}]
    assert seen == {"run_timeout": timedelta(seconds=settings.apify.timeout_secs), "memory": 2048,
                    "charge": Decimal("1.0")}
