"""Safe wrapper around apify-client: run an actor, wait, fetch items, save raw JSON."""

from __future__ import annotations

import logging
import time as _time
from datetime import date
from typing import Any, Callable

from crowdpulse.config import Settings, get_secret, load_config
from crowdpulse.ingest.cache import load_raw, raw_path, save_raw

logger = logging.getLogger(__name__)

# HTTP statuses where retrying cannot help (bad input, bad token, unknown actor).
_NON_RETRYABLE_STATUS = {400, 401, 403, 404}
_BACKOFF_BASE_SECS = 5.0
_BACKOFF_MAX_SECS = 60.0


class ApifyRunError(RuntimeError):
    """The actor run did not finish successfully."""

    def __init__(self, message: str, *, run_id: str | None = None, status: str | None = None):
        super().__init__(message)
        self.run_id = run_id
        self.status = status


def _default_client_factory(token: str) -> Any:
    # Imported lazily so the package (and offline tests) work without apify-client installed.
    from apify_client import ApifyClient

    return ApifyClient(token)


def _field(obj: Any, *names: str) -> Any:
    """Read a field from a dict or an object, trying camelCase and snake_case names."""
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if not isinstance(obj, dict) and hasattr(obj, name):
            return getattr(obj, name)
    return None


def proxy_input(settings: Settings, *, residential: bool | None = None) -> dict[str, Any]:
    """The proxyConfiguration block for actor input. Residential when configured."""
    use_res = settings.apify.use_residential_proxy if residential is None else residential
    if use_res:
        return {"useApifyProxy": True, "apifyProxyGroups": ["RESIDENTIAL"]}
    return {"useApifyProxy": True}


def _is_retryable(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None)
    return status not in _NON_RETRYABLE_STATUS


def run_actor(
    actor_id: str,
    run_input: dict[str, Any],
    *,
    source: str,
    key: str,
    settings: Settings | None = None,
    use_cache: bool = False,
    client: Any = None,
    client_factory: Callable[[str], Any] = _default_client_factory,
    day: date | None = None,
    sleep: Callable[[float], None] = _time.sleep,
) -> list[dict[str, Any]]:
    """Start an Apify actor, wait for it, download its dataset items and save them raw.

    `source` and `key` name the raw file: data/raw/<source>/<day>/<key>.json.
    With use_cache=True, an existing complete file for today is returned without
    calling Apify. Retries with exponential backoff up to settings.apify.max_retries.
    """
    settings = settings or load_config()
    path = raw_path(settings.data_dir, source, key, day)
    params = {"actor_id": actor_id, "run_input": run_input}

    if use_cache:
        cached = load_raw(path, params=params)
        if cached is not None:
            return cached["items"]

    if client is None:
        client = client_factory(get_secret("APIFY_TOKEN"))

    attempts = settings.apify.max_retries + 1
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            items, run_id = _run_once(client, actor_id, run_input, settings.apify.timeout_secs)
        except Exception as exc:  # noqa: BLE001 - classified below
            last_error = exc
            if not _is_retryable(exc):
                logger.error("Apify actor %s failed with a non-retryable error: %s", actor_id, exc)
                raise
            if attempt == attempts:
                break
            delay = min(_BACKOFF_BASE_SECS * 2 ** (attempt - 1), _BACKOFF_MAX_SECS)
            logger.warning(
                "Apify actor %s attempt %d/%d failed (%s); retrying in %.0fs",
                actor_id, attempt, attempts, exc, delay,
            )
            sleep(delay)
            continue

        if not items:
            logger.warning("Apify actor %s (run %s) returned 0 items for %s", actor_id, run_id, key)
        save_raw(path, items, source=source, key=key, params=params, extra={"run_id": run_id})
        return items

    assert last_error is not None
    if isinstance(last_error, ApifyRunError):
        raise ApifyRunError(
            f"Apify actor {actor_id} failed after {attempts} attempts: {last_error}",
            run_id=last_error.run_id,
            status=last_error.status,
        ) from last_error
    raise ApifyRunError(f"Apify actor {actor_id} failed after {attempts} attempts: {last_error}") from last_error


def _run_once(
    client: Any, actor_id: str, run_input: dict[str, Any], timeout_secs: int
) -> tuple[list[dict[str, Any]], str | None]:
    run = client.actor(actor_id).call(run_input=run_input, timeout_secs=timeout_secs)
    if run is None:
        raise ApifyRunError(f"Apify actor {actor_id} returned no run object")
    run_id = _field(run, "id")
    status = _field(run, "status")
    status = getattr(status, "value", status)  # enum or plain string
    if status != "SUCCEEDED":
        raise ApifyRunError(
            f"run {run_id} ended with status {status}", run_id=run_id, status=str(status)
        )
    dataset_id = _field(run, "defaultDatasetId", "default_dataset_id")
    if not dataset_id:
        raise ApifyRunError(f"run {run_id} has no default dataset", run_id=run_id, status=status)
    items = [dict(item) for item in client.dataset(dataset_id).iterate_items()]
    logger.info("Apify actor %s run %s: %d items", actor_id, run_id, len(items))
    return items, run_id
