"""Tiny polite JSON-over-HTTP helper (stdlib only): retries, backoff, rate limit."""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any, Callable

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
)

GetJson = Callable[[str], Any]


class HttpError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def make_get_json(
    *,
    headers: dict[str, str] | None = None,
    min_interval: float = 0.35,
    retries: int = 3,
    timeout: float = 20.0,
    sleep: Callable[[float], None] = time.sleep,
) -> GetJson:
    """Return get_json(url) -> parsed JSON, spacing requests at least min_interval apart."""
    hdrs = {"User-Agent": USER_AGENT, "Accept": "application/json, text/plain, */*"}
    hdrs.update(headers or {})
    last = [0.0]

    def get_json(url: str) -> Any:
        for attempt in range(retries + 1):
            wait = min_interval - (time.monotonic() - last[0])
            if wait > 0:
                sleep(wait)
            last[0] = time.monotonic()
            req = urllib.request.Request(url, headers=hdrs)
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code in (400, 401, 403, 404) or attempt == retries:
                    raise HttpError(f"GET {url} -> HTTP {exc.code}", exc.code) from exc
                err: Exception = exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ConnectionError) as exc:
                if attempt == retries:
                    raise HttpError(f"GET {url} failed: {exc}") from exc
                err = exc
            delay = 2.0 * 2**attempt
            logger.warning("GET %s failed (%s); retry in %.0fs", url, err, delay)
            sleep(delay)
        raise HttpError(f"GET {url} failed")  # pragma: no cover

    return get_json
