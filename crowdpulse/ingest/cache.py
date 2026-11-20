"""Raw-response storage under data/raw/<source>/<YYYY-MM-DD>/<key>.json.

Every paid API response is saved here. With caching on, a file written today
(UTC) is reused instead of calling the API again.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(value: str) -> str:
    """Make a string safe to use as a single path component."""
    cleaned = _SAFE.sub("_", value).strip("._")
    if not cleaned:
        raise ValueError(f"Cannot build a file name from {value!r}")
    return cleaned


def utc_today() -> date:
    return datetime.now(timezone.utc).date()


def raw_path(data_dir: Path, source: str, key: str, day: date | None = None) -> Path:
    day = day or utc_today()
    return Path(data_dir) / "raw" / safe_name(source) / day.isoformat() / f"{safe_name(key)}.json"


def save_raw(
    path: Path,
    items: list[Any],
    *,
    source: str,
    key: str,
    params: dict[str, Any] | None = None,
    complete: bool = True,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write items plus metadata atomically (temp file, then rename)."""
    payload: dict[str, Any] = {
        "source": source,
        "key": key,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "complete": complete,
        "params": params or {},
        "n_items": len(items),
        "items": items,
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    logger.info("Saved %d raw items to %s", len(items), path)
    return path


def load_raw(path: Path, *, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Return the cached payload, or None if missing, unreadable or incomplete."""
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable cache file %s: %s", path, exc)
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        logger.warning("Ignoring malformed cache file %s", path)
        return None
    if not payload.get("complete", True):
        logger.info("Cache file %s is from a partial run; refetching", path)
        return None
    if params is not None and payload.get("params") != json.loads(json.dumps(params, default=str)):
        logger.warning("Cache file %s was made with different parameters; reusing it anyway", path)
    logger.info("Cache hit: %s (%d items)", path, len(payload["items"]))
    return payload
