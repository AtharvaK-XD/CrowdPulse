"""Shared fixtures for offline backend tests. Nothing here touches the network."""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crowdpulse.config import Settings, load_config  # noqa: E402

SAMPLES = Path(__file__).parent / "samples"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if any test tries to open a socket."""

    def guard(*args, **kwargs):
        raise RuntimeError("Network access is not allowed in offline tests")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "create_connection", guard)


@pytest.fixture
def samples_dir() -> Path:
    return SAMPLES


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Sample config with data_dir pointed at a temp folder."""
    return load_config(SAMPLES / "config_sample.yaml", data_dir=tmp_path / "data")


def load_sample(name: str):
    with (SAMPLES / name).open(encoding="utf-8") as fh:
        return json.load(fh)
