from __future__ import annotations

import time

import httpx
import pytest

from watchwise.config import (
    AgentSettings,
    DecisionSettings,
    GateSettings,
    MQTTSettings,
    MemorySettings,
    NVRSettings,
    WatchwiseConfig,
    VLMSettings,
)
from watchwise.fakes.nvr_server import FakeNVRState, build_fake_nvr
from watchwise.memory.long_term import PatternLibrary
from watchwise.memory.store import Store
from watchwise.memory.vectors import HashEmbedder
from watchwise.nvr.client import NVRClient
from watchwise.types import AlertContext, Severity

# Fixed anchor so time-of-day dependent behaviour is deterministic.
# 2026-03-10 02:30 local -> night.
NIGHT_TS = time.mktime((2026, 3, 10, 2, 30, 0, 0, 0, -1))
# 2026-03-10 14:00 local -> daytime.
DAY_TS = time.mktime((2026, 3, 10, 14, 0, 0, 0, 0, -1))


@pytest.fixture(autouse=True)
def _no_developer_dotenv(monkeypatch):
    """`watchwise` commands load ./.env, which on a developer's machine holds
    real keys and tokens. Tests must never see them."""
    from watchwise import cli

    real = cli.load_dotenv
    monkeypatch.setattr(cli, "load_dotenv", lambda path=None: real(path) if path is not None else [])


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "watchwise.db")
    yield s
    s.close()


@pytest.fixture
def patterns(store):
    return PatternLibrary(store, HashEmbedder(128), half_life_days=30.0)


@pytest.fixture
def nvr_state():
    return FakeNVRState()


@pytest.fixture
async def nvr_client(nvr_state):
    app, _ = build_fake_nvr(nvr_state)
    transport = httpx.ASGITransport(app=app)
    http = httpx.AsyncClient(transport=transport, base_url="http://fake-nvr")
    client = NVRClient(NVRSettings(base_url="http://fake-nvr"), client=http)
    yield client
    await http.aclose()


@pytest.fixture
def config(tmp_path):
    return WatchwiseConfig(
        nvr=NVRSettings(base_url="http://fake-nvr"),
        mqtt=MQTTSettings(enabled=False),
        gate=GateSettings(dedupe_window_s=60.0),
        vlm=VLMSettings(backend="stub", model="stub-vlm", frames=2),
        agent=AgentSettings(backend="scripted", max_concurrency=1, timeout_s=30.0),
        memory=MemorySettings(db_path=str(tmp_path / "watchwise.db"), embedding_dim=128),
        decision=DecisionSettings(),
    )


def make_alert(
    *,
    review_id: str = "rev-1",
    camera: str = "cam_drive",
    labels: list[str] | None = None,
    zones: list[str] | None = None,
    start_time: float | None = None,
    duration: float = 8.0,
) -> AlertContext:
    start = start_time if start_time is not None else DAY_TS
    return AlertContext(
        review_id=review_id,
        camera=camera,
        start_time=start,
        end_time=start + duration,
        severity=Severity.ALERT,
        labels=labels if labels is not None else ["person"],
        zones=zones or [],
    )
