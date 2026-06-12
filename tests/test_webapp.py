"""FastAPI web-demo tests.

The full HTTP paths run on the keyless mock so they stay hermetic and fast; the
key-resolution and rate-limit logic is unit-tested directly (no network). Skipped
entirely when the optional [web] extra (fastapi) is not installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from starlette.testclient import TestClient  # noqa: E402

from discovery_agents import webapp  # noqa: E402
from discovery_agents.webapp import _RateLimiter, app, resolve_config  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_healthz(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_index_renders_form(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "<form" in r.text and "Discovery Agents" in r.text
    assert "name='provider'" in r.text and "name='dataset'" in r.text


def test_api_run_mock_returns_directions(client: TestClient) -> None:
    r = client.post("/api/run", json={"provider": "mock", "dataset": "sample"})
    assert r.status_code == 200
    body = r.json()
    assert len(body["directions"]) >= 5
    assert body["selected"] is not None
    assert body["evals"] and body["provider_used"]["mode"] == "mock"


def test_api_run_banking_dataset(client: TestClient) -> None:
    r = client.post("/api/run", json={"provider": "mock", "dataset": "banking"})
    assert r.status_code == 200
    body = r.json()
    assert len(body["directions"]) >= 5
    # Banking directions cite the banking evidence ids (B*), proving the dataset flowed through.
    assert any(
        cid.startswith("B") for d in body["directions"] for cid in d["evidence_ids"]
    )


def test_api_run_custom_evidence(client: TestClient) -> None:
    payload = {
        "provider": "mock",
        "brief": {"goal": "Make onboarding faster"},
        "evidence": [
            {"id": "E1", "text": "Users stall on the blank first screen", "tags": ["blank_state"]},
            {"id": "E2", "text": "Teams argue over which idea to build", "tags": ["alignment"]},
        ],
    }
    r = client.post("/api/run", json=payload)
    assert r.status_code == 200
    assert r.json()["directions"]


def test_post_form_renders_results(client: TestClient) -> None:
    r = client.post("/", data={"dataset": "sample", "provider": "mock"})
    assert r.status_code == 200
    assert "Selected direction" in r.text and "Eval scorecard" in r.text


def test_api_key_never_echoed(client: TestClient) -> None:
    secret = "super-secret-key-XYZ"
    r = client.post("/", data={"dataset": "sample", "provider": "mock", "api_key": secret})
    assert r.status_code == 200
    assert secret not in r.text  # keys must never appear in the response


# --- key resolution (no pipeline run) ---------------------------------------


def test_resolve_byok(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    config, mode = resolve_config("gemini", "user-key")
    assert mode == "byok" and config.provider == "gemini" and config.api_key == "user-key"


def test_resolve_mock_when_no_key(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    config, mode = resolve_config("gemini", None)
    assert mode == "mock" and config.provider == "mock"


def test_resolve_server_key_then_rate_limited(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "server-key")
    monkeypatch.setattr(webapp, "_LIMITER", _RateLimiter(limit=1, window_s=1000.0))
    # First request gets the free server run; the config carries no key (adapter reads env).
    config, mode = resolve_config("gemini", None)
    assert mode == "server" and config.provider == "gemini" and config.api_key is None
    # Second is over the limit -> mock.
    _, mode2 = resolve_config("gemini", None)
    assert mode2 == "mock"


def test_rate_limiter_sliding_window() -> None:
    rl = _RateLimiter(limit=2, window_s=10.0)
    assert [rl.allow(100.0), rl.allow(101.0), rl.allow(102.0)] == [True, True, False]
    # After the window passes, capacity frees up again.
    assert rl.allow(120.0) is True
