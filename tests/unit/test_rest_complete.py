"""Real REST contract tests for the CPU-safe server surface.

These tests intentionally exercise a running FastAPI application rather than
mocking route functions.  Hardware-dependent operations must fail explicitly;
an HTTP 200 response claiming success would be a contract violation.
"""

from __future__ import annotations

import pytest


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from aether.runtime.config import RuntimeConfig
    from aether.server.app import create_app

    return TestClient(create_app(RuntimeConfig(hf_offline=True, enable_semantic_cache=False)))


def test_system_routes_return_measured_runtime_state(client) -> None:
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"

    api_health = client.get("/v1/health")
    assert api_health.status_code == 200
    assert api_health.json()["status"] == "healthy"

    hardware = client.get("/v1/hardware")
    assert hardware.status_code == 200
    body = hardware.json()
    assert isinstance(body["target_id"], str)
    assert isinstance(body["cpu_count"], int)
    assert body["gpu_count"] >= 0

    metrics = client.get("/v1/metrics")
    assert metrics.status_code == 200
    assert metrics.json()["runtime_up"] == 1


def test_targets_traces_and_kv_routes_are_real(client) -> None:
    targets = client.get("/v1/targets")
    assert targets.status_code == 200
    target_body = targets.json()
    assert target_body["count"] >= 1
    assert any(item["target_id"] == "cpu_avx512" for item in target_body["targets"])

    unknown = client.get("/v1/targets/not-a-real-target")
    assert unknown.status_code == 404

    traces = client.get("/v1/traces")
    assert traces.status_code == 200
    assert isinstance(traces.json(), dict)

    kv = client.get("/v1/kv/transfer/stats")
    assert kv.status_code == 200
    assert kv.json()["network_available"] is False
    assert kv.json()["fallback_active"] is True


def test_unavailable_operations_fail_closed(client) -> None:
    embedding = client.post(
        "/v1/embeddings",
        json={"model": "missing.aeg", "input": ["hello"]},
    )
    assert embedding.status_code == 400
    assert "missing.aeg" in embedding.json()["detail"]

    grammar = client.post(
        "/v1/grammar/compile",
        json={
            "grammar_name": "digits",
            "grammar_type": "regex",
            "grammar_spec": r"\\d+",
        },
    )
    assert grammar.status_code == 503
    assert "not queued" in grammar.json()["detail"]

    tee = client.get("/v1/tee/attestation")
    assert tee.status_code == 503


def test_openapi_exposes_the_extended_routes(client) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    required = {
        "/v1/health",
        "/v1/generate",
        "/v1/chat",
        "/v1/chat/completions",
        "/v1/embeddings",
        "/v1/rerank",
        "/v1/compile",
        "/v1/eval",
        "/v1/tools/call",
        "/v1/grammar/compile",
        "/v1/targets",
        "/v1/tee/session",
        "/v1/video/generate",
        "/v1/train/grpo/start",
        "/v1/kv/transfer/stats",
        "/v1/multi_agent/spawn",
        "/v1/kernels/generate",
        "/v1/kernels/{name}/verified",
    }
    assert required <= paths.keys()


def test_multi_agent_spawn_preserves_explicit_parent_lineage(client) -> None:
    created = client.post(
        "/v1/multi_agent/session",
        json={"agent_count": 1, "model": "local-model", "shared_prefix": "system"},
    )
    assert created.status_code == 200, created.text
    session = created.json()
    parent_id = session["agent_sessions"][0]

    spawned = client.post(
        "/v1/multi_agent/spawn",
        json={
            "session_id": session["session_id"],
            "model": "local-model",
            "context": "inherited context",
            "inherit_agent_id": parent_id,
        },
    )
    assert spawned.status_code == 200, spawned.text
    body = spawned.json()
    assert body["inherited_kv"] is True
    assert body["agent_session_id"].startswith(session["session_id"] + "/agent_")


def test_kernel_generation_reports_unsupported_target_without_traceback(client) -> None:
    response = client.post(
        "/v1/kernels/generate",
        json={"target": "cuda_sm90", "op_name": "rmsnorm"},
    )
    assert response.status_code == 501
    assert "not available" in str(response.json()["detail"]).lower()


def test_api_key_authentication_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AETHER_API_KEYS", "integration-secret")
    from fastapi.testclient import TestClient

    from aether.runtime.config import RuntimeConfig
    from aether.server.app import create_app

    client = TestClient(create_app(RuntimeConfig(hf_offline=True)))
    assert client.get("/health").status_code == 401
    assert client.get(
        "/health", headers={"Authorization": "Bearer wrong-secret"}
    ).status_code == 401
    response = client.get(
        "/health", headers={"Authorization": "Bearer integration-secret"}
    )
    assert response.status_code == 200
    assert response.headers["x-request-id"]
