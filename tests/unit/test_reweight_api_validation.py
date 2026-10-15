"""HTTP input validation preserves raw task coordinate types before coercion."""

import pytest

from aether.runtime.session_contract import validate_task_weights


@pytest.fixture
def client():
    """Mount the real router with a recorded runtime policy boundary."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from aether.server.routes import create_router

    class RuntimePolicy:
        def __init__(self):
            self.calls = []

        def set_model_task_weights(self, model, weights):
            self.calls.append((model, weights))
            return validate_task_weights(weights, normalize=True)

    policy = RuntimePolicy()
    app = FastAPI()
    app.include_router(create_router(policy), prefix="/v1")
    with TestClient(app) as http:
        yield http, policy


@pytest.mark.parametrize("value", [True, False, "1", "NaN", None, -1, []])
def test_invalid_coordinates_never_reach_runtime_after_coercion(client, value):
    """Boolean/string/negative coordinates remain invalid at the HTTP boundary."""
    http, runtime = client
    response = http.post("/v1/merge/reweight", json={"model": "model", "weights": {"task": value}})
    assert response.status_code == 422
    assert runtime.calls == []


def test_finite_large_coefficients_preserve_overflow_safe_normalization(client):
    """Valid numeric JSON can normalize without overflowing its finite sum."""
    http, runtime = client
    response = http.post(
        "/v1/merge/reweight", json={"model": "model", "weights": {"a": 1e308, "b": 1e308}}
    )
    assert response.status_code == 200
    assert response.json()["weights"] == {"a": 0.5, "b": 0.5}
    assert len(runtime.calls) == 1


def test_task_labels_are_data_at_the_http_boundary(client):
    """Names matching Python parameters remain ordinary valid task labels."""
    http, runtime = client
    response = http.post(
        "/v1/merge/reweight",
        json={"model": "model", "weights": {"weights": 1, "self": 1}},
    )
    assert response.status_code == 200
    assert response.json()["weights"] == {"self": 0.5, "weights": 0.5}
    assert runtime.calls == [("model", {"self": 1.0, "weights": 1.0})]


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_raw_json_has_a_controlled_error_response(client, literal):
    """Nonstandard JSON floats cannot crash error rendering or update runtime state."""
    http, runtime = client
    response = http.post(
        "/v1/merge/reweight",
        content='{"model":"model","weights":{"task":' + literal + "}}",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert "finite" in response.json()["detail"]
    assert runtime.calls == []
