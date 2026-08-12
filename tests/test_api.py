import io
import json
import uuid
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.control_plane import ControlPlane
from app.main import create_app, model_service

KEY_ID = "018f4f45-2d61-7dd0-a01b-0242ac120000"


def image_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def active(**changes):
    payload = {
        "active": True,
        "key_id": KEY_ID,
        "scopes": ["captcha:predict"],
        "expires_at": None,
        "cache_ttl_seconds": 30,
    }
    payload.update(changes)
    return payload


@pytest.fixture
def service(monkeypatch):
    requests = []
    introspection = active()
    usage_statuses = [202]

    def handler(request):
        requests.append(request)
        if request.url.path.endswith("/introspect"):
            value = introspection
            if isinstance(value, Exception):
                raise value
            return httpx.Response(200, json=value)
        status = usage_statuses.pop(0) if usage_statuses else 202
        return httpx.Response(
            status,
            json={"accepted": True, "duplicate": status == 200},
        )

    control = ControlPlane(
        "https://control.invalid",
        "service-secret",
        httpx.MockTransport(handler),
    )
    monkeypatch.setattr(model_service, "predict", lambda image: "ABCDE")
    monkeypatch.setattr(model_service, "ready", lambda: True)
    with TestClient(create_app(control)) as client:
        yield client, control, requests, introspection, usage_statuses


def post(client, headers=None):
    return client.post(
        "/predict",
        headers=headers or {},
        files={"file": ("captcha.png", image_bytes(), "image/png")},
    )


def test_health_public_and_request_id(service):
    client, _, requests, _, _ = service
    response = client.get("/health/live", headers={"X-Request-ID": "valid-id"})
    assert response.status_code == 200
    assert response.headers["x-request-id"] == "valid-id"
    assert response.headers["cache-control"] == "no-store"
    assert client.get("/health/ready").status_code == 200
    assert not requests


def test_invalid_request_id_is_replaced(service):
    response = service[0].get("/health/live", headers={"X-Request-ID": "bad id"})
    assert len(response.headers["x-request-id"]) == 32
    int(response.headers["x-request-id"], 16)


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Basic key"},
        {"Authorization": "Bearer "},
        {"Authorization": "Bearer bad key"},
        {"x-api-key": "user-key"},
    ],
)
def test_missing_or_malformed_bearer_rejected(service, headers):
    response = post(service[0], headers)
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"
    assert not service[2]


def test_inactive_is_public_401(service):
    service[3].clear()
    service[3].update({"active": False, "cache_ttl_seconds": 0})
    response = post(service[0], {"Authorization": "Bearer user-key"})
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


def test_valid_inference_contract_and_usage(service):
    client, _, requests, introspection, _ = service
    introspection["future_field"] = {"accepted": True}
    response = post(
        client,
        {"Authorization": "Bearer user-key", "X-Request-ID": "request-1"},
    )
    assert response.status_code == 200
    assert response.json() == {"prediction": "ABCDE"}
    inspect_request, usage_request = requests
    assert inspect_request.headers["authorization"] == "Bearer service-secret"
    assert inspect_request.headers["x-request-id"] == "request-1"
    assert json.loads(inspect_request.content) == {
        "api_key": "user-key",
        "required_scope": "captcha:predict",
    }
    payload = json.loads(usage_request.content)
    uuid.UUID(payload["batch_id"])
    assert (
        payload["records"][0]
        | {
            "latency_sum_ms": payload["records"][0]["latency_sum_ms"],
            "latency_max_ms": payload["records"][0]["latency_max_ms"],
        }
        == payload["records"][0]
    )
    assert payload["records"][0]["key_id"] == KEY_ID
    assert payload["records"][0]["endpoint_template"] == "/predict"
    assert payload["records"][0]["status_class"] == "2xx"


def test_positive_cache_skips_second_introspection(service):
    client, _, requests, _, _ = service
    headers = {"Authorization": "Bearer same-key"}
    assert post(client, headers).status_code == 200
    assert post(client, headers).status_code == 200
    assert sum(request.url.path.endswith("/introspect") for request in requests) == 1


@pytest.mark.parametrize(
    "change",
    [
        {"key_id": "bad"},
        {"scopes": ["wrong"]},
        {"expires_at": "bad"},
        {"cache_ttl_seconds": 31},
        {"rate_limits": {}},
        {"rate_limits": {"predict": {"requests": 1, "window_seconds": 30}}},
        {"rate_limits": {"predict": {"requests": 1}}},
    ],
)
def test_malformed_known_fields_fail_closed(service, change):
    service[3].update(change)
    response = post(service[0], {"Authorization": "Bearer user-key"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "authorization_unavailable"


def test_expired_response_fails_closed(service):
    service[3]["expires_at"] = (
        datetime.now(timezone.utc) - timedelta(seconds=1)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert post(service[0], {"Authorization": "Bearer user-key"}).status_code == 503


def test_timeout_fails_closed(service):
    service[3].clear()
    service[3].update({"exception": True})

    async def timeout(*args, **kwargs):
        raise httpx.ReadTimeout("timeout")

    service[1].client.post = timeout
    response = post(service[0], {"Authorization": "Bearer user-key"})
    assert response.status_code == 503


def test_rate_limit_and_retry_after(service):
    service[3]["rate_limits"] = {"predict": {"requests": 1, "window_seconds": 60}}
    headers = {"Authorization": "Bearer limited-key"}
    assert post(service[0], headers).status_code == 200
    response = post(service[0], headers)
    assert response.status_code == 429
    assert int(response.headers["retry-after"]) >= 1
    assert response.json()["error"]["code"] == "rate_limited"


def test_zero_requests_denies(service):
    service[3]["rate_limits"] = {"predict": {"requests": 0, "window_seconds": 60}}
    assert post(service[0], {"Authorization": "Bearer denied-key"}).status_code == 429


def test_usage_retries_stable_payload(service):
    client, control, requests, _, statuses = service
    statuses[:] = [503, 202]
    headers = {"Authorization": "Bearer usage-key"}
    assert post(client, headers).status_code == 200
    first = [request for request in requests if request.url.path.endswith("/batches")][
        0
    ]
    first_payload = json.loads(first.content)
    assert control.pending_usage == first_payload
    assert post(client, headers).status_code == 200
    usage = [request for request in requests if request.url.path.endswith("/batches")]
    assert json.loads(usage[1].content) == first_payload
    assert control.pending_usage is None
