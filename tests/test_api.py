import asyncio
import copy
import io
import json
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient
from PIL import Image

from app.control_plane import (
    MAX_BATCH_RECORDS,
    MAX_RESPONSE_BYTES,
    ControlPlane,
    ControlPlaneError,
    KeyAccess,
    RateLimit,
    UsageFlushOutcome,
)
from app.main import create_app, model_service

KEY_ID = "018f4f45-2d61-7dd0-a01b-0242ac120000"
CLIENT_API_KEY = "test-client-api-key-sentinel"
SERVICE_CREDENTIAL = "test-service-credential-sentinel"

INVALID_URLS = [
    ("http://user:pass@127.0.0.1:18200", "userinfo"),
    ("http://127.0.0.1:18200?q=1", "query"),
    ("http://127.0.0.1:18200#frag", "fragment"),
    ("http://127.0.0.1:18200/../../../etc/passwd", "traversal"),
    ("http://127.0.0.1:18200/a//b", "double_slash"),
    ("http://127.0.0.1:18200/a/b/../c", "dot_segment"),
    ("ftp://127.0.0.1:18200", "unsupported_scheme"),
    ("http://127.0.0.1:99999", "bad_port"),
    ("http://host name:18200", "whitespace"),
    ("http://127.0.0.1:18200\\a", "backslash"),
    ("http://127.0.0.1:18200/%2f..", "encoded_slash_dot"),
    ("http://127.0.0.1:18200/%40", "encoded_at"),
    ("http://127.0.0.1:18200/%5c", "encoded_backslash"),
    ("http://127.0.0.1:18200/%2e/", "encoded_dot"),
    ("http://127.0.0.1:18200/%2E/", "encoded_dot_upper"),
    ("http://127.0.0.1:18200/%2e%2e/", "encoded_dotdot"),
    ("http://127.0.0.1:18200/%2E%2E/", "encoded_dotdot_upper"),
    ("http://127.0.0.1:18200/.%2e/", "mixed_dot_encoded"),
    ("http://127.0.0.1:18200/%2e./", "mixed_encoded_dot"),
    ("http://127.0.0.1:18200/%2e%2e/admin", "encoded_dotdot_traversal"),
    ("http://127.0.0.1:18200/%2e", "encoded_dot_no_slash"),
    ("http://127.0.0.1:18200/%2e%2e", "encoded_dotdot_no_slash"),
    ("http://127.0.0.1:18200/%zz", "malformed_percent"),
    ("http://127.0.0.1:18200/%2", "truncated_percent"),
    ("http://127.0.0.1:18200/%2e%2", "truncated_after_dot"),
    ("x" * 600, "overlong"),
    (" http://127.0.0.1:18200", "leading_space"),
    ("http://127.0.0.1:18200 ", "trailing_space"),
    ("http://\x01127.0.0.1:18200", "control_char"),
    ("http://127.0.0.1:18200/%252e/", "double_encoded_dot"),
    ("http://127.0.0.1:18200/%252e%252e/", "double_encoded_dotdot"),
    ("http://127.0.0.1:18200/%252E%252E/", "double_encoded_dotdot_upper"),
    ("http://127.0.0.1:18200/%25252e%25252e/", "triple_encoded_dotdot"),
    ("http://127.0.0.1:18200/.%252e/", "mixed_literal_double_dot"),
    ("http://127.0.0.1:18200/%252e./", "double_encoded_dot_trail"),
    ("http://127.0.0.1:18200/%252e%252e/admin", "double_encoded_traversal_path"),
    ("http://127.0.0.1:18200/%252e%252e", "double_encoded_traversal_noslash"),
    ("http://127.0.0.1:18200/%252f", "double_encoded_slash"),
    ("http://127.0.0.1:18200/%255c", "double_encoded_backslash"),
    ("http://127.0.0.1:18200/%253a", "double_encoded_colon"),
    ("http://127.0.0.1:18200/%2540", "double_encoded_at"),
    ("http://127.0.0.1:18200/%25zz", "double_encoded_malformed"),
    ("http://127.0.0.1:18200/%252525252e", "five_level_encoded_dot"),
]

VALID_DOTTED_PATHS = [
    ("http://127.0.0.1:18200/v1.0", "http://127.0.0.1:18200/v1.0"),
    ("http://127.0.0.1:18200/api/model.json", "http://127.0.0.1:18200/api/model.json"),
    ("http://127.0.0.1:18200/.well-known", "http://127.0.0.1:18200/.well-known"),
    ("http://127.0.0.1:18200/a.b.c", "http://127.0.0.1:18200/a.b.c"),
]


class ChunkStream(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.read_chunks = 0
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            self.read_chunks += 1
            yield chunk

    async def aclose(self):
        self.closed = True


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
            body = json.loads(request.content)
            raw_key = body.get("api_key", "")
            value = copy.deepcopy(introspection)
            if isinstance(value.get("exception"), Exception):
                raise value["exception"]
            if value.get("_per_key"):
                kid = uuid.uuid5(uuid.NAMESPACE_URL, raw_key)
                value["key_id"] = str(kid)
            return httpx.Response(200, json=value)
        status = usage_statuses.pop(0) if usage_statuses else 202
        return httpx.Response(
            status,
            json={"accepted": True, "duplicate": status == 200},
        )

    control = ControlPlane(
        "https://control.invalid",
        "test-service-credential-sentinel",
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


def wait_for(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    assert predicate()


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


def test_openapi_declares_x_api_key(service):
    schema = service[0].get("/openapi.json").json()
    scheme = schema["components"]["securitySchemes"]["APIKeyHeader"]
    assert scheme["type"] == "apiKey"
    assert scheme["in"] == "header"
    assert scheme["name"] == "X-API-Key"
    assert "managed" in scheme["description"].lower()
    assert schema["paths"]["/predict"]["post"]["security"] == [{"APIKeyHeader": []}]


def test_openapi_has_extensions_and_operation_id(service):
    schema = service[0].get("/openapi.json").json()
    predict_op = schema["paths"]["/predict"]["post"]
    assert predict_op["operationId"] == "predict_captcha"
    assert predict_op["x-api-scopes"] == ["captcha:predict"]
    assert predict_op["x-rate-limit-category"] == "predict"
    assert predict_op["x-usage-enabled"] is True
    assert {"401", "429", "503"} <= set(predict_op["responses"])
    retry_after = predict_op["responses"]["429"]["headers"]["Retry-After"]
    assert retry_after["schema"]["minimum"] == 1
    assert "global" in predict_op["responses"]["429"]["description"].lower()
    assert "per-key" in predict_op["responses"]["429"]["description"].lower()
    assert "security" not in schema["paths"]["/health/live"]["get"]
    assert "security" not in schema["paths"]["/health/ready"]["get"]


def test_openapi_operation_ids_are_unique(service):
    schema = service[0].get("/openapi.json").json()
    ids = []
    for path_op in schema["paths"].values():
        for method_op in path_op.values():
            if isinstance(method_op, dict) and "operationId" in method_op:
                ids.append(method_op["operationId"])
    assert len(ids) == len(set(ids))


def test_openapi_no_configured_secrets(service, caplog):
    caplog.set_level(logging.DEBUG)
    body = service[0].get("/openapi.json").text
    response = post(service[0], {"X-API-Key": CLIENT_API_KEY})
    captured = caplog.text
    assert response.status_code == 200
    assert CLIENT_API_KEY not in body
    assert SERVICE_CREDENTIAL not in body
    assert CLIENT_API_KEY not in response.text
    assert SERVICE_CREDENTIAL not in response.text
    assert CLIENT_API_KEY not in captured
    assert SERVICE_CREDENTIAL not in captured


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer user-key"},
        {"x-api-key": ""},
        {"x-api-key": "bad key"},
    ],
)
def test_missing_or_malformed_x_api_key_rejected(service, headers):
    response = post(service[0], headers)
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"
    assert not service[2]


def test_inactive_is_public_401(service):
    service[3].clear()
    service[3].update({"active": False, "cache_ttl_seconds": 0})
    response = post(service[0], {"x-api-key": "user-key"})
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


def test_valid_inference_contract_and_usage(service):
    client, _, requests, introspection, _ = service
    introspection["future_field"] = {"accepted": True}
    response = post(
        client,
        {"x-api-key": "user-key", "X-Request-ID": "request-1"},
    )
    assert response.status_code == 200
    assert response.json() == {"prediction": "ABCDE"}
    wait_for(lambda: len(requests) == 2)
    inspect_request, usage_request = requests
    assert inspect_request.method == "POST"
    assert inspect_request.url.path == "/internal/v1/keys/introspect"
    assert inspect_request.headers["authorization"] == f"Bearer {SERVICE_CREDENTIAL}"
    assert inspect_request.headers["x-request-id"] == "request-1"
    assert json.loads(inspect_request.content) == {
        "api_key": "user-key",
        "required_scope": "captcha:predict",
    }
    assert usage_request.method == "POST"
    assert usage_request.url.path == "/internal/v1/usage/batches"
    payload = json.loads(usage_request.content)
    uuid.UUID(payload["batch_id"])
    assert len(payload["records"]) == 1
    record = payload["records"][0]
    assert set(record) == {
        "bucket_start",
        "key_id",
        "endpoint_template",
        "method",
        "status_class",
        "request_count",
        "error_count",
        "latency_sum_ms",
        "latency_max_ms",
    }
    assert datetime.strptime(record["bucket_start"], "%Y-%m-%dT%H:%M:%SZ")
    assert record["key_id"] == KEY_ID
    assert record["endpoint_template"] == "/predict"
    assert record["method"] == "POST"
    assert record["status_class"] == "2xx"
    assert record["request_count"] == 1
    assert record["error_count"] == 0
    assert type(record["latency_sum_ms"]) is int
    assert type(record["latency_max_ms"]) is int
    assert 0 <= record["latency_max_ms"] <= record["latency_sum_ms"]


def test_positive_cache_skips_second_introspection(service):
    client, _, requests, _, _ = service
    headers = {"x-api-key": "same-key"}
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
    response = post(service[0], {"x-api-key": "user-key"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "authorization_unavailable"


def test_expired_response_fails_closed(service):
    service[3]["expires_at"] = (
        datetime.now(timezone.utc) - timedelta(seconds=1)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert post(service[0], {"x-api-key": "user-key"}).status_code == 503


def test_timeout_fails_closed(service):
    service[3].clear()
    service[3].update({"exception": True})

    async def timeout(*args, **kwargs):
        raise httpx.ReadTimeout("timeout")

    service[1].client.post = timeout
    response = post(service[0], {"x-api-key": "user-key"})
    assert response.status_code == 503


def test_rate_limit_and_retry_after(service):
    service[1].enforcement = True
    service[3]["rate_limits"] = {"predict": {"requests": 1, "window_seconds": 60}}
    headers = {"x-api-key": "limited-key"}
    assert post(service[0], headers).status_code == 200
    response = post(service[0], headers)
    assert response.status_code == 429
    assert int(response.headers["retry-after"]) >= 1
    assert response.json()["error"]["code"] == "rate_limited"


def test_zero_requests_denies(service):
    service[1].enforcement = True
    service[3]["rate_limits"] = {"predict": {"requests": 0, "window_seconds": 60}}
    assert post(service[0], {"x-api-key": "denied-key"}).status_code == 429


def test_usage_retries_stable_payload():
    attempts = []

    async def handler(request):
        payload = json.loads(request.content)
        attempts.append(payload)
        status = 503 if len(attempts) == 1 else 202
        return httpx.Response(
            status,
            json={"accepted": True, "duplicate": False},
        )

    async def scenario():
        control = ControlPlane(
            "https://control.invalid",
            SERVICE_CREDENTIAL,
            httpx.MockTransport(handler),
            usage_retry_attempts=2,
            usage_retry_delay_seconds=0.01,
        )
        control.record_usage(KEY_ID, 200, 10)
        assert await control.flush_usage("retry") is UsageFlushOutcome.ACCEPTED
        await control.close()

    asyncio.run(scenario())
    assert len(attempts) == 2
    assert attempts[0] == attempts[1]
    uuid.UUID(attempts[0]["batch_id"])


@pytest.mark.parametrize("failure", [429, 503, "timeout", "network"])
def test_transient_usage_failures_retain_exact_batch(failure):
    attempts = []

    async def handler(request):
        attempts.append(json.loads(request.content))
        if failure == "timeout":
            raise httpx.ReadTimeout("response detail must stay private")
        if failure == "network":
            raise httpx.ConnectError("response detail must stay private")
        return httpx.Response(failure, content=b"response detail must stay private")

    async def scenario():
        control = ControlPlane(
            "https://control.invalid",
            SERVICE_CREDENTIAL,
            httpx.MockTransport(handler),
            usage_retry_attempts=2,
            usage_retry_delay_seconds=0.01,
        )
        control.record_usage(KEY_ID, 200, 10)
        outcome = await control.flush_usage("retry")
        assert outcome is UsageFlushOutcome.TRANSIENT
        assert attempts[0] == attempts[1] == control.pending_usage
        assert control.pending_usage_count() == 1
        await control.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("status", [400, 409])
def test_permanent_usage_failure_dead_letters_and_advances(status, caplog):
    payloads = []

    async def handler(request):
        payloads.append(json.loads(request.content))
        if len(payloads) == 1:
            return httpx.Response(status, content=b"private validation response")
        return httpx.Response(202, json={"accepted": True, "duplicate": False})

    async def scenario():
        control = ControlPlane(
            "https://control.invalid",
            SERVICE_CREDENTIAL,
            httpx.MockTransport(handler),
            usage_max_dimensions=2,
        )
        control._last_drop_log = -100
        control.record_usage(KEY_ID, 200, 10)
        first = await control.prepare_usage_batch()
        newer_key = "00000000-0000-7000-8000-000000000002"
        control.record_usage(newer_key, 429, 20)
        outcome = await control._send_batch(first, "poison")
        assert outcome is UsageFlushOutcome.DEAD_LETTERED
        assert await control.flush_usage("next") is UsageFlushOutcome.ACCEPTED
        assert control.pending_usage_count() == 0
        assert control.usage_dropped_batches_count == 1
        assert control.usage_dropped_records_count == 1
        assert control.usage_failure_counts == {
            "stale": 0,
            "permanent_payload": 1,
            "authentication": 0,
            "permanent_response": 0,
        }
        await control.close()

    asyncio.run(scenario())
    assert payloads[0]["batch_id"] != payloads[1]["batch_id"]
    assert payloads[1]["records"][0]["status_class"] == "4xx"
    assert "category=permanent_payload batches=1 records=1" in caplog.text
    assert KEY_ID not in caplog.text
    assert SERVICE_CREDENTIAL not in caplog.text
    assert "private validation response" not in caplog.text


def test_stale_usage_is_pruned_before_sending(caplog):
    requests = []

    async def handler(request):
        requests.append(request)
        return httpx.Response(202, json={"accepted": True, "duplicate": False})

    async def scenario():
        control = ControlPlane(
            "https://control.invalid",
            SERVICE_CREDENTIAL,
            httpx.MockTransport(handler),
            usage_record_max_age_seconds=60,
        )
        control._last_drop_log = -100
        control.record_usage(KEY_ID, 200, 10)
        record = next(iter(control.usage.values()))
        record["bucket_start"] = (
            datetime.now(timezone.utc) - timedelta(seconds=61)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        assert await control.flush_usage("stale") is UsageFlushOutcome.IDLE
        assert control.pending_usage_count() == 0
        assert control.usage_dropped_batches_count == 0
        assert control.usage_dropped_records_count == 1
        assert control.usage_failure_counts["stale"] == 1
        await control.close()

    asyncio.run(scenario())
    assert requests == []
    assert "category=stale batches=0 records=1" in caplog.text
    assert KEY_ID not in caplog.text


@pytest.mark.parametrize("status", [401, 403])
def test_auth_usage_failures_are_bounded_and_cannot_block_queue(status):
    payloads = []

    async def handler(request):
        payloads.append(json.loads(request.content))
        if len(payloads) <= 2:
            return httpx.Response(status, content=b"credential response stays private")
        return httpx.Response(202, json={"accepted": True, "duplicate": False})

    async def scenario():
        control = ControlPlane(
            "https://control.invalid",
            SERVICE_CREDENTIAL,
            httpx.MockTransport(handler),
            usage_retry_attempts=2,
            usage_retry_delay_seconds=0.01,
            usage_max_dimensions=2,
        )
        control.record_usage(KEY_ID, 200, 10)
        first = await control.prepare_usage_batch()
        control.record_usage("00000000-0000-7000-8000-000000000002", 200, 20)
        outcome = await control._send_batch(first, "auth")
        assert outcome is UsageFlushOutcome.DEAD_LETTERED
        assert payloads[0] == payloads[1]
        assert control.usage_failure_counts["authentication"] == 1
        assert await control.flush_usage("next") is UsageFlushOutcome.ACCEPTED
        assert control.pending_usage_count() == 0
        await control.close()

    asyncio.run(scenario())
    assert payloads[2]["batch_id"] != payloads[0]["batch_id"]


def test_redirect_is_rejected(monkeypatch):
    async def handler(request):
        return httpx.Response(
            307,
            headers={"Location": "https://other.example.invalid/introspect"},
        )

    control = ControlPlane(
        "https://control.invalid",
        SERVICE_CREDENTIAL,
        httpx.MockTransport(handler),
    )
    monkeypatch.setattr(model_service, "predict", lambda image: "ABCDE")
    with TestClient(create_app(control)) as client:
        response = post(client, {"x-api-key": CLIENT_API_KEY})
    assert response.status_code == 503


def test_stream_rejects_oversized_declared_length_without_reading():
    stream = ChunkStream([b"{}"])

    async def handler(request):
        return httpx.Response(
            200,
            headers={"Content-Length": str(MAX_RESPONSE_BYTES + 1)},
            stream=stream,
        )

    async def scenario():
        control = ControlPlane(
            "https://control.invalid",
            SERVICE_CREDENTIAL,
            httpx.MockTransport(handler),
        )
        with pytest.raises(ControlPlaneError):
            await control.introspect(CLIENT_API_KEY, "declared")
        await control.close()

    asyncio.run(scenario())
    assert stream.read_chunks == 0
    assert stream.closed


def test_stream_stops_when_chunked_body_exceeds_limit(caplog):
    sentinel = b"configured-response-secret-sentinel"
    stream = ChunkStream([b"{", b"x" * MAX_RESPONSE_BYTES, sentinel])

    async def handler(request):
        return httpx.Response(200, stream=stream)

    async def scenario():
        control = ControlPlane(
            "https://control.invalid",
            SERVICE_CREDENTIAL,
            httpx.MockTransport(handler),
        )
        with pytest.raises(ControlPlaneError):
            await control.introspect(CLIENT_API_KEY, "chunked")
        await control.close()

    asyncio.run(scenario())
    assert stream.read_chunks == 2
    assert stream.closed
    assert sentinel.decode() not in caplog.text


def test_stream_accepts_body_exactly_at_limit():
    prefix = b'{"active":false,"cache_ttl_seconds":0,"padding":"'
    suffix = b'"}'
    body = prefix + b"x" * (MAX_RESPONSE_BYTES - len(prefix) - len(suffix)) + suffix
    stream = ChunkStream([body])

    async def handler(request):
        return httpx.Response(
            200,
            headers={"Content-Length": str(MAX_RESPONSE_BYTES)},
            stream=stream,
        )

    async def scenario():
        control = ControlPlane(
            "https://control.invalid",
            SERVICE_CREDENTIAL,
            httpx.MockTransport(handler),
        )
        assert await control.introspect(CLIENT_API_KEY, "exact") is None
        await control.close()

    asyncio.run(scenario())
    assert stream.read_chunks == 1
    assert stream.closed


def test_malformed_json_fails_closed_without_content_leak(caplog):
    private_body = b"malformed-private-response-sentinel"

    async def handler(request):
        return httpx.Response(200, content=private_body)

    async def scenario():
        control = ControlPlane(
            "https://control.invalid",
            SERVICE_CREDENTIAL,
            httpx.MockTransport(handler),
        )
        with pytest.raises(ControlPlaneError):
            await control.introspect(CLIENT_API_KEY, "malformed")
        await control.close()

    asyncio.run(scenario())
    assert private_body.decode() not in caplog.text


def test_partial_rate_limits_policy_tolerated(service):
    service[3]["rate_limits"] = {"other": {"requests": 10, "window_seconds": 60}}
    response = post(service[0], {"x-api-key": "user-key"})
    assert response.status_code == 200


def test_different_keys_have_isolated_buckets(service):
    service[1].enforcement = True
    service[3]["rate_limits"] = {"predict": {"requests": 2, "window_seconds": 60}}
    service[3]["_per_key"] = True
    key_a = "key-a-abcdefghijklmnop"
    key_b = "key-b-abcdefghijklmnop"
    assert post(service[0], {"x-api-key": key_a}).status_code == 200
    assert post(service[0], {"x-api-key": key_b}).status_code == 200
    assert post(service[0], {"x-api-key": key_a}).status_code == 200
    assert post(service[0], {"x-api-key": key_b}).status_code == 200
    assert post(service[0], {"x-api-key": key_a}).status_code == 429
    assert post(service[0], {"x-api-key": key_b}).status_code == 429


def test_missing_x_api_key_header_uses_uppercase(service):
    response = service[0].post(
        "/predict",
        headers={"X-API-Key": "user-key"},
        files={"file": ("captcha.png", image_bytes(), "image/png")},
    )
    assert response.status_code == 200


def test_bearer_only_client_key_rejected(service):
    response = service[0].post(
        "/predict",
        headers={"Authorization": "Bearer user-key"},
        files={"file": ("captcha.png", image_bytes(), "image/png")},
    )
    assert response.status_code == 401


def test_rate_limiting_before_expensive_work(service, monkeypatch):
    service[1].enforcement = True
    service[3]["rate_limits"] = {"predict": {"requests": 0, "window_seconds": 60}}
    invoked = []

    def forbidden(*args, **kwargs):
        invoked.append(True)
        raise AssertionError("expensive image or inference work ran")

    monkeypatch.setattr(UploadFile, "read", forbidden)
    monkeypatch.setattr(Image, "open", forbidden)
    monkeypatch.setattr(model_service, "predict", forbidden)
    response = post(service[0], {"x-api-key": "limited-key"})
    assert response.status_code == 429
    assert int(response.headers["retry-after"]) >= 1
    assert invoked == []


def test_service_credential_rejected_as_client_key(service):
    response = post(service[0], {"x-api-key": "test-service-credential-sentinel"})
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"
    assert not service[2]


def test_enforcement_false_uses_safety_limiter():
    from app.control_plane import KeyAccess

    control = ControlPlane(
        "https://control.invalid", "s" * 40, enforcement=False, global_limit=3
    )
    access = KeyAccess(KEY_ID, 30, None)
    results = [control.enforce_rate_limit(access) for _ in range(5)]
    assert sum(result is None for result in results) == 3
    assert all(result is None or result >= 1 for result in results)


def test_enforcement_true_missing_policy_fails_closed_without_global_charge():
    from app.control_plane import KeyAccess

    control = ControlPlane(
        "https://control.invalid", "s" * 40, enforcement=True, global_limit=1
    )
    with pytest.raises(ControlPlaneError):
        control.enforce_rate_limit(KeyAccess(KEY_ID, 30, None))
    control.enforcement = False
    assert control.enforce_rate_limit(KeyAccess(KEY_ID, 30, None)) is None


def test_zero_and_exhausted_policy_do_not_consume_global_quota():
    from app.control_plane import KeyAccess, RateLimit

    control = ControlPlane(
        "https://control.invalid", "s" * 40, enforcement=True, global_limit=1
    )
    denied = KeyAccess(KEY_ID, 30, RateLimit(0, 60))
    assert control.enforce_rate_limit(denied) >= 1
    control.enforcement = False
    assert control.enforce_rate_limit(KeyAccess(KEY_ID, 30, None)) is None


def test_concurrent_global_limiter_never_exceeds_budget():
    control = ControlPlane(
        "https://control.invalid", "s" * 40, enforcement=False, global_limit=25
    )
    access = KeyAccess(KEY_ID, 30, None)
    barrier = threading.Barrier(32)

    def admit(_):
        barrier.wait()
        return control.enforce_rate_limit(access)

    with ThreadPoolExecutor(max_workers=32) as pool:
        results = list(pool.map(admit, range(32)))
    assert sum(result is None for result in results) == 25
    assert all(result is None or result >= 1 for result in results)


def test_concurrent_per_key_limiter_and_key_isolation():
    control = ControlPlane(
        "https://control.invalid", "s" * 40, enforcement=True, global_limit=1000
    )
    first = KeyAccess("00000000-0000-7000-8000-000000000001", 30, RateLimit(10, 60))
    second = KeyAccess("00000000-0000-7000-8000-000000000002", 30, RateLimit(10, 60))
    barrier = threading.Barrier(40)
    accesses = [first, second] * 20

    def admit(access):
        barrier.wait()
        return access.key_id, control.enforce_rate_limit(access)

    with ThreadPoolExecutor(max_workers=40) as pool:
        results = list(pool.map(admit, accesses))
    first_results = [result for key_id, result in results if key_id == first.key_id]
    second_results = [result for key_id, result in results if key_id == second.key_id]
    assert sum(result is None for result in first_results) == 10
    assert sum(result is None for result in second_results) == 10


def test_global_rejection_does_not_charge_per_key_budget():
    control = ControlPlane(
        "https://control.invalid", "s" * 40, enforcement=True, global_limit=1
    )
    first = KeyAccess("00000000-0000-7000-8000-000000000001", 30, RateLimit(2, 60))
    second = KeyAccess("00000000-0000-7000-8000-000000000002", 30, RateLimit(2, 60))
    assert control.enforce_rate_limit(first) is None
    assert control.enforce_rate_limit(second) >= 1
    assert second.key_id not in control._key_windows
    control._global_window = None
    assert control.enforce_rate_limit(second) is None


def test_introspection_cache_is_bounded_and_prunes_expired_entries(monkeypatch):
    clock = [100.0]
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        value = active(key_id=str(uuid.uuid5(uuid.NAMESPACE_URL, str(calls))))
        return httpx.Response(200, json=value)

    async def scenario():
        monkeypatch.setattr("app.control_plane.time.monotonic", lambda: clock[0])
        control = ControlPlane(
            "https://control.invalid",
            SERVICE_CREDENTIAL,
            httpx.MockTransport(handler),
            cache_max_entries=3,
        )
        for index in range(10):
            await control.introspect(f"client-key-{index}", str(index))
            assert len(control.cache) <= 3
        active_digests = set(control.cache)
        clock[0] += 31
        await control.introspect("new-client-key", "prune")
        assert len(control.cache) == 1
        assert not active_digests & set(control.cache)
        await control.close()

    asyncio.run(scenario())
    assert calls == 11


def test_limiter_state_is_bounded_and_prunes_expired_windows(monkeypatch):
    now = [120]
    monkeypatch.setattr("app.control_plane.time.time", lambda: now[0])
    control = ControlPlane(
        "https://control.invalid",
        SERVICE_CREDENTIAL,
        enforcement=True,
        global_limit=1000,
        limiter_max_keys=3,
    )
    policy = RateLimit(10, 60)
    active_keys = [str(uuid.uuid4()) for _ in range(3)]
    for key_id in active_keys:
        assert control.enforce_rate_limit(KeyAccess(key_id, 30, policy)) is None
    assert len(control._key_windows) == 3
    assert control.enforce_rate_limit(KeyAccess(str(uuid.uuid4()), 30, policy)) >= 1
    assert len(control._key_windows) == 3
    now[0] += 60
    replacement = str(uuid.uuid4())
    assert control.enforce_rate_limit(KeyAccess(replacement, 30, policy)) is None
    assert control._key_windows == {replacement: (3, 1)}


def test_usage_single_flight_no_duplicate_posts_and_next_batch():
    started = asyncio.Event()
    release = asyncio.Event()
    in_flight = 0
    max_in_flight = 0
    payloads = []

    async def handler(request):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        payloads.append(json.loads(request.content))
        started.set()
        await release.wait()
        in_flight -= 1
        return httpx.Response(202, json={"accepted": True, "duplicate": False})

    async def scenario():
        control = ControlPlane(
            "https://control.invalid",
            "test-service-credential-sentinel",
            httpx.MockTransport(handler),
        )
        control.record_usage(KEY_ID, 200, 10)

        async def schedule(index):
            return control.schedule_usage_flush(str(index))

        tasks = await asyncio.gather(*(schedule(i) for i in range(20)))
        assert len({id(task) for task in tasks}) == 1
        await started.wait()
        next_key = "00000000-0000-7000-8000-000000000002"
        control.record_usage(next_key, 429, 20)
        release.set()
        await tasks[0]
        deadline = asyncio.get_running_loop().time() + 1
        while len(payloads) < 2 and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.005)
        with control._flush_task_lock:
            follow_up = control._flush_task
        if follow_up is not None:
            await follow_up
        assert max_in_flight == 1
        assert len({payload["batch_id"] for payload in payloads}) == 2
        assert payloads[1]["records"][0]["status_class"] == "4xx"
        await control.close()

    asyncio.run(scenario())


def test_failed_flush_keeps_identical_batch_without_task_pressure():
    payloads = []

    async def handler(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(503, json={"detail": "unavailable"})

    async def scenario():
        control = ControlPlane(
            "https://control.invalid",
            "test-service-credential-sentinel",
            httpx.MockTransport(handler),
            usage_retry_attempts=2,
            usage_retry_delay_seconds=0.01,
        )
        control.record_usage(KEY_ID, 200, 10)
        task = control.schedule_usage_flush("failure")
        assert task is not None
        await task
        assert payloads[0] == payloads[1] == control.pending_usage
        assert control.schedule_usage_flush("cooldown") is None
        assert control.pending_usage_count() == 1
        await control.close()

    asyncio.run(scenario())


def test_usage_queue_bound_counts_pending_and_live_duplicates(caplog):
    async def scenario():
        control = ControlPlane(
            "https://control.invalid",
            SERVICE_CREDENTIAL,
            usage_max_dimensions=3,
        )
        identifiers = [str(uuid.uuid4()) for _ in range(3)]
        for key_id in identifiers:
            control.record_usage(key_id, 200, 1)
        pending = await control.prepare_usage_batch()
        assert len(pending["records"]) == 3
        for key_id in identifiers:
            control.record_usage(key_id, 200, 1)
        for _ in range(10):
            control.record_usage(str(uuid.uuid4()), 200, 1)
        assert len(control._known_usage_dimensions) == 3
        assert len(control.usage) == 3
        assert control.pending_usage_count() == 6
        assert control.max_usage_record_slots == 6
        assert control.usage_overflow_count == 10
        assert {record["request_count"] for record in control.usage.values()} == {1}
        await control.close()

    asyncio.run(scenario())
    assert "max_unique_dimensions=3" in caplog.text
    assert "max_usage_record_slots=6" in caplog.text
    assert KEY_ID not in caplog.text


def test_usage_queue_default_object_ceiling_includes_pending_batch():
    control = ControlPlane("https://control.invalid", SERVICE_CREDENTIAL)
    assert control._usage_max_dimensions == 2000
    assert MAX_BATCH_RECORDS == 500
    assert control.max_usage_record_slots == 2500


def test_real_predict_response_does_not_wait_for_usage_http(monkeypatch):
    usage_started = threading.Event()
    release = threading.Event()

    async def handler(request):
        if request.url.path.endswith("/introspect"):
            return httpx.Response(200, json=active())
        usage_started.set()
        while not release.is_set():
            await asyncio.sleep(0.005)
        return httpx.Response(202, json={"accepted": True, "duplicate": False})

    monkeypatch.setattr(model_service, "predict", lambda image: "ABCDE")
    control = ControlPlane(
        "https://control.invalid",
        SERVICE_CREDENTIAL,
        httpx.MockTransport(handler),
    )
    with TestClient(create_app(control)) as client:
        started = time.monotonic()
        result = post(client, {"X-API-Key": "user-key"})
        elapsed = time.monotonic() - started
        assert result.status_code == 200
        assert elapsed < 0.5
        assert usage_started.wait(timeout=1)
        release.set()


def test_usage_stream_limit_is_transient_and_stops_reading():
    sentinel = b"usage-response-secret-sentinel"
    stream = ChunkStream([b"{", b"x" * MAX_RESPONSE_BYTES, sentinel])

    async def handler(request):
        return httpx.Response(202, stream=stream)

    async def scenario():
        control = ControlPlane(
            "https://control.invalid",
            SERVICE_CREDENTIAL,
            httpx.MockTransport(handler),
            usage_retry_attempts=1,
        )
        control.record_usage(KEY_ID, 200, 1)
        assert await control.flush_usage("large") is UsageFlushOutcome.TRANSIENT
        assert control.pending_usage_count() == 1
        await control.close()

    asyncio.run(scenario())
    assert stream.read_chunks == 2
    assert stream.closed


def test_shutdown_progresses_past_permanent_batch():
    payloads = []

    async def handler(request):
        payloads.append(json.loads(request.content))
        if len(payloads) == 1:
            return httpx.Response(400, json={"detail": "invalid"})
        return httpx.Response(202, json={"accepted": True, "duplicate": False})

    async def scenario():
        control = ControlPlane(
            "https://control.invalid",
            SERVICE_CREDENTIAL,
            httpx.MockTransport(handler),
            usage_max_dimensions=2,
        )
        control.record_usage(KEY_ID, 200, 1)
        await control.prepare_usage_batch()
        control.record_usage("00000000-0000-7000-8000-000000000002", 200, 1)
        assert await control.shutdown_usage()
        assert control.usage_dropped_batches_count == 1
        await control.close()

    asyncio.run(scenario())
    assert len(payloads) == 2
    assert payloads[0]["batch_id"] != payloads[1]["batch_id"]


def test_shutdown_drains_multiple_batches_and_closes_after_drain():
    payloads = []
    closed_after = []

    async def handler(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(202, json={"accepted": True, "duplicate": False})

    async def scenario():
        control = ControlPlane(
            "https://control.invalid",
            "test-service-credential-sentinel",
            httpx.MockTransport(handler),
            usage_max_dimensions=700,
        )
        for index in range(600):
            control.record_usage(str(uuid.uuid4()), 200, 1)
        assert await control.shutdown_usage()
        closed_after.append(control.pending_usage_count())
        await control.close()

    asyncio.run(scenario())
    assert [len(payload["records"]) for payload in payloads] == [500, 100]
    assert closed_after == [0]


def test_shutdown_is_bounded_when_usage_endpoint_stays_down(caplog):
    async def handler(request):
        await asyncio.sleep(10)
        return httpx.Response(503)

    async def scenario():
        control = ControlPlane(
            "https://control.invalid",
            "test-service-credential-sentinel",
            httpx.MockTransport(handler),
            usage_retry_attempts=1,
            usage_shutdown_timeout_seconds=0.1,
        )
        control.record_usage(KEY_ID, 200, 1)
        started = asyncio.get_running_loop().time()
        assert not await control.shutdown_usage()
        assert asyncio.get_running_loop().time() - started < 0.5
        await control.close()

    asyncio.run(scenario())
    assert "remaining_records=1" in caplog.text


def test_actual_lifespan_drains_inflight_and_multiple_batches(monkeypatch):
    usage_started = threading.Event()
    release = threading.Event()
    payloads = []
    closed_with = []

    async def handler(request):
        if request.url.path.endswith("/introspect"):
            return httpx.Response(200, json=active())
        payloads.append(json.loads(request.content))
        if len(payloads) == 1:
            usage_started.set()
            while not release.is_set():
                await asyncio.sleep(0.005)
        return httpx.Response(202, json={"accepted": True, "duplicate": False})

    control = ControlPlane(
        "https://control.invalid",
        "test-service-credential-sentinel",
        httpx.MockTransport(handler),
        usage_max_dimensions=700,
    )
    original_close = control.close

    async def close_spy():
        closed_with.append(control.pending_usage_count())
        await original_close()

    control.close = close_spy
    monkeypatch.setattr(model_service, "predict", lambda image: "ABCDE")
    with TestClient(create_app(control)) as client:
        assert post(client, {"X-API-Key": "user-key"}).status_code == 200
        assert usage_started.wait(timeout=1)
        for _ in range(600):
            control.record_usage(str(uuid.uuid4()), 200, 1)
        release.set()
    assert closed_with == [0]
    assert sum(len(payload["records"]) for payload in payloads) == 601
    assert max(len(payload["records"]) for payload in payloads) <= 500


def test_actual_lifespan_failure_is_bounded_and_closes_after_drain_attempt(
    monkeypatch, caplog
):
    closed_with = []

    async def handler(request):
        if request.url.path.endswith("/introspect"):
            return httpx.Response(200, json=active())
        await asyncio.sleep(10)
        return httpx.Response(503)

    control = ControlPlane(
        "https://control.invalid",
        "test-service-credential-sentinel",
        httpx.MockTransport(handler),
        usage_retry_attempts=1,
        usage_shutdown_timeout_seconds=0.1,
    )
    original_close = control.close

    async def close_spy():
        closed_with.append(control.pending_usage_count())
        await original_close()

    control.close = close_spy
    monkeypatch.setattr(model_service, "predict", lambda image: "ABCDE")
    started = time.monotonic()
    with TestClient(create_app(control)) as client:
        assert post(client, {"X-API-Key": "user-key"}).status_code == 200
    assert time.monotonic() - started < 0.5
    assert closed_with == [1]
    assert "remaining_records=1" in caplog.text


def test_actual_lifespan_shutdown_exception_still_closes_client():
    events = []

    class LifecycleControl:
        async def shutdown_usage(self):
            events.append("shutdown")
            raise RuntimeError("shutdown failed")

        async def close(self):
            events.append("close")

    async def scenario():
        application = create_app(LifecycleControl())
        with pytest.raises(RuntimeError, match="shutdown failed"):
            async with application.router.lifespan_context(application):
                pass

    asyncio.run(scenario())
    assert events == ["shutdown", "close"]


def test_actual_lifespan_outer_cancellation_cleans_up_and_propagates():
    events = []
    cleanup_started = asyncio.Event()
    release = asyncio.Event()

    class LifecycleControl:
        async def shutdown_usage(self):
            events.append("shutdown")
            cleanup_started.set()
            await release.wait()
            events.append("drained")

        async def close(self):
            events.append("close")

    async def scenario():
        application = create_app(LifecycleControl())

        async def run_lifespan():
            async with application.router.lifespan_context(application):
                await asyncio.Event().wait()

        task = asyncio.create_task(run_lifespan())
        await asyncio.sleep(0)
        task.cancel()
        await cleanup_started.wait()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert events == ["shutdown", "drained", "close"]


def test_service_credential_fixed_digest_equal_and_different_lengths():
    credential = "test-credential-value-1234567890"
    control = ControlPlane("https://control.invalid", credential)
    assert control.is_service_credential(credential)
    assert not control.is_service_credential("x" * len(credential))
    assert not control.is_service_credential(credential + "x")
    assert not control.is_service_credential("")


@pytest.mark.parametrize(
    "name,value",
    [
        ("API_KEY_RATE_LIMIT_ENFORCEMENT", "maybe"),
        ("API_KEY_RATE_LIMIT_ENFORCEMENT", " true"),
        ("PREDICT_GLOBAL_RATE_LIMIT", "0"),
        ("PREDICT_GLOBAL_RATE_LIMIT", "1000001"),
        ("USAGE_MAX_PENDING_DIMENSIONS", "0"),
        ("USAGE_MAX_PENDING_DIMENSIONS", "100001"),
        ("USAGE_RECORD_MAX_AGE_SECONDS", "59"),
        ("USAGE_RECORD_MAX_AGE_SECONDS", "82801"),
        ("USAGE_RETRY_ATTEMPTS", "0"),
        ("USAGE_RETRY_ATTEMPTS", "11"),
        ("USAGE_RETRY_DELAY_SECONDS", "0"),
        ("USAGE_RETRY_DELAY_SECONDS", "nan"),
        ("USAGE_SHUTDOWN_TIMEOUT_SECONDS", "0"),
        ("USAGE_SHUTDOWN_TIMEOUT_SECONDS", "61"),
        ("AUTH_CACHE_MAX_TTL_SECONDS", "0"),
        ("AUTH_CACHE_MAX_TTL_SECONDS", "31"),
        ("AUTH_CACHE_MAX_ENTRIES", "0"),
        ("PREDICT_MAX_TRACKED_KEYS", "0"),
    ],
)
def test_invalid_settings_fail_startup(monkeypatch, name, value):
    monkeypatch.setenv("CONTROL_PLANE_BASE_URL", "https://control.invalid")
    monkeypatch.setenv("SERVICE_CREDENTIAL", "test-service-credential-sentinel")
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        with TestClient(create_app()):
            pass


# ---------------------------------------------------------------------------
# Base-URL validation tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url,label",
    INVALID_URLS,
    ids=[tag for _, tag in INVALID_URLS],
)
def test_strict_base_url_rejects_unsafe_forms(url, label):
    with pytest.raises(ValueError, match="CONTROL_PLANE_BASE_URL"):
        ControlPlane._validate_base_url(url)


def test_strict_base_url_rejects_unsafe_forms_no_client_built():
    for url, _ in INVALID_URLS:
        try:
            ControlPlane(url, SERVICE_CREDENTIAL)
        except ValueError:
            pass
    for url, _ in INVALID_URLS:
        with pytest.raises(ValueError, match="CONTROL_PLANE_BASE_URL"):
            ControlPlane(url, SERVICE_CREDENTIAL)


def test_strict_base_url_valid_forms():
    valid_urls = [
        ("http://127.0.0.1:18200", "http://127.0.0.1:18200/"),
        ("http://127.0.0.1:18200/prefix", "http://127.0.0.1:18200/prefix"),
        ("http://127.0.0.1:80/prefix", "http://127.0.0.1/prefix"),
        ("https://host.invalid:443/", "https://host.invalid/"),
        ("http://[::1]:18200/path", "http://[::1]:18200/path"),
        ("HTTP://HOST.INVALID:80/", "http://host.invalid/"),
    ]
    for url, expected in valid_urls:
        canonical = ControlPlane._validate_base_url(url)
        assert canonical == expected, f"{url!r} -> {canonical!r}"


def test_strict_base_url_generates_correct_endpoints():
    canonical = ControlPlane._validate_base_url("https://cp.example.com/v1")
    assert canonical.endswith("/v1")
    introspect = canonical + ControlPlane._INTROSPECT_PATH
    usage = canonical + ControlPlane._USAGE_PATH
    assert introspect != usage
    assert ControlPlane._INTROSPECT_PATH.startswith("/internal/")
    assert ControlPlane._USAGE_PATH.startswith("/internal/")


def test_strict_base_url_no_submitted_url_in_error():
    secret = "super-secret-credential"
    for url, _ in INVALID_URLS:
        with pytest.raises(ValueError) as exc_info:
            ControlPlane._validate_base_url(url)
        error_text = str(exc_info.value)
        assert url not in error_text
        assert secret not in error_text


def test_strict_base_url_validation_before_client_construction():
    for url, _ in INVALID_URLS:
        with pytest.raises(ValueError):
            ControlPlane(url, SERVICE_CREDENTIAL)


def test_strict_base_url_all_encoded_dot_patterns_rejected():
    encoded_dot_cases = [
        "http://127.0.0.1:18200/%2e/",
        "http://127.0.0.1:18200/%2E/",
        "http://127.0.0.1:18200/%2e%2e/",
        "http://127.0.0.1:18200/%2E%2E/",
        "http://127.0.0.1:18200/.%2e/",
        "http://127.0.0.1:18200/%2e./",
        "http://127.0.0.1:18200/.%2E/",
        "http://127.0.0.1:18200/%2E./",
        "http://127.0.0.1:18200/%2e%2e/admin",
        "http://127.0.0.1:18200/%2e",
        "http://127.0.0.1:18200/%2e%2e",
        "http://127.0.0.1:18200/%252e/",
        "http://127.0.0.1:18200/%252e%252e/",
        "http://127.0.0.1:18200/%252E%252E/",
        "http://127.0.0.1:18200/%25252e%25252e/",
        "http://127.0.0.1:18200/.%252e/",
        "http://127.0.0.1:18200/%252e./",
        "http://127.0.0.1:18200/%252e%252e/admin",
        "http://127.0.0.1:18200/%252e%252e",
    ]
    for url in encoded_dot_cases:
        with pytest.raises(ValueError, match="CONTROL_PLANE_BASE_URL"):
            ControlPlane._validate_base_url(url)
        with pytest.raises(ValueError):
            ControlPlane(url, SERVICE_CREDENTIAL)


def test_strict_base_url_valid_dotted_paths_preserved():
    for url, expected in VALID_DOTTED_PATHS:
        canonical = ControlPlane._validate_base_url(url)
        assert canonical == expected, f"{url!r} -> {canonical!r}"


def test_strict_base_url_parity_with_control_plane_strict_semantics():
    parity_rejections = [
        "http://user:pass@127.0.0.1:18200",
        "http://127.0.0.1:18200?q=1",
        "http://127.0.0.1:18200#frag",
        "ftp://127.0.0.1:18200",
        "http://127.0.0.1:99999",
        "http://127.0.0.1:18200/%2e%2e/",
        "http://127.0.0.1:18200/%2f",
        "http://127.0.0.1:18200/%5c",
        "http://127.0.0.1:18200/%40",
    ]
    for url in parity_rejections:
        with pytest.raises(ValueError, match="CONTROL_PLANE_BASE_URL"):
            ControlPlane._validate_base_url(url)


def test_strict_base_url_decode_depth_and_fail_closed():
    depth_cases = [
        ("http://127.0.0.1:18200/%252e/", "double_dot"),
        ("http://127.0.0.1:18200/%252e%252e/", "double_dotdot"),
        ("http://127.0.0.1:18200/%25252e%25252e/", "triple_dotdot"),
        ("http://127.0.0.1:18200/%252525252e", "four_level_dot_fail_closed"),
    ]
    for url, label in depth_cases:
        with pytest.raises(ValueError, match="CONTROL_PLANE_BASE_URL"):
            ControlPlane._validate_base_url(url)
        with pytest.raises(ValueError):
            ControlPlane(url, SERVICE_CREDENTIAL)


def test_strict_base_url_malformed_revealed_after_decode():
    malformed_cases = [
        "http://127.0.0.1:18200/%25zz",
        "http://127.0.0.1:18200/%252",
        "http://127.0.0.1:18200/%252e%2",
    ]
    for url in malformed_cases:
        with pytest.raises(ValueError, match="CONTROL_PLANE_BASE_URL"):
            ControlPlane._validate_base_url(url)


def test_strict_base_url_nested_encoded_dangerous_chars():
    nested_cases = [
        ("http://127.0.0.1:18200/%252f", "double_encoded_slash"),
        ("http://127.0.0.1:18200/%255c", "double_encoded_backslash"),
        ("http://127.0.0.1:18200/%253a", "double_encoded_colon"),
        ("http://127.0.0.1:18200/%2540", "double_encoded_at"),
    ]
    for url, label in nested_cases:
        with pytest.raises(ValueError, match="CONTROL_PLANE_BASE_URL"):
            ControlPlane._validate_base_url(url)


def test_strict_base_url_no_rejected_url_in_exception_or_log(caplog):
    caplog.set_level(logging.DEBUG)
    malicious_urls = [
        "http://127.0.0.1:18200/%252e%252e/admin",
        "http://127.0.0.1:18200/%252f",
        "http://127.0.0.1:18200/%253a",
    ]
    for url in malicious_urls:
        with pytest.raises(ValueError) as exc_info:
            ControlPlane._validate_base_url(url)
        error_text = str(exc_info.value)
        assert url not in error_text
        assert "%252e" not in error_text
        assert "%252f" not in error_text
    assert SERVICE_CREDENTIAL not in caplog.text


# ---------------------------------------------------------------------------
# Usage structure churn tests
# ---------------------------------------------------------------------------


def test_usage_churn_after_success_resets_pending_and_known():
    async def scenario():
        control = ControlPlane(
            "https://control.invalid",
            SERVICE_CREDENTIAL,
            usage_max_dimensions=5,
        )
        for i in range(8):
            control.record_usage(str(i), 200, 1)
        batch = await control.prepare_usage_batch()
        assert batch is not None
        assert control.pending_usage_count() == len(batch["records"])
        pending_keys = set(control._pending_dimension_keys)
        assert pending_keys
        for i in range(3):
            control.record_usage(f"new-{i}", 200, 1)
        assert control.pending_usage_count() > 0

        async def handler(request):
            return httpx.Response(202, json={"accepted": True, "duplicate": False})

        control.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result = await control.flush_usage("test")
        assert result is UsageFlushOutcome.ACCEPTED
        assert control.pending_usage is None
        assert not control._pending_dimension_keys
        assert control._pending_auth_attempts == 0
        assert control.pending_usage_count() == len(control.usage)
        await control.close()

    asyncio.run(scenario())


def test_usage_churn_after_dead_letter_resets_pending_and_known():
    async def scenario():
        control = ControlPlane(
            "https://control.invalid",
            SERVICE_CREDENTIAL,
            usage_max_dimensions=5,
        )
        for i in range(8):
            control.record_usage(str(i), 200, 1)
        batch = await control.prepare_usage_batch()
        assert batch is not None
        for i in range(3):
            control.record_usage(f"new-{i}", 429, 1)

        async def handler(request):
            return httpx.Response(400, json={"detail": "invalid"})

        control.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result = await control.flush_usage("test")
        assert result is UsageFlushOutcome.DEAD_LETTERED
        assert control.pending_usage is None
        assert not control._pending_dimension_keys
        assert control._pending_auth_attempts == 0
        assert control.pending_usage_count() == len(control.usage)
        await control.close()

    asyncio.run(scenario())


def test_usage_churn_after_stale_prune_removes_old_dimensions():
    async def scenario():
        control = ControlPlane(
            "https://control.invalid",
            SERVICE_CREDENTIAL,
            usage_max_dimensions=5,
            usage_record_max_age_seconds=60,
        )
        control.record_usage(KEY_ID, 200, 1)
        record = next(iter(control.usage.values()))
        record["bucket_start"] = (
            datetime.now(timezone.utc) - timedelta(seconds=61)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        result = await control.flush_usage("stale")
        assert result is UsageFlushOutcome.IDLE
        assert control.pending_usage is None
        assert not control._pending_dimension_keys
        assert control.pending_usage_count() == 0
        await control.close()

    asyncio.run(scenario())


def test_usage_overflow_returns_to_bounded_dimensions():
    control = ControlPlane(
        "https://control.invalid",
        SERVICE_CREDENTIAL,
        usage_max_dimensions=3,
    )
    for _ in range(50):
        control.record_usage(str(uuid.uuid4()), 200, 1)
    assert len(control._known_usage_dimensions) == 3
    assert len(control.usage) == 3
    assert control.usage_overflow_count == 47
    control.usage.clear()
    assert control.pending_usage_count() == 0
