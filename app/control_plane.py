import asyncio
import hashlib
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

SCOPE = "captcha:predict"
TIMEOUT = httpx.Timeout(2.0, connect=1.0)


class ControlPlaneError(Exception):
    pass


@dataclass(frozen=True)
class RateLimit:
    requests: int
    window_seconds: int


@dataclass(frozen=True)
class KeyAccess:
    key_id: str
    cache_ttl_seconds: int
    rate_limit: RateLimit | None


class ControlPlane:
    def __init__(
        self,
        base_url: str,
        credential: str,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not base_url or not credential:
            raise ValueError(
                "CONTROL_PLANE_BASE_URL dan SERVICE_CREDENTIAL wajib diisi."
            )
        try:
            url = httpx.URL(base_url)
        except Exception as exc:
            raise ValueError("CONTROL_PLANE_BASE_URL tidak valid.") from exc
        if url.scheme not in {"http", "https"} or not url.host:
            raise ValueError("CONTROL_PLANE_BASE_URL harus berupa URL HTTP(S) absolut.")
        if any(character.isspace() for character in credential):
            raise ValueError("SERVICE_CREDENTIAL tidak valid.")
        self.base_url = base_url.rstrip("/")
        self.credential = credential
        self.client = httpx.AsyncClient(timeout=TIMEOUT, transport=transport)
        self.cache: dict[str, tuple[float, KeyAccess]] = {}
        # ponytail: limiter process-local; pindahkan ke Redis sebelum multi-worker.
        self.windows: dict[str, tuple[int, int]] = {}
        self.usage: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.pending_usage: dict[str, Any] | None = None
        self._usage_lock = asyncio.Lock()

    async def close(self) -> None:
        await self.client.aclose()

    def _headers(self, request_id: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.credential}",
            "X-Request-ID": request_id,
        }

    async def introspect(self, raw_key: str, request_id: str) -> KeyAccess | None:
        digest = hashlib.sha256(raw_key.encode()).hexdigest()
        cached = self.cache.get(digest)
        now = time.monotonic()
        if cached and cached[0] > now:
            return cached[1]
        self.cache.pop(digest, None)
        try:
            response = await self.client.post(
                f"{self.base_url}/internal/v1/keys/introspect",
                headers=self._headers(request_id),
                json={"api_key": raw_key, "required_scope": SCOPE},
            )
            if response.status_code != 200:
                raise ControlPlaneError
            payload = response.json()
            access = self._validate_introspection(payload)
        except (httpx.HTTPError, ValueError, TypeError, ControlPlaneError) as exc:
            raise ControlPlaneError from exc
        if access is None:
            return None
        if access.cache_ttl_seconds:
            self.cache[digest] = (now + access.cache_ttl_seconds, access)
        return access

    @staticmethod
    def _validate_introspection(payload: Any) -> KeyAccess | None:
        if type(payload) is not dict or type(payload.get("active")) is not bool:
            raise ControlPlaneError
        ttl = payload.get("cache_ttl_seconds")
        if type(ttl) is not int or not 0 <= ttl <= 30:
            raise ControlPlaneError
        if not payload["active"]:
            if ttl != 0:
                raise ControlPlaneError
            return None
        key_id = payload.get("key_id")
        scopes = payload.get("scopes")
        expires_at = payload.get("expires_at")
        if type(key_id) is not str or type(scopes) is not list:
            raise ControlPlaneError
        try:
            if str(uuid.UUID(key_id)) != key_id:
                raise ValueError
        except ValueError as exc:
            raise ControlPlaneError from exc
        if (
            not scopes
            or any(type(scope) is not str for scope in scopes)
            or SCOPE not in scopes
        ):
            raise ControlPlaneError
        if expires_at is not None:
            if type(expires_at) is not str:
                raise ControlPlaneError
            try:
                expiry = datetime.strptime(expires_at, "%Y-%m-%dT%H:%M:%SZ").replace(
                    tzinfo=timezone.utc
                )
            except ValueError as exc:
                raise ControlPlaneError from exc
            remaining = int((expiry - datetime.now(timezone.utc)).total_seconds())
            if remaining <= 0:
                raise ControlPlaneError
            ttl = min(ttl, remaining)
        rate_limit = ControlPlane._validate_rate_limit(payload)
        return KeyAccess(key_id, min(ttl, 30), rate_limit)

    @staticmethod
    def _validate_rate_limit(payload: dict[str, Any]) -> RateLimit | None:
        if "rate_limits" not in payload:
            return None
        policies = payload["rate_limits"]
        if type(policies) is not dict or set(policies) != {"predict"}:
            raise ControlPlaneError
        policy = policies["predict"]
        if type(policy) is not dict or set(policy) != {"requests", "window_seconds"}:
            raise ControlPlaneError
        requests = policy["requests"]
        window = policy["window_seconds"]
        if type(requests) is not int or not 0 <= requests <= 10_000 or window != 60:
            raise ControlPlaneError
        return RateLimit(requests, window)

    def enforce_rate_limit(self, access: KeyAccess) -> int | None:
        policy = access.rate_limit
        if policy is None:
            return None
        now = int(time.time())
        window = now // policy.window_seconds
        previous_window, count = self.windows.get(access.key_id, (window, 0))
        if previous_window != window:
            count = 0
        if policy.requests == 0 or count >= policy.requests:
            return max(1, (window + 1) * policy.window_seconds - now)
        self.windows[access.key_id] = (window, count + 1)
        return None

    def record_usage(self, key_id: str, status_code: int, latency_ms: int) -> None:
        bucket = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        bucket_start = bucket.strftime("%Y-%m-%dT%H:%M:%SZ")
        status_class = f"{status_code // 100}xx"
        key = (bucket_start, key_id, status_class)
        record = self.usage.setdefault(
            key,
            {
                "bucket_start": bucket_start,
                "key_id": key_id,
                "endpoint_template": "/predict",
                "method": "POST",
                "status_class": status_class,
                "request_count": 0,
                "error_count": 0,
                "latency_sum_ms": 0,
                "latency_max_ms": 0,
            },
        )
        record["request_count"] += 1
        record["error_count"] += status_code >= 400
        record["latency_sum_ms"] += latency_ms
        record["latency_max_ms"] = max(record["latency_max_ms"], latency_ms)

    async def flush_usage(self, request_id: str) -> None:
        # ponytail: antrean mati bersama process; gunakan durable queue untuk recovery.
        async with self._usage_lock:
            if self.pending_usage is None and self.usage:
                records = list(self.usage.values())[:500]
                selected = {
                    (item["bucket_start"], item["key_id"], item["status_class"])
                    for item in records
                }
                self.usage = {
                    key: value
                    for key, value in self.usage.items()
                    if key not in selected
                }
                self.pending_usage = {
                    "batch_id": str(uuid.uuid4()),
                    "records": records,
                }
            if self.pending_usage is None:
                return
            try:
                response = await self.client.post(
                    f"{self.base_url}/internal/v1/usage/batches",
                    headers=self._headers(request_id),
                    json=self.pending_usage,
                )
                payload = response.json()
                if response.status_code not in {200, 202}:
                    return
                if (
                    type(payload) is not dict
                    or payload.get("accepted") is not True
                    or type(payload.get("duplicate")) is not bool
                ):
                    return
            except (httpx.HTTPError, ValueError, TypeError):
                return
            self.pending_usage = None
