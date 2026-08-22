import asyncio
import hashlib
import hmac
import json
import logging
import math
import re
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any
from urllib.parse import urlsplit

import httpx

SCOPE = "captcha:predict"
TIMEOUT = httpx.Timeout(2.0, connect=1.0)
MAX_RESPONSE_BYTES = 64 * 1024
MAX_BATCH_RECORDS = 500
MAX_COUNTER = 2**63 - 1
STATE_PRUNE_LIMIT = 32
DEFAULT_GLOBAL_LIMIT = 1000
DEFAULT_CACHE_MAX_ENTRIES = 4096
DEFAULT_LIMITER_MAX_KEYS = 10_000
DEFAULT_USAGE_MAX_DIMENSIONS = 2000
DEFAULT_USAGE_RECORD_MAX_AGE_SECONDS = 23 * 60 * 60
DEFAULT_USAGE_RETRY_ATTEMPTS = 3
DEFAULT_USAGE_RETRY_DELAY_SECONDS = 0.1
DEFAULT_USAGE_SHUTDOWN_TIMEOUT_SECONDS = 5.0
logger = logging.getLogger(__name__)


class ControlPlaneError(Exception):
    pass


class NetworkError(ControlPlaneError):
    """Control plane unreachable or connection/read failure."""


class CredentialError(ControlPlaneError):
    """Control plane rejected our service credential (HTTP 401/403)."""


class ResponseStatusError(ControlPlaneError):
    """Control plane replied with an unexpected HTTP status."""


class ResponseSchemaError(ControlPlaneError):
    """Control plane replied with an unexpected or malformed body."""


class RateLimitPolicyMissing(ControlPlaneError):
    """Enforcement is active but the introspection carries no predict policy."""


class ResponseTooLarge(ResponseSchemaError):
    pass


class UsageFlushOutcome(str, Enum):
    IDLE = "idle"
    ACCEPTED = "accepted"
    TRANSIENT = "transient"
    DEAD_LETTERED = "dead_lettered"


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
    _MAX_BASE_URL_LEN = 500
    _MAX_PATH_LEN = 256
    _MAX_DECODE_DEPTH = 4
    _INTROSPECT_PATH = "/internal/v2/keys/introspect"
    _USAGE_PATH = "/internal/v1/usage/batches"

    @staticmethod
    def _validate_base_url(raw: str) -> str:
        """Strict stdlib-only URL validation matching the control-plane service
        URL validator.  Returns the canonical base (scheme://host[:port][/path])
        or raises ValueError.  No HTTP client is built on failure."""
        if not isinstance(raw, str) or len(raw) > ControlPlane._MAX_BASE_URL_LEN:
            raise ValueError("CONTROL_PLANE_BASE_URL tidak valid.")
        if raw != raw.strip() or any(ord(c) < 33 for c in raw):
            raise ValueError("CONTROL_PLANE_BASE_URL tidak valid.")
        if re.search(r"%(?:2[fF]|3[aA]|40|5[cC])", raw):
            raise ValueError("CONTROL_PLANE_BASE_URL tidak valid.")
        parts = urlsplit(raw)
        if parts.scheme not in ("http", "https"):
            raise ValueError("CONTROL_PLANE_BASE_URL harus berupa URL HTTP(S) absolut.")
        if parts.username or parts.password or "@" in parts.netloc:
            raise ValueError("CONTROL_PLANE_BASE_URL tidak valid.")
        if parts.query or parts.fragment:
            raise ValueError("CONTROL_PLANE_BASE_URL tidak valid.")
        host = (parts.hostname or "").lower()
        if not host or any(c.isspace() for c in host):
            raise ValueError("CONTROL_PLANE_BASE_URL tidak valid.")
        if "\\" in parts.netloc or "\\" in parts.path:
            raise ValueError("CONTROL_PLANE_BASE_URL tidak valid.")
        try:
            explicit_port = parts.port
        except ValueError:
            raise ValueError("CONTROL_PLANE_BASE_URL tidak valid.") from None
        if explicit_port is not None and not 0 < explicit_port < 65536:
            raise ValueError("CONTROL_PLANE_BASE_URL tidak valid.")
        try:
            host.encode("ascii")
        except UnicodeEncodeError:
            raise ValueError("CONTROL_PLANE_BASE_URL tidak valid.") from None
        path = parts.path.rstrip("/") or "/"
        if len(path) > ControlPlane._MAX_PATH_LEN:
            raise ValueError("CONTROL_PLANE_BASE_URL tidak valid.")
        segments = path.split("/")
        for seg in segments:
            current = seg
            if current in (".", ".."):
                raise ValueError("CONTROL_PLANE_BASE_URL tidak valid.")
            for _ in range(ControlPlane._MAX_DECODE_DEPTH):
                if "%" not in current:
                    break
                decoded_parts = bytearray()
                i = 0
                while i < len(current):
                    if current[i] == "%":
                        if i + 2 >= len(current):
                            raise ValueError("CONTROL_PLANE_BASE_URL tidak valid.")
                        try:
                            decoded_parts.append(int(current[i + 1 : i + 3], 16))
                        except ValueError:
                            raise ValueError(
                                "CONTROL_PLANE_BASE_URL tidak valid."
                            ) from None
                        i += 3
                    else:
                        decoded_parts.append(ord(current[i]))
                        i += 1
                try:
                    decoded = decoded_parts.decode("ascii")
                except UnicodeDecodeError:
                    raise ValueError("CONTROL_PLANE_BASE_URL tidak valid.") from None
                if decoded in (".", ".."):
                    raise ValueError("CONTROL_PLANE_BASE_URL tidak valid.")
                if any(c in decoded for c in "/\\:@"):
                    raise ValueError("CONTROL_PLANE_BASE_URL tidak valid.")
                if any(ord(c) < 32 or ord(c) == 127 for c in decoded):
                    raise ValueError("CONTROL_PLANE_BASE_URL tidak valid.")
                if decoded == current:
                    break
                current = decoded
            else:
                raise ValueError("CONTROL_PLANE_BASE_URL tidak valid.")
        if "//" in path:
            raise ValueError("CONTROL_PLANE_BASE_URL tidak valid.")
        default_port = 80 if parts.scheme == "http" else 443
        if not explicit_port or explicit_port == default_port:
            port_label = ""
        else:
            port_label = f":{explicit_port}"
        display_host = f"[{host}]" if ":" in host else host
        canonical = f"{parts.scheme}://{display_host}{port_label}{path}"
        return canonical

    def __init__(
        self,
        base_url: str,
        credential: str,
        transport: httpx.AsyncBaseTransport | None = None,
        cache_max_ttl_seconds: int = 30,
        enforcement: bool = False,
        global_limit: int = DEFAULT_GLOBAL_LIMIT,
        usage_max_dimensions: int = DEFAULT_USAGE_MAX_DIMENSIONS,
        usage_record_max_age_seconds: int = DEFAULT_USAGE_RECORD_MAX_AGE_SECONDS,
        usage_retry_attempts: int = DEFAULT_USAGE_RETRY_ATTEMPTS,
        usage_retry_delay_seconds: float = DEFAULT_USAGE_RETRY_DELAY_SECONDS,
        usage_shutdown_timeout_seconds: float = DEFAULT_USAGE_SHUTDOWN_TIMEOUT_SECONDS,
        cache_max_entries: int = DEFAULT_CACHE_MAX_ENTRIES,
        limiter_max_keys: int = DEFAULT_LIMITER_MAX_KEYS,
    ) -> None:
        if not credential:
            raise ValueError(
                "CONTROL_PLANE_BASE_URL dan SERVICE_CREDENTIAL wajib diisi."
            )
        canonical_base = self._validate_base_url(base_url)
        if any(character.isspace() for character in credential):
            raise ValueError("SERVICE_CREDENTIAL tidak valid.")
        self._validate_range("AUTH_CACHE_MAX_TTL_SECONDS", cache_max_ttl_seconds, 1, 30)
        if type(enforcement) is not bool:
            raise ValueError("API_KEY_RATE_LIMIT_ENFORCEMENT must be true or false.")
        self._validate_range("PREDICT_GLOBAL_RATE_LIMIT", global_limit, 1, 1_000_000)
        self._validate_range(
            "USAGE_MAX_PENDING_DIMENSIONS", usage_max_dimensions, 1, 100_000
        )
        self._validate_range(
            "USAGE_RECORD_MAX_AGE_SECONDS",
            usage_record_max_age_seconds,
            60,
            DEFAULT_USAGE_RECORD_MAX_AGE_SECONDS,
        )
        self._validate_range("USAGE_RETRY_ATTEMPTS", usage_retry_attempts, 1, 10)
        self._validate_float_range(
            "USAGE_RETRY_DELAY_SECONDS", usage_retry_delay_seconds, 0.01, 5.0
        )
        self._validate_float_range(
            "USAGE_SHUTDOWN_TIMEOUT_SECONDS",
            usage_shutdown_timeout_seconds,
            0.1,
            60.0,
        )
        self._validate_range("AUTH_CACHE_MAX_ENTRIES", cache_max_entries, 1, 100_000)
        self._validate_range("PREDICT_MAX_TRACKED_KEYS", limiter_max_keys, 1, 1_000_000)
        self.base_url = canonical_base.rstrip("/")
        self.credential = credential
        self._credential_digest = hmac.digest(
            b"captcha-ai-model-client-check", credential.encode(), "sha256"
        )
        self.enforcement = enforcement
        self.cache_max_ttl_seconds = cache_max_ttl_seconds
        self.client = httpx.AsyncClient(
            timeout=TIMEOUT, transport=transport, follow_redirects=False
        )
        self.cache: dict[str, tuple[float, KeyAccess]] = {}
        self._cache_max_entries = cache_max_entries
        self._cache_prune_cursor = 0
        self._limiter_lock = threading.Lock()
        self._limiter_max_keys = limiter_max_keys
        self._global_limit = global_limit
        self._window_seconds = 60
        self._global_window: tuple[int, int] | None = None
        self._key_windows: dict[str, tuple[int, int]] = {}
        self._key_window_prune_cursor = 0
        self._usage_max_dimensions = usage_max_dimensions
        self._usage_record_max_age_seconds = usage_record_max_age_seconds
        self._usage_retry_attempts = usage_retry_attempts
        self._usage_retry_delay_seconds = usage_retry_delay_seconds
        self._usage_shutdown_timeout_seconds = usage_shutdown_timeout_seconds
        self._usage_lock = threading.Lock()
        self._send_lock = asyncio.Lock()
        # ponytail: usage is memory-only; process exit or prolonged credential
        # outage can lose unflushed records.  Auth failures (401/403) are
        # retried within a bounded budget then dead-lettered so they never
        # block newer telemetry.  Logs expose only safe counts/categories.
        self.usage: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.pending_usage: dict[str, Any] | None = None
        self._pending_dimension_keys: set[tuple[str, str, str]] = set()
        self._known_usage_dimensions: set[tuple[str, str, str]] = set()
        self._pending_auth_attempts = 0
        self.usage_overflow_count = 0
        self.usage_dropped_batches_count = 0
        self.usage_dropped_records_count = 0
        self.usage_failure_counts = {
            "stale": 0,
            "permanent_payload": 0,
            "authentication": 0,
            "permanent_response": 0,
            "response_schema": 0,
        }
        self._last_overflow_log = 0.0
        self._last_drop_log = 0.0
        self._last_failure_log = 0.0
        self._next_flush_allowed = 0.0
        self._flush_task: asyncio.Task | None = None
        self._flush_task_lock = threading.Lock()
        self._accept_flush_scheduling = True

    @staticmethod
    def _validate_range(name: str, value: int, minimum: int, maximum: int) -> None:
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be an integer from {minimum} to {maximum}.")

    @staticmethod
    def _validate_float_range(
        name: str, value: float, minimum: float, maximum: float
    ) -> None:
        if type(value) not in {int, float} or isinstance(value, bool):
            raise ValueError(f"{name} must be a number from {minimum} to {maximum}.")
        numeric = float(value)
        if not math.isfinite(numeric) or not minimum <= numeric <= maximum:
            raise ValueError(f"{name} must be a number from {minimum} to {maximum}.")

    @property
    def max_usage_record_slots(self) -> int:
        """Maximum number of usage record dict objects the queue can hold:
        live unique dimensions + immutable pending batch records."""
        return self._usage_max_dimensions + min(
            self._usage_max_dimensions, MAX_BATCH_RECORDS
        )

    async def close(self) -> None:
        await self.client.aclose()

    def is_service_credential(self, raw_key: str) -> bool:
        candidate = hmac.digest(
            b"captcha-ai-model-client-check", raw_key.encode(), "sha256"
        )
        return hmac.compare_digest(candidate, self._credential_digest)

    def _headers(self, request_id: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.credential}",
            "X-Request-ID": request_id,
        }

    @staticmethod
    async def _read_json_response(response: httpx.Response) -> Any:
        declared = response.headers.get("Content-Length")
        if declared is not None:
            try:
                declared_size = int(declared)
            except ValueError as exc:
                raise ResponseSchemaError from exc
            if declared_size < 0 or declared_size > MAX_RESPONSE_BYTES:
                raise ResponseTooLarge
        body = bytearray()
        async for chunk in response.aiter_bytes():
            if len(body) + len(chunk) > MAX_RESPONSE_BYTES:
                raise ResponseTooLarge
            body.extend(chunk)
        try:
            return json.loads(body)
        except (UnicodeDecodeError, ValueError, TypeError) as exc:
            raise ResponseSchemaError from exc

    def _prune_cache(self, now: float) -> None:
        keys = tuple(self.cache)
        if not keys:
            self._cache_prune_cursor = 0
            return
        start = self._cache_prune_cursor % len(keys)
        checked = min(STATE_PRUNE_LIMIT, len(keys))
        for offset in range(checked):
            digest = keys[(start + offset) % len(keys)]
            cached = self.cache.get(digest)
            if cached is not None and cached[0] <= now:
                self.cache.pop(digest, None)
        self._cache_prune_cursor = (start + checked) % max(len(self.cache), 1)

    def _cache_access(self, digest: str, access: KeyAccess, expires_at: float) -> None:
        if digest not in self.cache and len(self.cache) >= self._cache_max_entries:
            self.cache.pop(next(iter(self.cache)), None)
        self.cache[digest] = (expires_at, access)

    async def introspect(
        self,
        raw_key: str,
        request_id: str,
        method: str,
        endpoint_template: str,
    ) -> KeyAccess | None:
        digest = hashlib.sha256(
            b"\0".join((raw_key.encode(), method.encode(), endpoint_template.encode()))
        ).hexdigest()
        now = time.monotonic()
        self._prune_cache(now)
        cached = self.cache.get(digest)
        if cached and cached[0] > now:
            return cached[1]
        self.cache.pop(digest, None)
        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}{self._INTROSPECT_PATH}",
                headers=self._headers(request_id),
                json={
                    "api_key": raw_key,
                    "method": method,
                    "endpoint_template": endpoint_template,
                },
            ) as response:
                if response.is_redirect or response.status_code != 200:
                    if response.status_code in (401, 403):
                        raise CredentialError
                    raise ResponseStatusError
                payload = await self._read_json_response(response)
            access = self._validate_introspection(payload, self.cache_max_ttl_seconds)
        except httpx.HTTPError as exc:
            raise NetworkError from exc
        except (ValueError, TypeError) as exc:
            raise ResponseSchemaError from exc
        except ControlPlaneError:
            raise
        if access is None:
            return None
        if access.cache_ttl_seconds:
            self._cache_access(digest, access, now + access.cache_ttl_seconds)
        return access

    @staticmethod
    def _validate_introspection(
        payload: Any, cache_max_ttl: int = 30
    ) -> KeyAccess | None:
        if type(payload) is not dict or type(payload.get("active")) is not bool:
            raise ResponseSchemaError
        ttl = payload.get("cache_ttl_seconds")
        if type(ttl) is not int or not 0 <= ttl <= 30:
            raise ResponseSchemaError
        if not payload["active"]:
            if ttl != 0:
                raise ResponseSchemaError
            return None
        key_id = payload.get("key_id")
        scopes = payload.get("scopes")
        expires_at = payload.get("expires_at")
        if type(key_id) is not str or type(scopes) is not list:
            raise ResponseSchemaError
        try:
            if str(uuid.UUID(key_id)) != key_id:
                raise ValueError
        except ValueError as exc:
            raise ResponseSchemaError from exc
        if (
            not scopes
            or any(type(scope) is not str for scope in scopes)
            or SCOPE not in scopes
        ):
            raise ResponseSchemaError
        if expires_at is not None:
            if type(expires_at) is not str:
                raise ResponseSchemaError
            try:
                expiry = datetime.strptime(expires_at, "%Y-%m-%dT%H:%M:%SZ").replace(
                    tzinfo=timezone.utc
                )
            except ValueError as exc:
                raise ResponseSchemaError from exc
            remaining = int((expiry - datetime.now(timezone.utc)).total_seconds())
            if remaining <= 0:
                raise ResponseSchemaError
            ttl = min(ttl, remaining)
        rate_limit = ControlPlane._validate_rate_limit(payload)
        return KeyAccess(key_id, min(ttl, cache_max_ttl), rate_limit)

    @staticmethod
    def _validate_rate_limit(payload: dict[str, Any]) -> RateLimit | None:
        if "rate_limit" not in payload:
            return None
        policy = payload["rate_limit"]
        if type(policy) is not dict or set(policy) != {"requests", "window_seconds"}:
            raise ResponseSchemaError
        requests = policy["requests"]
        window = policy["window_seconds"]
        if (
            type(requests) is not int
            or not 0 <= requests <= 10_000
            or type(window) is not int
            or window != 60
        ):
            raise ResponseSchemaError
        return RateLimit(requests, window)

    @staticmethod
    def _retry_after(window: int, window_seconds: int, now: int) -> int:
        return max(1, (window + 1) * window_seconds - now)

    def _prune_key_windows(self, current_window: int) -> None:
        keys = tuple(self._key_windows)
        if not keys:
            self._key_window_prune_cursor = 0
            return
        start = self._key_window_prune_cursor % len(keys)
        checked = min(STATE_PRUNE_LIMIT, len(keys))
        for offset in range(checked):
            key_id = keys[(start + offset) % len(keys)]
            bucket = self._key_windows.get(key_id)
            if bucket is not None and bucket[0] != current_window:
                self._key_windows.pop(key_id, None)
        self._key_window_prune_cursor = (start + checked) % max(
            len(self._key_windows), 1
        )

    def enforce_rate_limit(self, access: KeyAccess) -> int | None:
        enforcement = self.enforcement
        policy = access.rate_limit
        if enforcement and policy is None:
            logger.warning(
                "rate_limit_policy_missing: enforcement active but introspection "
                "returned no per-key policy; failing closed for key_id=%s",
                access.key_id,
            )
            raise RateLimitPolicyMissing
        now = int(time.time())
        global_window = now // self._window_seconds
        with self._limiter_lock:
            previous_global_window, global_count = self._global_window or (
                global_window,
                0,
            )
            if previous_global_window != global_window:
                global_count = 0
            if enforcement:
                key_window = now // policy.window_seconds
                self._prune_key_windows(key_window)
                previous_key_window, key_count = self._key_windows.get(
                    access.key_id, (key_window, 0)
                )
                if previous_key_window != key_window:
                    key_count = 0
                if policy.requests == 0 or key_count >= policy.requests:
                    return self._retry_after(key_window, policy.window_seconds, now)
                if (
                    access.key_id not in self._key_windows
                    and len(self._key_windows) >= self._limiter_max_keys
                ):
                    return self._retry_after(key_window, policy.window_seconds, now)
            if global_count >= self._global_limit:
                return self._retry_after(global_window, self._window_seconds, now)
            if enforcement:
                self._key_windows[access.key_id] = (key_window, key_count + 1)
            self._global_window = (global_window, global_count + 1)
        return None

    def record_usage(self, key_id: str, status_code: int, latency_ms: int) -> None:
        bucket = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        bucket_start = bucket.strftime("%Y-%m-%dT%H:%M:%SZ")
        status_class = f"{status_code // 100}xx"
        key = (bucket_start, key_id, status_class)
        now = time.monotonic()
        should_warn = False
        overflow = False
        with self._usage_lock:
            record = self.usage.get(key)
            if record is None:
                if key not in self._known_usage_dimensions:
                    if len(self._known_usage_dimensions) >= self._usage_max_dimensions:
                        self.usage_overflow_count = min(
                            self.usage_overflow_count + 1, MAX_COUNTER
                        )
                        if now - self._last_overflow_log >= 60:
                            self._last_overflow_log = now
                            should_warn = True
                        overflow = True
                    else:
                        self._known_usage_dimensions.add(key)
                if not overflow:
                    record = {
                        "bucket_start": bucket_start,
                        "key_id": key_id,
                        "endpoint_template": "/predict",
                        "method": "POST",
                        "status_class": status_class,
                        "request_count": 0,
                        "error_count": 0,
                        "latency_sum_ms": 0,
                        "latency_max_ms": 0,
                    }
                    self.usage[key] = record
            if not overflow:
                record["request_count"] = min(record["request_count"] + 1, 1_000_000)
                record["error_count"] = min(
                    record["error_count"] + (status_code >= 400),
                    record["request_count"],
                )
                latency = min(max(latency_ms, 0), 86_400_000)
                record["latency_sum_ms"] = min(
                    record["latency_sum_ms"] + latency, 86_400_000_000
                )
                record["latency_max_ms"] = min(
                    max(record["latency_max_ms"], latency),
                    record["latency_sum_ms"],
                )
        if overflow:
            self._log_overflow(should_warn)

    def _log_overflow(self, should_warn: bool) -> None:
        if should_warn:
            logger.warning(
                "usage aggregate queue full; dropped_observations=%s "
                "max_unique_dimensions=%s max_usage_record_slots=%s",
                self.usage_overflow_count,
                self._usage_max_dimensions,
                self.max_usage_record_slots,
            )

    @staticmethod
    def _record_is_stale(record: dict[str, Any], cutoff: datetime) -> bool:
        try:
            bucket = datetime.strptime(
                record["bucket_start"], "%Y-%m-%dT%H:%M:%SZ"
            ).replace(tzinfo=timezone.utc)
        except (KeyError, TypeError, ValueError):
            return True
        return bucket < cutoff

    def _account_drop_locked(self, category: str, batches: int, records: int) -> bool:
        self.usage_dropped_batches_count = min(
            self.usage_dropped_batches_count + batches, MAX_COUNTER
        )
        self.usage_dropped_records_count = min(
            self.usage_dropped_records_count + records, MAX_COUNTER
        )
        self.usage_failure_counts[category] = min(
            self.usage_failure_counts[category] + 1, MAX_COUNTER
        )
        now = time.monotonic()
        if now - self._last_drop_log < 60:
            return False
        self._last_drop_log = now
        return True

    def _log_drop(
        self, category: str, batches: int, records: int, should_warn: bool
    ) -> None:
        if should_warn:
            logger.warning(
                "usage records dropped; category=%s batches=%s records=%s",
                category,
                batches,
                records,
            )

    def _prune_stale_usage_locked(self, cutoff: datetime) -> tuple[int, bool]:
        stale_keys = [
            key
            for key, record in self.usage.items()
            if self._record_is_stale(record, cutoff)
        ]
        for key in stale_keys:
            self.usage.pop(key, None)
            if key not in self._pending_dimension_keys:
                self._known_usage_dimensions.discard(key)
        if not stale_keys:
            return 0, False
        return len(stale_keys), self._account_drop_locked("stale", 0, len(stale_keys))

    async def prepare_usage_batch(self) -> dict[str, Any] | None:
        cutoff = datetime.now(timezone.utc) - timedelta(
            seconds=self._usage_record_max_age_seconds
        )
        stale_count = 0
        should_warn = False
        with self._usage_lock:
            if self.pending_usage is not None:
                return self.pending_usage
            stale_count, should_warn = self._prune_stale_usage_locked(cutoff)
            if not self.usage:
                batch = None
            else:
                keys = list(self.usage)[:MAX_BATCH_RECORDS]
                records = [self.usage.pop(key) for key in keys]
                self._pending_dimension_keys = set(keys)
                self._pending_auth_attempts = 0
                self.pending_usage = {
                    "batch_id": str(uuid.uuid4()),
                    "records": records,
                }
                batch = self.pending_usage
        if stale_count:
            self._log_drop("stale", 0, stale_count, should_warn)
        return batch

    def schedule_usage_flush(self, request_id: str) -> asyncio.Task | None:
        with self._flush_task_lock:
            if not self._accept_flush_scheduling:
                return None
            if self._flush_task is not None and not self._flush_task.done():
                return self._flush_task
            if time.monotonic() < self._next_flush_allowed:
                return None
            task = asyncio.create_task(self._flush_worker(request_id))
            self._flush_task = task
            task.add_done_callback(self._consume_flush_result)
            return task

    def _consume_flush_result(self, task: asyncio.Task) -> None:
        outcome = UsageFlushOutcome.TRANSIENT
        try:
            outcome = task.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            self._log_flush_failure("unexpected")
        with self._flush_task_lock:
            if self._flush_task is task:
                self._flush_task = None
            if (
                outcome in {UsageFlushOutcome.ACCEPTED, UsageFlushOutcome.DEAD_LETTERED}
                and self._accept_flush_scheduling
                and self.pending_usage_count() > 0
                and self._flush_task is None
            ):
                worker = (
                    self._delayed_flush_worker
                    if outcome is UsageFlushOutcome.DEAD_LETTERED
                    else self._flush_worker
                )
                next_task = asyncio.create_task(worker("scheduled"))
                self._flush_task = next_task
                next_task.add_done_callback(self._consume_flush_result)

    def _log_flush_failure(self, category: str) -> None:
        now = time.monotonic()
        if now - self._last_failure_log < 60:
            return
        self._last_failure_log = now
        logger.warning("usage flush failed; category=%s", category)

    def _log_permanent_rejection(self, batch: dict[str, Any]) -> None:
        record_count = len(batch.get("records", []))
        logger.warning("usage batch permanently rejected; records=%s", record_count)

    async def _delayed_flush_worker(self, request_id: str) -> UsageFlushOutcome:
        await asyncio.sleep(self._usage_retry_delay_seconds)
        return await self._flush_worker(request_id)

    async def _flush_worker(self, request_id: str) -> UsageFlushOutcome:
        batch = await self.prepare_usage_batch()
        if batch is None:
            return UsageFlushOutcome.IDLE
        return await self._send_batch(batch, request_id)

    async def flush_usage(self, request_id: str) -> UsageFlushOutcome:
        batch = await self.prepare_usage_batch()
        if batch is None:
            return UsageFlushOutcome.IDLE
        return await self._send_batch(batch, request_id)

    async def _send_batch(
        self, batch: dict[str, Any], request_id: str
    ) -> UsageFlushOutcome:
        async with self._send_lock:
            return await self._send_batch_locked(batch, request_id)

    def _acknowledge_batch(self, batch: dict[str, Any]) -> bool:
        with self._usage_lock:
            if self.pending_usage is not batch:
                return False
            self.pending_usage = None
            self._pending_auth_attempts = 0
            for key in self._pending_dimension_keys:
                if key not in self.usage:
                    self._known_usage_dimensions.discard(key)
            self._pending_dimension_keys.clear()
        return True

    def _dead_letter_batch(self, batch: dict[str, Any], category: str) -> bool:
        with self._usage_lock:
            if self.pending_usage is not batch:
                return False
            record_count = len(batch["records"])
            self.pending_usage = None
            self._pending_auth_attempts = 0
            for key in self._pending_dimension_keys:
                if key not in self.usage:
                    self._known_usage_dimensions.discard(key)
            self._pending_dimension_keys.clear()
            should_warn = self._account_drop_locked(category, 1, record_count)
        self._next_flush_allowed = 0.0
        self._log_drop(category, 1, record_count, should_warn)
        return True

    def _pending_batch_is_stale(self, batch: dict[str, Any]) -> bool:
        cutoff = datetime.now(timezone.utc) - timedelta(
            seconds=self._usage_record_max_age_seconds
        )
        return any(self._record_is_stale(record, cutoff) for record in batch["records"])

    async def _retry_delay(self, attempt: int) -> None:
        await asyncio.sleep(min(self._usage_retry_delay_seconds * (2**attempt), 5.0))

    async def _send_batch_locked(
        self, batch: dict[str, Any], request_id: str
    ) -> UsageFlushOutcome:
        with self._usage_lock:
            if self.pending_usage is not batch:
                return UsageFlushOutcome.ACCEPTED
        if self._pending_batch_is_stale(batch):
            self._dead_letter_batch(batch, "stale")
            return UsageFlushOutcome.DEAD_LETTERED
        failure = "transient"
        for attempt in range(self._usage_retry_attempts):
            if self._pending_batch_is_stale(batch):
                self._dead_letter_batch(batch, "stale")
                return UsageFlushOutcome.DEAD_LETTERED
            try:
                async with self.client.stream(
                    "POST",
                    f"{self.base_url}{self._USAGE_PATH}",
                    headers=self._headers(request_id),
                    json=batch,
                ) as response:
                    status = response.status_code
                    if status in {200, 202}:
                        payload = await self._read_json_response(response)
                        if (
                            type(payload) is dict
                            and payload.get("accepted") is True
                            and type(payload.get("duplicate")) is bool
                        ):
                            self._acknowledge_batch(batch)
                            self._next_flush_allowed = 0.0
                            return UsageFlushOutcome.ACCEPTED
                        with self._usage_lock:
                            self.usage_failure_counts["response_schema"] = min(
                                self.usage_failure_counts["response_schema"] + 1,
                                MAX_COUNTER,
                            )
                        failure = "response_schema"
                    elif status in {401, 403}:
                        with self._usage_lock:
                            if self.pending_usage is not batch:
                                return UsageFlushOutcome.ACCEPTED
                            self._pending_auth_attempts += 1
                            auth_attempts = self._pending_auth_attempts
                        if auth_attempts >= self._usage_retry_attempts:
                            self._dead_letter_batch(batch, "authentication")
                            return UsageFlushOutcome.DEAD_LETTERED
                        failure = "authentication"
                    elif status == 429 or status >= 500:
                        failure = "retryable_status"
                    elif 400 <= status < 500:
                        self._log_permanent_rejection(batch)
                        self._dead_letter_batch(batch, "permanent_payload")
                        return UsageFlushOutcome.DEAD_LETTERED
                    else:
                        self._dead_letter_batch(batch, "permanent_response")
                        return UsageFlushOutcome.DEAD_LETTERED
            except (httpx.TimeoutException, httpx.NetworkError):
                failure = "network"
            except (httpx.HTTPError, ControlPlaneError, ValueError, TypeError):
                failure = "response_error"
            if attempt + 1 < self._usage_retry_attempts:
                await self._retry_delay(attempt)
        self._next_flush_allowed = time.monotonic() + min(
            self._usage_retry_delay_seconds * (2**self._usage_retry_attempts),
            5.0,
        )
        self._log_flush_failure(failure)
        return UsageFlushOutcome.TRANSIENT

    def pending_usage_count(self) -> int:
        with self._usage_lock:
            pending = (
                len(self.pending_usage["records"])
                if self.pending_usage is not None
                else 0
            )
            return pending + len(self.usage)

    async def shutdown_usage(self) -> bool:
        with self._flush_task_lock:
            self._accept_flush_scheduling = False
            task = self._flush_task
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._usage_shutdown_timeout_seconds
        tracked_outcome = UsageFlushOutcome.IDLE
        try:
            if task is not None:
                remaining = deadline - loop.time()
                if remaining > 0:
                    try:
                        tracked_outcome = await asyncio.wait_for(
                            asyncio.shield(task), timeout=remaining
                        )
                    except asyncio.TimeoutError:
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                        tracked_outcome = UsageFlushOutcome.TRANSIENT
                    except Exception:
                        self._log_flush_failure("unexpected")
                        tracked_outcome = UsageFlushOutcome.TRANSIENT
            if tracked_outcome is not UsageFlushOutcome.TRANSIENT:
                while self.pending_usage_count() > 0:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        break
                    try:
                        outcome = await asyncio.wait_for(
                            self.flush_usage("shutdown"), timeout=remaining
                        )
                    except asyncio.TimeoutError:
                        break
                    if outcome is UsageFlushOutcome.TRANSIENT:
                        break
                    if outcome is UsageFlushOutcome.IDLE:
                        break
        finally:
            with self._flush_task_lock:
                current = self._flush_task
            if current is not None and not current.done():
                current.cancel()
                await asyncio.gather(current, return_exceptions=True)
            elif current is not None:
                try:
                    current.result()
                except (asyncio.CancelledError, Exception):
                    pass
            with self._flush_task_lock:
                if self._flush_task is current:
                    self._flush_task = None
        remaining_count = self.pending_usage_count()
        if remaining_count:
            logger.warning(
                "usage shutdown drain incomplete; remaining_records=%s "
                "category=timeout_or_unavailable",
                remaining_count,
            )
        return remaining_count == 0
