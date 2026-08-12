import io
import os
import re
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, Security, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from PIL import Image, UnidentifiedImageError

from app.control_plane import ControlPlane, ControlPlaneError
from app.model import model_service

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png"}
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
API_KEY_HEADER = APIKeyHeader(
    name="x-api-key",
    auto_error=False,
    description="Managed API key issued by the API Key Management dashboard.",
)


def error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code, detail={"code": code, "message": message}
    )


def response(status_code: int, content: dict, headers: dict[str, str] | None = None):
    return JSONResponse(
        status_code=status_code,
        content=content,
        headers={"Cache-Control": "no-store", **(headers or {})},
    )


def create_app(control_plane: ControlPlane | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        configured = control_plane or ControlPlane(
            os.environ.get("CONTROL_PLANE_BASE_URL", ""),
            os.environ.get("SERVICE_CREDENTIAL", ""),
        )
        application.state.control_plane = configured
        yield
        if control_plane is None:
            await configured.close()

    application = FastAPI(
        title="CAPTCHA Inference API", version="0.1.0", lifespan=lifespan
    )

    @application.middleware("http")
    async def request_context(request: Request, call_next):
        supplied = request.headers.get("X-Request-ID", "")
        request_id = (
            supplied if REQUEST_ID_PATTERN.fullmatch(supplied) else uuid.uuid4().hex
        )
        request.state.request_id = request_id
        started = time.monotonic()
        try:
            result = await call_next(request)
        except Exception:
            result = response(
                500,
                {"error": {"code": "internal_error", "message": "Internal error."}},
            )
        result.headers["X-Request-ID"] = request_id
        result.headers["Cache-Control"] = "no-store"
        key_id = getattr(request.state, "key_id", None)
        if request.url.path == "/predict" and key_id:
            latency = max(0, round((time.monotonic() - started) * 1000))
            application.state.control_plane.record_usage(
                key_id, result.status_code, latency
            )
            await application.state.control_plane.flush_usage(request_id)
        return result

    @application.exception_handler(HTTPException)
    async def http_exception_handler(_, exc: HTTPException) -> JSONResponse:
        detail = exc.detail
        if not isinstance(detail, dict):
            detail = {"code": "http_error", "message": str(detail)}
        return response(exc.status_code, {"error": detail}, dict(exc.headers or {}))

    @application.exception_handler(RequestValidationError)
    async def validation_exception_handler(_, __) -> JSONResponse:
        return response(
            422,
            {"error": {"code": "invalid_request", "message": "Request tidak valid."}},
        )

    @application.get("/health/live")
    def live() -> JSONResponse:
        return response(200, {"status": "ok"})

    @application.get("/health/ready")
    def ready() -> JSONResponse:
        if model_service.ready():
            return response(200, {"status": "ok"})
        return response(503, {"status": "unavailable"})

    @application.post("/predict")
    async def predict(
        request: Request,
        file: UploadFile = File(...),
        raw_key: str | None = Security(API_KEY_HEADER),
    ) -> dict[str, str]:
        if raw_key is None:
            raise error(401, "unauthorized", "x-api-key wajib diisi.")
        if (
            not 8 <= len(raw_key) <= 512
            or raw_key.strip() != raw_key
            or " " in raw_key
            or not raw_key.isascii()
        ):
            raise error(401, "unauthorized", "x-api-key tidak valid.")
        try:
            access = await application.state.control_plane.introspect(
                raw_key, request.state.request_id
            )
        except ControlPlaneError:
            raise error(
                503,
                "authorization_unavailable",
                "Layanan otorisasi tidak tersedia.",
            ) from None
        if access is None:
            raise error(401, "unauthorized", "API key tidak aktif.")
        request.state.key_id = access.key_id
        retry_after = application.state.control_plane.enforce_rate_limit(access)
        if retry_after is not None:
            exception = error(429, "rate_limited", "Rate limit terlampaui.")
            exception.headers = {"Retry-After": str(retry_after)}
            raise exception
        if file.content_type not in ALLOWED_TYPES:
            raise error(415, "unsupported_media_type", "Gunakan gambar PNG atau JPEG.")
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        await file.close()
        if not data:
            raise error(400, "empty_file", "File gambar kosong.")
        if len(data) > MAX_UPLOAD_BYTES:
            raise error(413, "file_too_large", "Ukuran gambar maksimal 5 MB.")
        try:
            with Image.open(io.BytesIO(data)) as source:
                source.verify()
            with Image.open(io.BytesIO(data)) as source:
                if source.format not in {"JPEG", "PNG"}:
                    raise error(
                        415,
                        "unsupported_image_format",
                        "Isi file harus berupa PNG atau JPEG.",
                    )
                image = source.convert("RGB")
        except HTTPException:
            raise
        except (
            Image.DecompressionBombError,
            UnidentifiedImageError,
            OSError,
            ValueError,
        ):
            raise error(400, "invalid_image", "File bukan gambar valid.") from None
        try:
            text = await run_in_threadpool(model_service.predict, image)
        except RuntimeError as exc:
            raise error(503, "model_unavailable", str(exc)) from exc
        return {"prediction": text}

    return application


app = create_app()
