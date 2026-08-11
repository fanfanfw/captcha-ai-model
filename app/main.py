import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from app.model import model_service

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png"}

app = FastAPI(title="CAPTCHA Inference API", version="0.1.0")


def error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException) -> JSONResponse:
    detail = exc.detail
    if not isinstance(detail, dict):
        detail = {"code": "http_error", "message": str(detail)}
    return JSONResponse(status_code=exc.status_code, content={"error": detail})


@app.get("/health/live")
def live() -> JSONResponse:
    return JSONResponse(content={"status": "ok"}, headers={"Cache-Control": "no-store"})


@app.get("/health/ready")
def ready() -> JSONResponse:
    if model_service.ready():
        return JSONResponse(
            content={"status": "ok"}, headers={"Cache-Control": "no-store"}
        )
    return JSONResponse(
        status_code=503,
        content={"status": "unavailable"},
        headers={"Cache-Control": "no-store"},
    )


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, str]:
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
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError, ValueError):
        raise error(400, "invalid_image", "File bukan gambar valid.") from None

    try:
        text = await run_in_threadpool(model_service.predict, image)
    except RuntimeError as exc:
        raise error(503, "model_unavailable", str(exc)) from exc
    return {"prediction": text}
