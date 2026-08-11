import io

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app, model_service

client = TestClient(app)


def image_bytes(format_name: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buffer, format=format_name)
    return buffer.getvalue()


def test_health(monkeypatch):
    monkeypatch.setattr(model_service, "ready", lambda: True)
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["cache-control"] == "no-store"
    assert client.get("/health/ready").status_code == 200


def test_not_ready(monkeypatch):
    monkeypatch.setattr(model_service, "ready", lambda: False)
    response = client.get("/health/ready")
    assert response.status_code == 503
    assert response.json() == {"status": "unavailable"}


def test_predict_without_loading_model(monkeypatch):
    monkeypatch.setattr(model_service, "predict", lambda image: "ABCDE")
    response = client.post(
        "/predict",
        files={"file": ("captcha.png", image_bytes(), "image/png")},
    )
    assert response.status_code == 200
    assert response.json() == {"prediction": "ABCDE"}


def test_predict_rejects_invalid_image():
    response = client.post(
        "/predict", files={"file": ("captcha.png", b"not an image", "image/png")}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_image"


def test_predict_rejects_wrong_mime():
    response = client.post(
        "/predict", files={"file": ("captcha.txt", b"data", "text/plain")}
    )
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_media_type"
