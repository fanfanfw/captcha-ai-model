import threading
from pathlib import Path

from PIL import Image

MODEL_PATH = Path(__file__).resolve().parent.parent / "captcha-model" / "best"


class ModelService:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.model_path = model_path
        self._lock = threading.Lock()
        self._processor = None
        self._model = None
        self._device = None

    def checkpoint_ready(self) -> bool:
        required = ("config.json", "model.safetensors", "preprocessor_config.json")
        return self.model_path.is_dir() and all(
            (self.model_path / name).is_file() for name in required
        )

    def ready(self) -> bool:
        return self.checkpoint_ready()

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.checkpoint_ready():
            raise RuntimeError("Checkpoint model tidak tersedia.")

        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor = TrOCRProcessor.from_pretrained(self.model_path, use_fast=False)
        model = VisionEncoderDecoderModel.from_pretrained(
            self.model_path, early_stopping=False
        )
        self._device = device
        self._processor = processor
        self._model = model.to(device).eval()

    def predict(self, image: Image.Image) -> str:
        with self._lock:
            self._load()
            import torch

            pixels = self._processor(
                image.convert("RGB"), return_tensors="pt"
            ).pixel_values.to(self._device)
            with torch.inference_mode():
                generated = self._model.generate(pixels, max_length=7, num_beams=1)
            return self._processor.batch_decode(generated, skip_special_tokens=True)[0]


model_service = ModelService()
