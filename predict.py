import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

MODEL_ID = "anuashok/ocr-captcha-v3"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="File gambar atau folder CAPTCHA")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = TrOCRProcessor.from_pretrained(MODEL_ID, use_fast=False)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID, early_stopping=False).to(device).eval()

    paths = [args.path] if args.path.is_file() else sorted(args.path.glob("*.png"))
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        parser.error(f"gambar PNG tidak ditemukan: {args.path}")

    for path in paths:
        with Image.open(path) as image:
            pixel_values = processor(image.convert("RGB"), return_tensors="pt").pixel_values.to(device)
        with torch.inference_mode():
            generated_ids = model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"{path.name}\t{text}")


if __name__ == "__main__":
    main()
