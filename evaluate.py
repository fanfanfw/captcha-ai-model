import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

MODEL_ID = "anuashok/ocr-captcha-v3"
ROOT = Path(__file__).parent


def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for i, left_character in enumerate(left, 1):
        current = [i]
        for j, right_character in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left_character != right_character)))
        previous = current
    return previous[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--split", type=Path, default=ROOT / "test.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=ROOT / "baseline_predictions.csv")
    args = parser.parse_args()

    with args.split.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = TrOCRProcessor.from_pretrained(args.model, use_fast=False)
    model = VisionEncoderDecoderModel.from_pretrained(args.model, early_stopping=False).to(device).eval()
    predictions = []

    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        images = []
        for row in batch:
            with Image.open(ROOT / "captcha" / row["filename"]) as image:
                images.append(image.convert("RGB"))
        pixels = processor(images, return_tensors="pt").pixel_values.to(device)
        with torch.inference_mode():
            generated = model.generate(pixels)
        texts = processor.batch_decode(generated, skip_special_tokens=True)
        for row, text in zip(batch, texts):
            predictions.append({"filename": row["filename"], "text": row["text"], "prediction": text})
        print(f"{min(start + args.batch_size, len(rows))}/{len(rows)}")

    exact = sum(row["text"] == row["prediction"] for row in predictions)
    errors = sum(edit_distance(row["text"], row["prediction"]) for row in predictions)
    characters = sum(len(row["text"]) for row in predictions)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["filename", "text", "prediction"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(predictions)
    print(f"exact_match={exact / len(predictions):.4%} ({exact}/{len(predictions)})")
    print(f"cer={errors / characters:.4%} ({errors}/{characters})")
    print(f"predictions={args.output}")


if __name__ == "__main__":
    main()
