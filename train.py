import argparse
import csv
import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, get_linear_schedule_with_warmup

MODEL_ID = "anuashok/ocr-captcha-v3"
ROOT = Path(__file__).parent


class CaptchaDataset(Dataset):
    def __init__(self, csv_path, limit=None):
        with csv_path.open(newline="", encoding="utf-8") as file:
            self.rows = list(csv.DictReader(file))
        if limit:
            self.rows = self.rows[:limit]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(ROOT / "captcha" / row["filename"]) as image:
            return image.convert("RGB"), row["text"]


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
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "captcha-model")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = TrOCRProcessor.from_pretrained(MODEL_ID, use_fast=False)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID, early_stopping=False).to(device)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 7
    model.config.num_beams = 1

    def collate(batch):
        images, texts = zip(*batch)
        pixels = processor(images=list(images), return_tensors="pt").pixel_values
        labels = processor.tokenizer(list(texts), padding=True, return_tensors="pt").input_ids
        labels[labels == processor.tokenizer.pad_token_id] = -100
        return pixels, labels, texts

    train_dataset = CaptchaDataset(ROOT / "train.csv", args.train_limit)
    validation_dataset = CaptchaDataset(ROOT / "validation.csv", args.validation_limit)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=4, pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size * 2, collate_fn=collate, num_workers=4, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_exact = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for step, (pixels, labels, _) in enumerate(train_loader, 1):
            pixels = pixels.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = model(pixel_values=pixels, labels=labels).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()
            if step % 20 == 0 or step == len(train_loader):
                print(f"epoch={epoch} train={step}/{len(train_loader)} loss={total_loss / step:.4f}")

        model.eval()
        exact = errors = characters = samples = 0
        with torch.inference_mode():
            for pixels, _, texts in validation_loader:
                generated = model.generate(pixels.to(device, non_blocking=True), max_length=7, num_beams=1)
                predictions = processor.batch_decode(generated, skip_special_tokens=True)
                exact += sum(expected == predicted for expected, predicted in zip(texts, predictions))
                errors += sum(edit_distance(expected, predicted) for expected, predicted in zip(texts, predictions))
                characters += sum(len(text) for text in texts)
                samples += len(texts)
        metrics = {"epoch": epoch, "train_loss": total_loss / len(train_loader), "exact_match": exact / samples, "cer": errors / characters}
        history.append(metrics)
        print(f"epoch={epoch} validation_exact={metrics['exact_match']:.4%} validation_cer={metrics['cer']:.4%}")
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if metrics["exact_match"] > best_exact:
            best_exact = metrics["exact_match"]
            model.save_pretrained(args.output_dir / "best")
            processor.save_pretrained(args.output_dir / "best")
            print(f"saved={args.output_dir / 'best'}")


if __name__ == "__main__":
    main()
