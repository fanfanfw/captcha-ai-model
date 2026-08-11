import csv
import random
from pathlib import Path

ROOT = Path(__file__).parent
SEED = 42

with (ROOT / "labels.csv").open(newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

random.Random(SEED).shuffle(rows)
train_end = int(len(rows) * 0.8)
validation_end = train_end + int(len(rows) * 0.1)
splits = {
    "train": rows[:train_end],
    "validation": rows[train_end:validation_end],
    "test": rows[validation_end:],
}

for name, split in splits.items():
    path = ROOT / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["filename", "text"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(split)
    print(f"{path.name}: {len(split)}")
