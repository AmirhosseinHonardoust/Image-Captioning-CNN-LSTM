"""Convert Flickr-style caption files into the CSV format used by this project.

Expected input examples:
    image.jpg#0\tA dog runs through grass.
    image.jpg,A dog runs through grass.

The output CSV contains: image_path,caption,split
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def parse_caption_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line:
        return None

    if "\t" in line:
        image_id, caption = line.split("\t", 1)
    elif "," in line:
        image_id, caption = line.split(",", 1)
    else:
        return None

    image_name = image_id.split("#", 1)[0].strip()
    caption = caption.strip()
    if not image_name or not caption:
        return None
    return image_name, caption


def assign_split(image_name: str, split_map: dict[str, str], rng: random.Random) -> str:
    if image_name in split_map:
        return split_map[image_name]

    draw = rng.random()
    if draw < 0.8:
        split = "train"
    elif draw < 0.9:
        split = "val"
    else:
        split = "test"
    split_map[image_name] = split
    return split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions-file", required=True, help="Flickr-style captions text file")
    parser.add_argument(
        "--images-subdir", default="images", help="Prefix stored in image_path column"
    )
    parser.add_argument("--output", default="captions.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    captions_file = Path(args.captions_file)
    rows = []
    split_map: dict[str, str] = {}
    rng = random.Random(args.seed)

    with captions_file.open("r", encoding="utf-8") as f:
        for raw_line in f:
            parsed = parse_caption_line(raw_line)
            if parsed is None:
                continue
            image_name, caption = parsed
            rows.append(
                {
                    "image_path": str(Path(args.images_subdir) / image_name),
                    "caption": caption,
                    "split": assign_split(image_name, split_map, rng),
                }
            )

    if not rows:
        raise ValueError(f"No valid caption rows were parsed from {captions_file}")

    with Path(args.output).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "caption", "split"])
        writer.writeheader()
        writer.writerows(rows)

    counts = {
        split: sum(row["split"] == split for row in rows) for split in ["train", "val", "test"]
    }
    print(f"Wrote {len(rows)} rows to {args.output}: {counts}")


if __name__ == "__main__":
    main()
