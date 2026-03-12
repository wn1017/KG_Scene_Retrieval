from __future__ import annotations

import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import csv
from collections import Counter
from pathlib import Path

from config import TEXT_CSV_PATH
from src.nlp_parser import parse_query


DEFAULT_OUTPUT_PATH = TEXT_CSV_PATH.with_name(f"{TEXT_CSV_PATH.stem}_parsed.csv")


def read_text_rows(csv_path: Path):
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            yield int(row["id"]), str(row["text"])


def build_text_metadata(text: str) -> dict[str, str]:
    parsed = parse_query(text)
    return {
        "language": parsed.get("language") or "en",
        "model_key": parsed.get("model_key") or "engclip",
        "weather": parsed.get("weather") or "",
        "timeofday": parsed.get("time") or "",
        "location": parsed.get("location") or "",
        "obj_types": ",".join(parsed.get("objects") or []),
    }


def iter_text_records(csv_path: Path, limit: int | None = None):
    total = 0
    for row_id, text in read_text_rows(csv_path):
        yield {
            "id": int(row_id),
            "text": text,
            **build_text_metadata(text),
        }
        total += 1
        if limit is not None and total >= limit:
            break


def write_parsed_csv(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "text", "language", "model_key", "weather", "timeofday", "location", "obj_types"]
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def print_summary(records: list[dict], output_path: Path) -> None:
    language_counter = Counter(record["language"] for record in records)
    model_counter = Counter(record["model_key"] for record in records)
    print(f"Parsed {len(records)} text rows from {TEXT_CSV_PATH}.")
    print(f"Output file: {output_path}")
    print(f"Languages: {dict(language_counter)}")
    print(f"Model keys: {dict(model_counter)}")
    print("No text entities were inserted into Milvus.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse text queries into structured metadata for evaluation and demo usage")
    parser.add_argument("--csv", type=Path, default=TEXT_CSV_PATH, help="Path to the text CSV file")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Path to the parsed output CSV")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke tests")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.csv.exists():
        raise FileNotFoundError(f"Text CSV not found: {args.csv}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than 0 when provided")

    records = list(iter_text_records(args.csv, args.limit))
    write_parsed_csv(records, args.output)
    print_summary(records, args.output)


if __name__ == "__main__":
    main()
