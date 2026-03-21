from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nlp_parser import parse_query
from src.nuscenes_metadata import get_image_metadata, load_nuscenes_metadata, resolve_frame_path


FIELD_NAMES = ("图片", "查询", "语言", "备注")
NOTE_COUNT_MAP = {
    "单条件": 1,
    "两条件": 2,
    "三条件": 3,
    "四条件": 4,
    "五条件": 5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a benchmark CSV from a markdown-style manual query list.")
    parser.add_argument(
        "--input",
        default=r"C:\Users\doubleu\Desktop\query.md",
        help="Path to the manual query markdown file.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "benchmark" / "manual_query_benchmark.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def split_key_value(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped:
        return None
    for separator in ("：", ":"):
        if separator in stripped:
            key, value = stripped.split(separator, 1)
            key = key.strip()
            value = value.strip()
            if key in FIELD_NAMES:
                return key, value
    return None


def parse_query_blocks(text: str) -> list[dict[str, str]]:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    records: list[dict[str, str]] = []
    for block in blocks:
        record: dict[str, str] = {}
        for line in block.splitlines():
            parsed = split_key_value(line)
            if parsed is None:
                continue
            key, value = parsed
            record[key] = value
        if record:
            records.append(record)
    return records


def note_to_group(note: str) -> tuple[int | None, str]:
    normalized = str(note or "").strip()
    count = NOTE_COUNT_MAP.get(normalized)
    if count is None:
        return None, ""
    if count <= 1:
        return count, "open_semantic"
    if count == 2:
        return count, "two_condition"
    return count, "three_plus"


def parsed_condition_count(parsed_query: dict) -> int:
    return int(bool(parsed_query.get("weather"))) + int(bool(parsed_query.get("time"))) + int(
        bool(parsed_query.get("location"))
    ) + len(parsed_query.get("objects") or [])


def build_warning(note_count: int | None, parsed_query: dict, image_metadata: dict[str, str]) -> str:
    warnings: list[str] = []
    parsed_count = parsed_condition_count(parsed_query)
    if note_count is not None and parsed_count < note_count:
        warnings.append(f"hint={note_count}, parsed={parsed_count}")
    if not image_metadata.get("scene_token"):
        warnings.append("missing_scene_token")
    if not parsed_query.get("weather") and not parsed_query.get("time") and not parsed_query.get("location") and not (
        parsed_query.get("objects") or []
    ):
        warnings.append("no_structured_filters")
    return "; ".join(warnings)


def build_dataframe(records: list[dict[str, str]]) -> pd.DataFrame:
    load_nuscenes_metadata()
    rows: list[dict[str, object]] = []
    for index, record in enumerate(records, start=1):
        image_path = record.get("图片", "")
        resolved_path = resolve_frame_path(image_path)
        image_metadata = get_image_metadata(image_path)
        parsed_query = parse_query(record.get("查询", ""))
        note_count, query_group = note_to_group(record.get("备注", ""))

        rows.append(
            {
                "query_id": f"M{index:03d}",
                "image_path": image_path,
                "resolved_image_path": str(resolved_path) if resolved_path else image_metadata.get("resolved_path", ""),
                "image_exists": bool(resolved_path and resolved_path.exists()),
                "query_text": record.get("查询", ""),
                "language": record.get("语言", ""),
                "note": record.get("备注", ""),
                "condition_count_hint": note_count,
                "query_group": query_group,
                "scene_token": image_metadata.get("scene_token", ""),
                "sample_token": image_metadata.get("sample_token", ""),
                "camera": image_metadata.get("camera", ""),
                "frame_path": image_metadata.get("frame_path", ""),
                "weather_hint": image_metadata.get("weather", ""),
                "timeofday_hint": image_metadata.get("timeofday", ""),
                "location_hint": image_metadata.get("location", ""),
                "objects_hint": image_metadata.get("obj_types", ""),
                "parsed_weather": parsed_query.get("weather"),
                "parsed_time": parsed_query.get("time"),
                "parsed_location": parsed_query.get("location"),
                "parsed_objects": json.dumps(parsed_query.get("objects") or [], ensure_ascii=False),
                "parsed_condition_count": parsed_condition_count(parsed_query),
                "parsed_language": parsed_query.get("language", ""),
                "model_key": parsed_query.get("model_key", ""),
                "parse_warning": build_warning(note_count, parsed_query, image_metadata),
                "relevant_scene_tokens": "",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    text = input_path.read_text(encoding="utf-8")
    records = parse_query_blocks(text)
    dataframe = build_dataframe(records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"queries={len(dataframe)}")
    print(f"output={output_path}")
    print(dataframe[['query_group', 'language']].fillna('').value_counts().to_string())


if __name__ == "__main__":
    main()
