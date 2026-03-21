from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from neo4j import GraphDatabase

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from src.nuscenes_metadata import (
    build_scene_metadata,
    collect_sample_object_types,
    load_nuscenes_metadata,
    lookup_sample_data,
    normalize_path_key,
    resolve_frame_path,
)


COMMON_PASSWORD_GUESSES = [
    "",
    "12345678",
    "neo4j123456",
    "password",
    "neo4j",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the scene-level Neo4j KG data associated with a nuScenes image."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--image", help="Absolute or relative image path.")
    source_group.add_argument("--name", help="Image filename or filename fragment.")
    parser.add_argument(
        "--neo4j-password",
        default="",
        help="Override Neo4j password. If omitted, the script tries config/env/common defaults.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum matches to inspect when using --name.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of a human-readable report.",
    )
    return parser.parse_args()


def find_matches_by_name(name_fragment: str, limit: int) -> list[dict]:
    metadata_index = load_nuscenes_metadata()
    fragment = name_fragment.lower().strip()
    if not fragment:
        return []

    matches: list[dict] = []
    for record in metadata_index["filename_to_sample_data"].values():
        filename = str(record.get("filename", ""))
        if fragment in filename.lower() or fragment in Path(filename).name.lower():
            matches.append(record)

    matches.sort(key=lambda item: (item.get("filename", ""), item.get("timestamp", 0)))
    return matches[: max(limit, 1)]


def resolve_input_to_sample_data(args: argparse.Namespace) -> tuple[dict, str]:
    metadata_index = load_nuscenes_metadata()

    if args.image:
        resolved_path = resolve_frame_path(args.image, metadata_index)
        sample_data = lookup_sample_data(args.image, resolved_path, metadata_index)
        if not sample_data:
            raise SystemExit(f"未找到图片对应的 sample_data：{args.image}")
        display_source = str(resolved_path) if resolved_path else normalize_path_key(args.image)
        return sample_data, display_source

    matches = find_matches_by_name(args.name, args.limit)
    if not matches:
        raise SystemExit(f"未找到文件名匹配项：{args.name}")
    if len(matches) > 1:
        print("找到多个匹配，请改用更具体的文件名或直接传 --image：", file=sys.stderr)
        for index, record in enumerate(matches, start=1):
            print(f"{index}. {record.get('filename', '')}", file=sys.stderr)
        raise SystemExit(2)
    return matches[0], str(matches[0].get("filename", ""))


def candidate_passwords(override: str) -> list[str]:
    values: list[str] = []
    for value in [override, os.getenv("NEO4J_PASSWORD", ""), NEO4J_PASSWORD, *COMMON_PASSWORD_GUESSES]:
        if value not in values:
            values.append(value)
    return values


def fetch_scene_graph_summary(scene_token: str, password_override: str) -> dict[str, object]:
    query = """
    MATCH (s:Scene {scene_token: $scene_token})
    OPTIONAL MATCH (s)-[:WEATHER]->(w:Weather)
    OPTIONAL MATCH (s)-[:TIMEOFDAY]->(t:TimeOfDay)
    OPTIONAL MATCH (s)-[:LOCTYPE]->(l:Location)
    OPTIONAL MATCH (s)-[r:CONTAINS]->(o:Object)
    RETURN
      s.scene_token AS scene_token,
      s.name AS scene_name,
      s.description AS scene_description,
      s.num_samples AS num_samples,
      s.weather AS scene_weather_attr,
      s.timeofday AS scene_timeofday_attr,
      s.location_area AS scene_location_area,
      s.location_kind AS scene_location_kind,
      w.name AS weather_node,
      t.name AS timeofday_node,
      l.name AS location_name,
      l.kind AS location_kind,
      l.area AS location_area,
      collect(DISTINCT {name: o.name, count: r.count}) AS object_rows
    """

    last_error: Exception | None = None
    for password in candidate_passwords(password_override):
        driver = None
        try:
            auth = (NEO4J_USER, password) if NEO4J_USER or password else None
            driver = GraphDatabase.driver(NEO4J_URI, auth=auth)
            with driver.session() as session:
                row = session.run(query, {"scene_token": scene_token}).single()
            if not row:
                raise RuntimeError(f"Neo4j 中未找到 scene_token={scene_token}")
            result = dict(row)
            result["auth_password_used"] = password
            return result
        except Exception as exc:  # pragma: no cover - connection/auth fallbacks
            last_error = exc
        finally:
            if driver is not None:
                driver.close()
    raise RuntimeError(f"无法连接或查询 Neo4j：{last_error}")


def build_report(sample_data: dict, source_text: str, password_override: str) -> dict[str, object]:
    metadata_index = load_nuscenes_metadata()
    scene_token = str(sample_data.get("scene_token", ""))
    sample_token = str(sample_data.get("sample_token", ""))
    if not scene_token:
        raise RuntimeError("该图片未解析到 scene_token。")

    resolved_path = resolve_frame_path(sample_data.get("filename", ""), metadata_index)
    local_scene_metadata = build_scene_metadata(scene_token, metadata_index)
    graph_summary = fetch_scene_graph_summary(scene_token, password_override)
    object_types = collect_sample_object_types(sample_token, metadata_index) if sample_token else []

    return {
        "input_source": source_text,
        "resolved_path": str(resolved_path) if resolved_path else "",
        "sample_data_token": str(sample_data.get("sample_data_token", "")),
        "sample_token": sample_token,
        "scene_token": scene_token,
        "camera": str(sample_data.get("camera", "")),
        "is_key_frame": bool(sample_data.get("is_key_frame", False)),
        "timestamp": int(sample_data.get("timestamp", 0) or 0),
        "filename": str(sample_data.get("filename", "")),
        "scene_name": str(graph_summary.get("scene_name") or local_scene_metadata.get("scene_name", "")),
        "scene_description": str(
            graph_summary.get("scene_description") or local_scene_metadata.get("scene_description", "")
        ),
        "weather": str(graph_summary.get("weather_node") or local_scene_metadata.get("weather", "")),
        "timeofday": str(graph_summary.get("timeofday_node") or local_scene_metadata.get("timeofday", "")),
        "location_name": str(graph_summary.get("location_name") or ""),
        "location_area": str(graph_summary.get("location_area") or graph_summary.get("scene_location_area") or ""),
        "location_kind": str(graph_summary.get("location_kind") or graph_summary.get("scene_location_kind") or ""),
        "scene_num_samples": int(graph_summary.get("num_samples") or 0),
        "sample_objects": object_types,
        "scene_object_counts": sorted(
            [
                {"name": item.get("name", ""), "count": int(item.get("count") or 0)}
                for item in graph_summary.get("object_rows", [])
                if item.get("name")
            ],
            key=lambda item: (-item["count"], item["name"]),
        ),
    }


def print_human_report(report: dict[str, object]) -> None:
    print("图片 -> Scene -> Neo4j KG 查询结果")
    print(f"输入来源: {report['input_source']}")
    print(f"解析到图片: {report['resolved_path'] or report['filename']}")
    print(f"sample_data_token: {report['sample_data_token']}")
    print(f"sample_token: {report['sample_token']}")
    print(f"scene_token: {report['scene_token']}")
    print(f"camera: {report['camera']}")
    print(f"is_key_frame: {report['is_key_frame']}")
    print(f"timestamp: {report['timestamp']}")
    print(f"scene_name: {report['scene_name']}")
    print(f"scene_description: {report['scene_description']}")
    print(f"weather: {report['weather']}")
    print(f"timeofday: {report['timeofday']}")
    print(f"location_name: {report['location_name']}")
    print(f"location_area: {report['location_area']}")
    print(f"location_kind: {report['location_kind']}")
    print(f"scene_num_samples: {report['scene_num_samples']}")
    print(f"sample_objects: {', '.join(report['sample_objects']) if report['sample_objects'] else '<none>'}")
    print("scene_object_counts:")
    scene_object_counts = report.get("scene_object_counts", [])
    if not scene_object_counts:
        print("  <none>")
    else:
        for item in scene_object_counts:
            print(f"  - {item['name']}: {item['count']}")


def main() -> None:
    args = parse_args()
    sample_data, source_text = resolve_input_to_sample_data(args)
    report = build_report(sample_data, source_text, args.neo4j_password)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    print_human_report(report)


if __name__ == "__main__":
    main()
