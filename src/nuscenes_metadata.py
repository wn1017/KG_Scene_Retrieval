from __future__ import annotations

import csv
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path, PurePosixPath

from config import (
    NUSCENES_BLOB_ROOTS,
    NUSCENES_META_DIR,
    NUSCENES_ROOT,
    NUSCENES_SAMPLES_DIR,
    NUSCENES_SWEEPS_DIR,
)
from src.kg_builder import infer_location_kind, infer_time_of_day, infer_weather, normalize_object_name


def normalize_path_key(path_value: str | Path | None) -> str:
    if path_value is None:
        return ""
    return str(path_value).replace("\\", "/").lstrip("./")


def read_json_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def read_image_rows(csv_path: Path):
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            yield int(row["id"]), str(row["path"])


@lru_cache(maxsize=1)
def load_nuscenes_metadata(meta_dir: Path = NUSCENES_META_DIR) -> dict[str, object]:
    scene_records = read_json_records(meta_dir / "scene.json")
    sample_records = read_json_records(meta_dir / "sample.json")
    sample_data_records = read_json_records(meta_dir / "sample_data.json")
    sample_annotation_records = read_json_records(meta_dir / "sample_annotation.json")
    instance_records = read_json_records(meta_dir / "instance.json")
    category_records = read_json_records(meta_dir / "category.json")
    log_records = read_json_records(meta_dir / "log.json")

    scene_by_token = {record["token"]: record for record in scene_records}
    sample_by_token = {record["token"]: record for record in sample_records}
    log_by_token = {record["token"]: record for record in log_records}
    sample_to_scene = {record["token"]: record.get("scene_token", "") for record in sample_records}
    instance_to_category = {record["token"]: record.get("category_token", "") for record in instance_records}
    category_name_by_token = {record["token"]: record.get("name", "") for record in category_records}

    filename_to_sample_data: dict[str, dict] = {}
    basename_to_sample_data: dict[str, dict] = {}
    sample_data_by_token: dict[str, dict] = {}
    for record in sample_data_records:
        filename = normalize_path_key(record.get("filename", ""))
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        parts = PurePosixPath(filename).parts
        camera = parts[1] if len(parts) >= 3 else ""
        enriched = {
            "sample_data_token": record.get("token", ""),
            "sample_token": record.get("sample_token", ""),
            "scene_token": sample_to_scene.get(record.get("sample_token", ""), ""),
            "timestamp": int(record.get("timestamp", 0) or 0),
            "filename": filename,
            "camera": camera,
            "is_key_frame": bool(record.get("is_key_frame", False)),
            "prev": record.get("prev", ""),
            "next": record.get("next", ""),
        }
        filename_to_sample_data[filename] = enriched
        basename_to_sample_data[PurePosixPath(filename).name] = enriched
        sample_data_by_token[enriched["sample_data_token"]] = enriched

    annotations_by_sample: dict[str, list[dict]] = defaultdict(list)
    for annotation in sample_annotation_records:
        annotations_by_sample[annotation.get("sample_token", "")].append(annotation)

    return {
        "scene_by_token": scene_by_token,
        "sample_by_token": sample_by_token,
        "log_by_token": log_by_token,
        "filename_to_sample_data": filename_to_sample_data,
        "basename_to_sample_data": basename_to_sample_data,
        "sample_data_by_token": sample_data_by_token,
        "annotations_by_sample": annotations_by_sample,
        "instance_to_category": instance_to_category,
        "category_name_by_token": category_name_by_token,
    }


def infer_camera_from_path(path_value: str | Path | None) -> str:
    normalized = normalize_path_key(path_value)
    for part in PurePosixPath(normalized).parts:
        if part.startswith("CAM_"):
            return part
    return ""


def get_blob_roots() -> list[Path]:
    roots = [Path(root) for root in NUSCENES_BLOB_ROOTS]
    if not roots:
        roots = [NUSCENES_ROOT]
    return roots


def resolve_frame_path(raw_path: str | Path | None, metadata_index: dict[str, object] | None = None) -> Path | None:
    if raw_path is None:
        return None

    metadata_index = metadata_index or load_nuscenes_metadata()
    normalized = normalize_path_key(raw_path)
    if not normalized:
        return None

    candidates: list[Path] = [Path(normalized), Path.cwd() / normalized]
    blob_roots = get_blob_roots()
    parts = PurePosixPath(normalized).parts
    if parts:
        if parts[0] in {"samples", "sweeps"}:
            for blob_root in blob_roots:
                candidates.append(blob_root.joinpath(*parts))
        elif parts[0] == "img_data" and len(parts) >= 3:
            camera = parts[1]
            filename = parts[-1]
            for blob_root in blob_roots:
                candidates.append(blob_root / "samples" / camera / filename)
                candidates.append(blob_root / "sweeps" / camera / filename)
        elif parts[0].startswith("CAM_"):
            camera = parts[0]
            filename = parts[-1]
            for blob_root in blob_roots:
                candidates.append(blob_root / "samples" / camera / filename)
                candidates.append(blob_root / "sweeps" / camera / filename)

    basename = PurePosixPath(normalized).name
    sample_data = metadata_index["basename_to_sample_data"].get(basename)
    if sample_data:
        for blob_root in blob_roots:
            candidates.append(blob_root / sample_data["filename"])

    seen: set[str] = set()
    for candidate in candidates:
        candidate_key = str(candidate)
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        if candidate.exists():
            return candidate
    return None


def lookup_sample_data(
    raw_path: str | Path | None,
    resolved_path: Path | None = None,
    metadata_index: dict[str, object] | None = None,
) -> dict | None:
    metadata_index = metadata_index or load_nuscenes_metadata()
    candidate_keys: list[str] = []

    if raw_path:
        normalized = normalize_path_key(raw_path)
        candidate_keys.append(normalized)
        candidate_keys.append(PurePosixPath(normalized).name)

    if resolved_path:
        candidate_keys.append(resolved_path.name)
        for blob_root in get_blob_roots():
            try:
                candidate_keys.append(normalize_path_key(resolved_path.relative_to(blob_root)))
            except ValueError:
                continue

    for key in candidate_keys:
        sample_data = metadata_index["filename_to_sample_data"].get(key)
        if sample_data:
            return sample_data
        sample_data = metadata_index["basename_to_sample_data"].get(key)
        if sample_data:
            return sample_data
    return None


def collect_sample_object_types(sample_token: str, metadata_index: dict[str, object] | None = None) -> list[str]:
    metadata_index = metadata_index or load_nuscenes_metadata()
    object_names: set[str] = set()
    for annotation in metadata_index["annotations_by_sample"].get(sample_token, []):
        category_token = metadata_index["instance_to_category"].get(annotation.get("instance_token", ""), "")
        category_name = metadata_index["category_name_by_token"].get(category_token, "")
        if category_name:
            object_names.add(normalize_object_name(category_name))
    return sorted(object_names)


def build_scene_metadata(scene_token: str, metadata_index: dict[str, object] | None = None) -> dict[str, str]:
    metadata_index = metadata_index or load_nuscenes_metadata()
    scene_record = metadata_index["scene_by_token"].get(scene_token, {})
    log_record = metadata_index["log_by_token"].get(scene_record.get("log_token", ""), {})
    description = scene_record.get("description", "")
    location_area = log_record.get("location", "")
    location_kind = infer_location_kind(description)
    location_value = f"{location_area}:{location_kind}".strip(":")

    return {
        "weather": infer_weather(description),
        "timeofday": infer_time_of_day(description),
        "location": location_value,
        "scene_name": scene_record.get("name", ""),
        "scene_description": description,
    }


def get_image_metadata(raw_path: str | Path, metadata_index: dict[str, object] | None = None) -> dict[str, str]:
    metadata_index = metadata_index or load_nuscenes_metadata()
    resolved_path = resolve_frame_path(raw_path, metadata_index)
    sample_data = lookup_sample_data(raw_path, resolved_path, metadata_index)

    frame_path = normalize_path_key(raw_path)
    scene_token = ""
    sample_token = ""
    camera = infer_camera_from_path(raw_path)
    weather = ""
    timeofday = ""
    location = ""
    obj_types = ""

    if sample_data:
        scene_token = sample_data.get("scene_token", "")
        sample_token = sample_data.get("sample_token", "")
        camera = sample_data.get("camera", "") or camera
        frame_path = sample_data.get("filename", "") or frame_path

    if scene_token:
        scene_metadata = build_scene_metadata(scene_token, metadata_index)
        weather = scene_metadata["weather"]
        timeofday = scene_metadata["timeofday"]
        location = scene_metadata["location"]

    if sample_token:
        obj_types = ",".join(collect_sample_object_types(sample_token, metadata_index))

    if resolved_path and not frame_path:
        for blob_root in get_blob_roots():
            try:
                frame_path = normalize_path_key(resolved_path.relative_to(blob_root))
                break
            except ValueError:
                continue
        else:
            frame_path = normalize_path_key(resolved_path)

    return {
        "scene_token": scene_token,
        "sample_token": sample_token,
        "camera": camera,
        "frame_path": frame_path,
        "weather": weather,
        "timeofday": timeofday,
        "location": location,
        "obj_types": obj_types,
        "resolved_path": str(resolved_path) if resolved_path else "",
    }
