from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Iterable


REQUIRED_METADATA_TABLES = (
    "scene",
    "sample",
    "sample_data",
    "sample_annotation",
    "instance",
    "category",
    "log",
)


def normalize_path_key(path_value: str | Path | None) -> str:
    if path_value is None:
        return ""
    return str(path_value).replace("\\", "/").lstrip("./")


def iter_json_array_records(path: Path, chunk_size: int = 1 << 20):
    decoder = json.JSONDecoder()
    buffer = ""
    array_opened = False

    with path.open("r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(chunk_size)
            eof = not chunk
            if chunk:
                buffer += chunk

            while True:
                buffer = buffer.lstrip()
                if not array_opened:
                    if not buffer:
                        break
                    if buffer[0] != "[":
                        raise ValueError(f"{path} is not a JSON array.")
                    buffer = buffer[1:]
                    array_opened = True
                    continue

                if not buffer:
                    break
                if buffer[0] == "]":
                    return
                if buffer[0] == ",":
                    buffer = buffer[1:]
                    continue

                try:
                    record, offset = decoder.raw_decode(buffer)
                except json.JSONDecodeError as exc:
                    if eof:
                        raise ValueError(f"Failed to decode JSON array record from {path}: {exc}") from exc
                    break

                yield record
                buffer = buffer[offset:]

            if eof:
                break

    raise ValueError(f"{path} ended before the JSON array closed.")


def normalize_dataset_roots(dataset_root: Path | Iterable[Path] | None = None, dataset_roots: Iterable[Path] | None = None) -> list[Path]:
    roots: list[Path] = []
    if dataset_roots is not None:
        roots.extend(Path(root) for root in dataset_roots)
    elif dataset_root is not None:
        if isinstance(dataset_root, Path):
            roots.append(dataset_root)
        else:
            roots.extend(Path(root) for root in dataset_root)

    normalized_roots: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        root_key = str(root)
        if root_key in seen:
            continue
        seen.add(root_key)
        normalized_roots.append(root)
    return normalized_roots


def collect_available_camera_paths(dataset_root: Path | Iterable[Path] | None = None, dataset_roots: Iterable[Path] | None = None) -> set[str]:
    normalized_roots = normalize_dataset_roots(dataset_root=dataset_root, dataset_roots=dataset_roots)
    available_paths: set[str] = set()
    for root in normalized_roots:
        for top_level in ("samples", "sweeps"):
            base_dir = root / top_level
            if not base_dir.exists():
                continue

            for dirpath, _, filenames in os.walk(base_dir):
                if "CAM_" not in dirpath.replace("\\", "/"):
                    continue
                directory = Path(dirpath)
                for filename in filenames:
                    relative_path = directory.joinpath(filename).relative_to(root)
                    available_paths.add(relative_path.as_posix())

    return available_paths


def write_json_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def write_scene_tokens(path: Path, scene_tokens: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scene_tokens, ensure_ascii=False, indent=2), encoding="utf-8")


def write_subset_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def write_keyframe_csv(csv_path: Path, keyframe_paths: list[str], image_id_start: int) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "path"])
        for index, path_value in enumerate(keyframe_paths, start=image_id_start):
            writer.writerow([index, path_value])


def build_trainval_camera_subset(
    source_meta_dir: Path,
    output_meta_dir: Path,
    image_csv_path: Path,
    scene_token_path: Path,
    report_path: Path,
    image_id_start: int,
    dataset_root: Path | Iterable[Path] | None = None,
    dataset_roots: Iterable[Path] | None = None,
) -> dict:
    normalized_roots = normalize_dataset_roots(dataset_root=dataset_root, dataset_roots=dataset_roots)
    available_camera_paths = collect_available_camera_paths(dataset_roots=normalized_roots)
    if not available_camera_paths:
        joined_roots = ", ".join(str(root) for root in normalized_roots) or "<none>"
        raise RuntimeError(f"No camera files were found under {joined_roots}.")

    sample_data_records: list[dict] = []
    sample_tokens: set[str] = set()
    keyframe_paths: list[str] = []
    sweep_count = 0

    for record in iter_json_array_records(source_meta_dir / "sample_data.json"):
        filename = normalize_path_key(record.get("filename", ""))
        if filename not in available_camera_paths:
            continue

        sample_data_records.append(record)
        sample_tokens.add(str(record.get("sample_token", "")))
        if str(record.get("is_key_frame", False)).lower() == "true" or bool(record.get("is_key_frame", False)):
            if filename.startswith("samples/CAM_"):
                keyframe_paths.append(filename)
        else:
            sweep_count += 1

    sample_records: list[dict] = []
    scene_tokens: set[str] = set()
    for record in iter_json_array_records(source_meta_dir / "sample.json"):
        token = str(record.get("token", ""))
        if token not in sample_tokens:
            continue
        sample_records.append(record)
        scene_tokens.add(str(record.get("scene_token", "")))

    scene_records: list[dict] = []
    log_tokens: set[str] = set()
    for record in iter_json_array_records(source_meta_dir / "scene.json"):
        token = str(record.get("token", ""))
        if token not in scene_tokens:
            continue
        scene_records.append(record)
        log_tokens.add(str(record.get("log_token", "")))

    annotation_records: list[dict] = []
    instance_tokens: set[str] = set()
    for record in iter_json_array_records(source_meta_dir / "sample_annotation.json"):
        sample_token = str(record.get("sample_token", ""))
        if sample_token not in sample_tokens:
            continue
        annotation_records.append(record)
        instance_tokens.add(str(record.get("instance_token", "")))

    instance_records: list[dict] = []
    category_tokens: set[str] = set()
    for record in iter_json_array_records(source_meta_dir / "instance.json"):
        token = str(record.get("token", ""))
        if token not in instance_tokens:
            continue
        instance_records.append(record)
        category_tokens.add(str(record.get("category_token", "")))

    category_records = [
        record
        for record in iter_json_array_records(source_meta_dir / "category.json")
        if str(record.get("token", "")) in category_tokens
    ]
    log_records = [
        record for record in iter_json_array_records(source_meta_dir / "log.json") if str(record.get("token", "")) in log_tokens
    ]

    write_json_records(output_meta_dir / "scene.json", scene_records)
    write_json_records(output_meta_dir / "sample.json", sample_records)
    write_json_records(output_meta_dir / "sample_data.json", sample_data_records)
    write_json_records(output_meta_dir / "sample_annotation.json", annotation_records)
    write_json_records(output_meta_dir / "instance.json", instance_records)
    write_json_records(output_meta_dir / "category.json", category_records)
    write_json_records(output_meta_dir / "log.json", log_records)

    sorted_scene_tokens = sorted(token for token in scene_tokens if token)
    write_scene_tokens(scene_token_path, sorted_scene_tokens)
    write_keyframe_csv(image_csv_path, keyframe_paths, image_id_start=image_id_start)

    report = {
        "dataset_roots": [str(root) for root in normalized_roots],
        "source_meta_dir": str(source_meta_dir),
        "output_meta_dir": str(output_meta_dir),
        "image_csv_path": str(image_csv_path),
        "available_camera_files": len(available_camera_paths),
        "scene_count": len(sorted_scene_tokens),
        "sample_count": len(sample_records),
        "sample_data_count": len(sample_data_records),
        "keyframe_count": len(keyframe_paths),
        "sweep_count": sweep_count,
        "annotation_count": len(annotation_records),
        "instance_count": len(instance_records),
        "category_count": len(category_records),
        "log_count": len(log_records),
    }
    write_subset_report(report_path, report)
    return report
