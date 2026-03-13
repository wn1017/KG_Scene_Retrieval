from __future__ import annotations

from collections import OrderedDict
import json
import os
from functools import lru_cache
from html import escape
import re
import time
from pathlib import Path, PurePosixPath

import cv2
import gradio as gr
from gradio.processing_utils import convert_video_to_playable_mp4, ffmpeg_installed
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, ChineseCLIPModel, ChineseCLIPProcessor

from config import (
    CHNCLIP_MODEL_DIR,
    DEFAULT_TOP_K,
    ENABLE_RETRIEVAL_TIMINGS,
    ENGCLIP_MODEL_DIR,
    GENERATED_VIDEO_DIR,
    HIT_METADATA_CACHE_SIZE,
    IMAGE_CSV_PATH,
    IMAGE_ID_MIN,
    NUSCENES_META_DIR,
    NUSCENES_ROOT,
    NUSCENES_SAMPLES_DIR,
    NUSCENES_SWEEPS_DIR,
    PRIMARY_CAMERA,
    VIDEO_CLIP_CACHE_SIZE,
    VIDEO_FPS,
    VIDEO_FRAME_STRIDE,
    VIDEO_MAX_FRAMES,
    VIDEO_OUTPUT_MAX_WIDTH,
    VIDEO_RESULT_COUNT,
    VIDEO_SEARCH_LIMIT,
    VIDEO_TRANSITION_FRAMES,
)
from src.milvus_utils import SEARCH_PARAMS, collection as DEFAULT_COLLECTION, get_or_create_collection, get_search_output_fields, has_field, schema_needs_rebuild
from src.kg_builder import build_scene_records, filter_scene_records, query_scene_tokens
from src.nlp_parser import parse_query


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_RESULT_COUNT = DEFAULT_TOP_K


def normalize_path_key(path_value: str | Path | None) -> str:
    if path_value is None:
        return ""
    return str(path_value).replace("\\", "/").lstrip("./")


def read_json_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def load_image_catalog() -> tuple[pd.DataFrame, dict[int, str]]:
    if not IMAGE_CSV_PATH.exists():
        return pd.DataFrame(columns=["id", "path"]), {}

    dataframe = pd.read_csv(IMAGE_CSV_PATH)
    if not {"id", "path"}.issubset(dataframe.columns):
        return pd.DataFrame(columns=["id", "path"]), {}

    dataframe = dataframe[["id", "path"]].copy()
    dataframe["id"] = dataframe["id"].astype(int)
    dataframe["path"] = dataframe["path"].astype(str)
    return dataframe, dataframe.set_index("id")["path"].to_dict()


def load_nuscenes_index() -> tuple[dict, dict, dict, dict]:
    scene_records = read_json_records(NUSCENES_META_DIR / "scene.json")
    sample_records = read_json_records(NUSCENES_META_DIR / "sample.json")
    sample_data_records = read_json_records(NUSCENES_META_DIR / "sample_data.json")

    scene_by_token = {record["token"]: record for record in scene_records}
    sample_to_scene = {record["token"]: record.get("scene_token", "") for record in sample_records}
    filename_to_sample_data: dict[str, dict] = {}
    basename_to_sample_data: dict[str, dict] = {}
    camera_sequences: dict[tuple[str, str], list[dict]] = {}

    for record in sample_data_records:
        filename = normalize_path_key(record.get("filename", ""))
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        parts = PurePosixPath(filename).parts
        camera = parts[1] if len(parts) >= 3 else ""
        scene_token = sample_to_scene.get(record.get("sample_token", ""), "")
        enriched_record = {
            "sample_data_token": record.get("token", ""),
            "sample_token": record.get("sample_token", ""),
            "scene_token": scene_token,
            "timestamp": int(record.get("timestamp", 0) or 0),
            "filename": filename,
            "camera": camera,
            "is_key_frame": bool(record.get("is_key_frame", False)),
            "prev": record.get("prev", ""),
            "next": record.get("next", ""),
        }
        filename_to_sample_data[filename] = enriched_record
        basename_to_sample_data[PurePosixPath(filename).name] = enriched_record

        if scene_token and camera:
            sequence_key = (scene_token, camera)
            camera_sequences.setdefault(sequence_key, []).append(enriched_record)

    for sequence in camera_sequences.values():
        sequence.sort(key=lambda item: item["timestamp"])

    return scene_by_token, filename_to_sample_data, basename_to_sample_data, camera_sequences


IMAGE_DF, ID_TO_RAW_PATH = load_image_catalog()
SCENE_BY_TOKEN, FILENAME_TO_SAMPLE_DATA, BASENAME_TO_SAMPLE_DATA, CAMERA_SEQUENCES = load_nuscenes_index()
KG_SCENE_RECORDS = build_scene_records(NUSCENES_META_DIR)
KG_RECORD_BY_SCENE_TOKEN = {record["scene_token"]: record for record in KG_SCENE_RECORDS}
KNOWN_SCENE_TOKENS = set(SCENE_BY_TOKEN) | set(KG_RECORD_BY_SCENE_TOKEN)


def infer_camera_from_path(path_value: str | Path | None) -> str:
    normalized = normalize_path_key(path_value)
    for part in PurePosixPath(normalized).parts:
        if part.startswith("CAM_"):
            return part
    return ""


def resolve_frame_path(raw_path: str | Path | None) -> Path | None:
    if raw_path is None:
        return None

    normalized = normalize_path_key(raw_path)
    if not normalized:
        return None

    candidates: list[Path] = []
    direct_path = Path(normalized)
    candidates.append(direct_path)
    candidates.append(Path.cwd() / normalized)

    parts = PurePosixPath(normalized).parts
    if parts:
        if parts[0] in {"samples", "sweeps"}:
            candidates.append(NUSCENES_ROOT.joinpath(*parts))
        elif parts[0] == "img_data" and len(parts) >= 3:
            camera = parts[1]
            filename = parts[-1]
            candidates.append(NUSCENES_SAMPLES_DIR / camera / filename)
            candidates.append(NUSCENES_SWEEPS_DIR / camera / filename)
        elif parts[0].startswith("CAM_"):
            camera = parts[0]
            filename = parts[-1]
            candidates.append(NUSCENES_SAMPLES_DIR / camera / filename)
            candidates.append(NUSCENES_SWEEPS_DIR / camera / filename)

    basename = PurePosixPath(normalized).name
    metadata = BASENAME_TO_SAMPLE_DATA.get(basename)
    if metadata:
        candidates.append(NUSCENES_ROOT / metadata["filename"])

    seen: set[str] = set()
    for candidate in candidates:
        candidate_key = str(candidate)
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        if candidate.exists():
            return candidate
    return None


def get_sample_data_for_frame(resolved_path: Path | None, raw_path: str | Path | None = None) -> dict | None:
    candidate_keys: list[str] = []

    if raw_path:
        normalized = normalize_path_key(raw_path)
        candidate_keys.append(normalized)
        candidate_keys.append(PurePosixPath(normalized).name)

    if resolved_path:
        candidate_keys.append(resolved_path.name)
        try:
            relative_key = normalize_path_key(resolved_path.relative_to(NUSCENES_ROOT))
            candidate_keys.append(relative_key)
        except ValueError:
            pass

    for key in candidate_keys:
        if key in FILENAME_TO_SAMPLE_DATA:
            return FILENAME_TO_SAMPLE_DATA[key]
        if key in BASENAME_TO_SAMPLE_DATA:
            return BASENAME_TO_SAMPLE_DATA[key]
    return None


def trim_text(text: str | None, limit: int = 90) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def load_models() -> tuple[dict[str, object], str | None]:
    try:
        eng_processor = CLIPProcessor.from_pretrained(str(ENGCLIP_MODEL_DIR))
        eng_model = CLIPModel.from_pretrained(str(ENGCLIP_MODEL_DIR)).to(DEVICE).eval()
        chn_processor = ChineseCLIPProcessor.from_pretrained(str(CHNCLIP_MODEL_DIR))
        chn_model = ChineseCLIPModel.from_pretrained(str(CHNCLIP_MODEL_DIR)).to(DEVICE).eval()
    except Exception as exc:
        return {}, str(exc)

    return {
        "eng_processor": eng_processor,
        "eng_model": eng_model,
        "chn_processor": chn_processor,
        "chn_model": chn_model,
    }, None


MODELS, MODEL_LOAD_ERROR = load_models()


def load_collection_state():
    if DEFAULT_COLLECTION is not None:
        return DEFAULT_COLLECTION, None
    try:
        return get_or_create_collection(drop_existing=False), None
    except Exception as exc:
        return None, str(exc)


COLLECTION, MILVUS_ERROR = load_collection_state()
COLLECTION_LOAD_CACHE_IDS: set[int] = set()
VIDEO_CLIP_CACHE: OrderedDict[tuple[str, ...], str] = OrderedDict()


def clear_runtime_caches() -> None:
    COLLECTION_LOAD_CACHE_IDS.clear()
    VIDEO_CLIP_CACHE.clear()
    get_cached_hit_metadata.cache_clear()


def ensure_collection_loaded(active_collection) -> None:
    collection_key = id(active_collection)
    if collection_key in COLLECTION_LOAD_CACHE_IDS:
        return
    active_collection.load()
    COLLECTION_LOAD_CACHE_IDS.add(collection_key)


def get_live_collection(force_refresh: bool = False):
    global COLLECTION, MILVUS_ERROR
    if force_refresh or COLLECTION is None:
        COLLECTION_LOAD_CACHE_IDS.clear()
        COLLECTION, MILVUS_ERROR = load_collection_state()
    return COLLECTION


@lru_cache(maxsize=HIT_METADATA_CACHE_SIZE)
def get_cached_hit_metadata(record_id: int, raw_path: str, scene_token_hint: str = "", camera_hint: str = "") -> tuple[tuple[str, object], ...]:
    del record_id

    resolved_path = resolve_frame_path(raw_path)
    sample_data = get_sample_data_for_frame(resolved_path, raw_path)
    metadata: dict[str, object] = {
        "raw_frame_path": raw_path,
        "resolved_frame_path": str(resolved_path) if resolved_path else "",
    }

    inferred_camera = camera_hint or infer_camera_from_path(raw_path) or infer_camera_from_path(resolved_path)
    if inferred_camera:
        metadata["camera"] = inferred_camera
    if scene_token_hint:
        metadata["scene_token"] = scene_token_hint

    if sample_data:
        if sample_data.get("sample_token"):
            metadata["sample_token"] = sample_data.get("sample_token", "")
        if sample_data.get("scene_token"):
            metadata["scene_token"] = sample_data.get("scene_token", "")
        if sample_data.get("camera"):
            metadata["camera"] = sample_data.get("camera", "")
        if sample_data.get("filename"):
            metadata["source_filename"] = sample_data.get("filename", "")
        if sample_data.get("sample_data_token"):
            metadata["sample_data_token"] = sample_data.get("sample_data_token", "")

    scene_token = str(metadata.get("scene_token", "") or "")
    scene_record = SCENE_BY_TOKEN.get(scene_token, {}) if scene_token else {}
    kg_scene_record = KG_RECORD_BY_SCENE_TOKEN.get(scene_token, {}) if scene_token else {}

    if scene_record.get("name") or kg_scene_record.get("scene_name"):
        metadata["scene_name"] = scene_record.get("name") or kg_scene_record.get("scene_name", "")
    if scene_record.get("description") or kg_scene_record.get("description"):
        metadata["scene_description"] = scene_record.get("description") or kg_scene_record.get("description", "")
    if kg_scene_record:
        if kg_scene_record.get("weather"):
            metadata["weather"] = kg_scene_record.get("weather", "")
        if kg_scene_record.get("timeofday"):
            metadata["timeofday"] = kg_scene_record.get("timeofday", "")
        if kg_scene_record.get("location_area") or kg_scene_record.get("location_kind"):
            metadata["location"] = f"{kg_scene_record.get('location_area', '')}:{kg_scene_record.get('location_kind', '')}".strip(":")
        objects = kg_scene_record.get("objects", {})
        if objects:
            metadata["obj_types"] = ", ".join(objects.keys())

    return tuple(metadata.items())


def format_timing_breakdown(timings: dict[str, float]) -> str:
    if not ENABLE_RETRIEVAL_TIMINGS:
        return ""
    timing_labels = {
        "nlp": "NLP",
        "kg": "知识图谱",
        "clip": "CLIP",
        "milvus": "Milvus",
        "decode": "解码",
        "group": "分组",
        "frames": "取帧",
        "clips": "生成视频",
    }
    formatted = ", ".join(
        f"{timing_labels.get(name, name)}={duration * 1000:.0f}ms" for name, duration in timings.items()
    )
    return f"耗时：{formatted}" if formatted else ""


def append_timing_breakdown(status: str, timings: dict[str, float]) -> str:
    timing_note = format_timing_breakdown(timings)
    return f"{status} | {timing_note}" if timing_note else status


def build_video_clip_cache_key(anchor_record: dict, frame_paths: list[Path]) -> tuple[str, ...]:
    if not frame_paths:
        return (
            str(anchor_record.get("scene_token", "") or anchor_record.get("scene_name", "")),
            str(anchor_record.get("camera", PRIMARY_CAMERA)),
            "empty",
        )
    return (
        str(anchor_record.get("scene_token", "") or ""),
        str(anchor_record.get("scene_name", "") or ""),
        str(anchor_record.get("camera", PRIMARY_CAMERA)),
        str(frame_paths[0]),
        str(frame_paths[-1]),
        str(len(frame_paths)),
        str(VIDEO_FPS),
        str(VIDEO_MAX_FRAMES),
        str(VIDEO_FRAME_STRIDE),
        str(VIDEO_TRANSITION_FRAMES),
        str(VIDEO_OUTPUT_MAX_WIDTH),
    )


def get_cached_video_clip(cache_key: tuple[str, ...]) -> Path | None:
    cached_path = VIDEO_CLIP_CACHE.get(cache_key)
    if not cached_path:
        return None
    clip_path = Path(cached_path)
    if clip_path.exists() and is_browser_playable_clip(clip_path):
        VIDEO_CLIP_CACHE.move_to_end(cache_key)
        return clip_path
    VIDEO_CLIP_CACHE.pop(cache_key, None)
    return None


def remember_video_clip(cache_key: tuple[str, ...], clip_path: Path) -> None:
    VIDEO_CLIP_CACHE[cache_key] = str(clip_path)
    VIDEO_CLIP_CACHE.move_to_end(cache_key)
    while len(VIDEO_CLIP_CACHE) > VIDEO_CLIP_CACHE_SIZE:
        VIDEO_CLIP_CACHE.popitem(last=False)


STARTUP_MESSAGES = []
if MODEL_LOAD_ERROR:
    STARTUP_MESSAGES.append(f"模型加载失败：{MODEL_LOAD_ERROR}")
else:
    STARTUP_MESSAGES.append(f"模型已加载：{DEVICE.type}")
STARTUP_MESSAGES.append(f"图片索引：{len(ID_TO_RAW_PATH)} 条")
STARTUP_MESSAGES.append(
    f"nuScenes 元数据：{len(SCENE_BY_TOKEN)} 个场景，{len(FILENAME_TO_SAMPLE_DATA)} 条帧记录"
)
STARTUP_MESSAGES.append(f"知识图谱场景记录：{len(KG_SCENE_RECORDS)} 条")
if COLLECTION is None:
    STARTUP_MESSAGES.append(f"Milvus 不可用：{MILVUS_ERROR or '连接失败'}")
else:
    STARTUP_MESSAGES.append("Milvus 集合已连接")
    if schema_needs_rebuild(COLLECTION):
        STARTUP_MESSAGES.append(
            "当前集合仍使用旧版 schema；在重建前，元数据过滤能力可能受限。"
        )
INITIAL_STATUS = " | ".join(STARTUP_MESSAGES)


def format_parsed_query(parsed_query: dict) -> str:
    parts = []
    if parsed_query.get("weather"):
        parts.append(f"天气={parsed_query['weather']}")
    if parsed_query.get("time"):
        parts.append(f"时段={parsed_query['time']}")
    if parsed_query.get("location"):
        parts.append(f"位置={parsed_query['location']}")
    if parsed_query.get("objects"):
        parts.append("对象=" + ",".join(parsed_query["objects"]))
    if not parts:
        return "未提取到结构化条件。"
    return "; ".join(parts)




def get_local_candidate_scene_tokens(
    weather: str | None,
    timeofday: str | None,
    object_types: list[str],
    location_kind: str | None,
) -> list[str]:
    return filter_scene_records(
        KG_SCENE_RECORDS,
        weather=weather,
        timeofday=timeofday,
        object_types=object_types,
        location_kind=location_kind,
    )


def sanitize_candidate_scene_tokens(scene_tokens: list[str] | None) -> list[str]:
    sanitized_tokens: list[str] = []
    seen: set[str] = set()
    for token in scene_tokens or []:
        normalized_token = str(token or "").strip()
        if not normalized_token or normalized_token in seen:
            continue
        seen.add(normalized_token)
        sanitized_tokens.append(normalized_token)
    return sanitized_tokens


def get_candidate_scene_tokens(parsed_query: dict) -> tuple[list[str], str]:
    weather = parsed_query.get("weather")
    timeofday = parsed_query.get("time")
    object_types = parsed_query.get("objects") or []
    location_kind = parsed_query.get("location")

    if not any([weather, timeofday, object_types, location_kind]):
        return [], "未提取到知识图谱过滤条件，已回退到全库搜索。"

    try:
        neo4j_tokens = sanitize_candidate_scene_tokens(
            query_scene_tokens(
                weather=weather,
                timeofday=timeofday,
                object_types=object_types,
                location_kind=location_kind,
            )
        )
        valid_neo4j_tokens = [token for token in neo4j_tokens if token in KNOWN_SCENE_TOKENS]
        if valid_neo4j_tokens:
            if len(valid_neo4j_tokens) != len(neo4j_tokens):
                ignored_count = len(neo4j_tokens) - len(valid_neo4j_tokens)
                return valid_neo4j_tokens, (
                    f"Neo4j 已过滤出 {len(valid_neo4j_tokens)} 个场景，并忽略了 {ignored_count} 个无效 token。"
                )
            return valid_neo4j_tokens, f"Neo4j 已过滤出 {len(valid_neo4j_tokens)} 个场景。"
        if neo4j_tokens:
            local_tokens = get_local_candidate_scene_tokens(
                weather=weather,
                timeofday=timeofday,
                object_types=object_types,
                location_kind=location_kind,
            )
            if local_tokens:
                return local_tokens, f"Neo4j 返回了库外 token，已由本地知识图谱过滤出 {len(local_tokens)} 个场景。"
            return [], "Neo4j 返回了库外 scene token，已回退到全库搜索。"
        return [], "Neo4j 未找到候选场景，已回退到全库搜索。"
    except RuntimeError:
        local_tokens = get_local_candidate_scene_tokens(
            weather=weather,
            timeofday=timeofday,
            object_types=object_types,
            location_kind=location_kind,
        )
        if local_tokens:
                return local_tokens, f"Neo4j 不可用，已由本地知识图谱过滤出 {len(local_tokens)} 个场景。"
        return [], "Neo4j 不可用且本地知识图谱未找到候选场景，已回退到全库搜索。"


def build_scene_filter_expr(scene_tokens: list[str]) -> str:
    escaped_tokens = [token.replace('\\', '\\\\').replace('"', '\\"') for token in scene_tokens]
    joined_tokens = '\", \"'.join(escaped_tokens)
    return f'scene_token in ["{joined_tokens}"]'


def encode_text_query(text: str) -> tuple[np.ndarray, str]:
    if MODEL_LOAD_ERROR:
        raise RuntimeError(f"Models are unavailable: {MODEL_LOAD_ERROR}")

    if contains_chinese(text):
        processor = MODELS["chn_processor"]
        model = MODELS["chn_model"]
        model_name = "Chinese-CLIP"
    else:
        processor = MODELS["eng_processor"]
        model = MODELS["eng_model"]
        model_name = "English-CLIP"

    tokenizer = getattr(processor, "tokenizer", None)
    model_max_length = getattr(tokenizer, "model_max_length", None)
    if not isinstance(model_max_length, int) or model_max_length > 10000:
        text_config = getattr(model.config, "text_config", None)
        model_max_length = getattr(text_config, "max_position_embeddings", None)

    processor_kwargs = {"text": [text], "return_tensors": "pt", "padding": True}
    if isinstance(model_max_length, int) and model_max_length > 0:
        processor_kwargs["truncation"] = True
        processor_kwargs["max_length"] = model_max_length
    encoded_inputs = processor(**processor_kwargs)
    encoded_inputs = {key: value.to(DEVICE) for key, value in encoded_inputs.items()}
    with torch.inference_mode():
        features = model.get_text_features(**encoded_inputs)
        features = torch.nn.functional.normalize(features, dim=-1)
    return features[0].detach().cpu().numpy().astype(np.float32), model_name


def extract_hit_record(hit, output_fields: list[str]) -> dict:
    record = {
        "id": int(hit.id),
        "score": float(hit.score),
    }
    entity = getattr(hit, "entity", None)
    for field in output_fields:
        if entity is not None:
            record[field] = entity.get(field)
    return record


def enrich_hit_record(record: dict) -> dict:
    raw_path = str(record.get("frame_path") or ID_TO_RAW_PATH.get(record["id"], "") or "")
    metadata = dict(
        get_cached_hit_metadata(
            int(record["id"]),
            raw_path,
            str(record.get("scene_token") or ""),
            str(record.get("camera") or ""),
        )
    )

    enriched_record = dict(record)
    for key, value in metadata.items():
        if key == "raw_frame_path":
            enriched_record[key] = value
            continue
        if not enriched_record.get(key):
            enriched_record[key] = value
    return enriched_record


def search_frame_hits(query_vector: np.ndarray, limit: int, candidate_scene_tokens: list[str] | None = None) -> list[dict]:
    global COLLECTION, MILVUS_ERROR

    candidate_scene_tokens = sanitize_candidate_scene_tokens(candidate_scene_tokens)

    def _run_search(active_collection) -> list[dict]:
        ensure_collection_loaded(active_collection)
        if active_collection.num_entities == 0:
            return []

        use_scene_filter = bool(candidate_scene_tokens) and has_field(active_collection, "scene_token")
        output_fields = get_search_output_fields(active_collection)
        base_expr = f"id >= {IMAGE_ID_MIN}"
        expr = base_expr
        search_limit = max(limit * 8, 80) if candidate_scene_tokens else limit
        if use_scene_filter:
            expr = f"{base_expr} and {build_scene_filter_expr(candidate_scene_tokens)}"

        allowed_scene_tokens = set(candidate_scene_tokens)
        query_payload = [query_vector.tolist()]

        def _search(expr_value: str, search_limit_value: int):
            return active_collection.search(
                data=query_payload,
                anns_field="embedding",
                param=SEARCH_PARAMS,
                limit=search_limit_value,
                expr=expr_value,
                output_fields=output_fields,
            )

        def _collect_hits(search_result, filter_by_scene: bool, existing_ids: set[int] | None = None) -> list[dict]:
            existing_ids = existing_ids or set()
            collected_hits: list[dict] = []
            for hit in search_result[0]:
                enriched_record = enrich_hit_record(extract_hit_record(hit, output_fields))
                if filter_by_scene and enriched_record.get("scene_token") not in allowed_scene_tokens:
                    continue
                if enriched_record["id"] in existing_ids:
                    continue
                existing_ids.add(enriched_record["id"])
                collected_hits.append(enriched_record)
                if len(collected_hits) >= limit:
                    break
            return collected_hits

        search_result = _search(expr, search_limit)
        enriched_hits = _collect_hits(search_result, filter_by_scene=bool(candidate_scene_tokens and not use_scene_filter))
        if candidate_scene_tokens and len(enriched_hits) < limit:
            existing_ids = {record["id"] for record in enriched_hits}
            fallback_result = _search(base_expr, max(search_limit * 4, 240))
            for enriched_record in _collect_hits(fallback_result, filter_by_scene=True, existing_ids=existing_ids):
                enriched_hits.append(enriched_record)
                if len(enriched_hits) >= limit:
                    break
        return enriched_hits[:limit]

    active_collection = get_live_collection()
    if active_collection is None:
        raise RuntimeError(MILVUS_ERROR or "Milvus is unavailable")

    try:
        return _run_search(active_collection)
    except Exception as first_exc:
        COLLECTION = None
        MILVUS_ERROR = str(first_exc)
        COLLECTION_LOAD_CACHE_IDS.clear()
        refreshed_collection = get_live_collection(force_refresh=True)
        if refreshed_collection is None:
            raise RuntimeError(MILVUS_ERROR or "Milvus is unavailable") from first_exc
        try:
            return _run_search(refreshed_collection)
        except Exception as second_exc:
            COLLECTION = None
            MILVUS_ERROR = str(second_exc)
            COLLECTION_LOAD_CACHE_IDS.clear()
            raise RuntimeError(f"Milvus collection unavailable: {second_exc}") from second_exc

def build_image_caption(
    record: dict,
    parsed_query: dict | None = None,
    kg_status: str = "",
    model_name: str = "",
) -> str:
    caption_parts = [("相似度", f"{record['score']:.4f}")]
    reason_summary = " / ".join(build_result_reason_tags(record, parsed_query, kg_status, model_name))
    if reason_summary:
        caption_parts.append(("命中条件", reason_summary))
    if record.get("scene_name"):
        caption_parts.append(("场景", str(record["scene_name"])))
    elif record.get("scene_token"):
        caption_parts.append(("场景", str(record["scene_token"])))
    if record.get("camera"):
        caption_parts.append(("相机", str(record["camera"])))
    return build_result_details_html(caption_parts[:4])


def retrieve_images(text: str) -> tuple[list[tuple[Image.Image, str]], str, dict, str]:
    timings: dict[str, float] = {}

    stage_started_at = time.perf_counter()
    parsed_query = parse_query(text)
    timings["nlp"] = time.perf_counter() - stage_started_at

    stage_started_at = time.perf_counter()
    candidate_scene_tokens, kg_status = get_candidate_scene_tokens(parsed_query)
    timings["kg"] = time.perf_counter() - stage_started_at

    stage_started_at = time.perf_counter()
    query_vector, model_name = encode_text_query(text)
    timings["clip"] = time.perf_counter() - stage_started_at

    stage_started_at = time.perf_counter()
    hits = search_frame_hits(query_vector, IMAGE_RESULT_COUNT, candidate_scene_tokens)
    timings["milvus"] = time.perf_counter() - stage_started_at

    stage_started_at = time.perf_counter()
    image_results: list[tuple[Image.Image, str]] = []
    for hit in hits:
        resolved_path = hit.get("resolved_frame_path", "")
        if not resolved_path:
            continue
        image = Image.open(resolved_path).convert("RGB")
        image_results.append((image, build_image_caption(hit, parsed_query, kg_status, model_name)))
        if len(image_results) >= IMAGE_RESULT_COUNT:
            break
    timings["decode"] = time.perf_counter() - stage_started_at

    return image_results, model_name, parsed_query, append_timing_breakdown(kg_status, timings)


def derive_sequence_group_key(record: dict) -> tuple[str, str]:
    camera = record.get("camera") or PRIMARY_CAMERA
    if record.get("scene_token"):
        return record["scene_token"], camera

    frame_name = Path(record.get("resolved_frame_path") or record.get("raw_frame_path") or str(record["id"])).name
    if "__CAM_" in frame_name:
        return frame_name.split("__CAM_")[0], camera
    return str(record["id"]), camera


def collect_video_frames(anchor_record: dict) -> list[Path]:
    resolved_path = Path(anchor_record["resolved_frame_path"]) if anchor_record.get("resolved_frame_path") else None
    sample_data = get_sample_data_for_frame(resolved_path, anchor_record.get("raw_frame_path"))

    if not sample_data:
        return [resolved_path] if resolved_path and resolved_path.exists() else []

    scene_token = anchor_record.get("scene_token") or sample_data.get("scene_token", "")
    camera = anchor_record.get("camera") or sample_data.get("camera", "") or PRIMARY_CAMERA
    sequence = CAMERA_SEQUENCES.get((scene_token, camera), [])
    if not sequence:
        return [resolved_path] if resolved_path and resolved_path.exists() else []

    anchor_token = sample_data.get("sample_data_token", "")
    anchor_index = None
    for index, item in enumerate(sequence):
        if item.get("sample_data_token") == anchor_token:
            anchor_index = index
            break

    if anchor_index is None:
        anchor_timestamp = sample_data.get("timestamp", 0)
        anchor_index = min(range(len(sequence)), key=lambda idx: abs(sequence[idx]["timestamp"] - anchor_timestamp))

    half_window = VIDEO_MAX_FRAMES // 2
    start_index = max(0, anchor_index - half_window)
    end_index = min(len(sequence), start_index + VIDEO_MAX_FRAMES)
    start_index = max(0, end_index - VIDEO_MAX_FRAMES)

    frame_paths = []
    seen_paths: set[str] = set()
    for item in sequence[start_index:end_index:VIDEO_FRAME_STRIDE]:
        frame_path = NUSCENES_ROOT / item["filename"]
        normalized_path = str(frame_path)
        if frame_path.exists() and normalized_path not in seen_paths:
            frame_paths.append(frame_path)
            seen_paths.add(normalized_path)

    return frame_paths


def resize_video_frame(frame: np.ndarray) -> np.ndarray:
    if not VIDEO_OUTPUT_MAX_WIDTH:
        return frame
    height, width = frame.shape[:2]
    if width <= VIDEO_OUTPUT_MAX_WIDTH:
        return frame
    scaled_height = max(1, int(round(height * (VIDEO_OUTPUT_MAX_WIDTH / width))))
    return cv2.resize(frame, (VIDEO_OUTPUT_MAX_WIDTH, scaled_height), interpolation=cv2.INTER_AREA)


def prepare_video_render_frames(frame_paths: list[Path]) -> list[np.ndarray]:
    render_frames: list[np.ndarray] = []
    previous_path = ""

    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        frame = resize_video_frame(frame)
        current_path = str(frame_path)
        if current_path == previous_path:
            continue

        render_frames.append(frame)
        previous_path = current_path

    if len(render_frames) == 1:
        render_frames = render_frames * max(2, VIDEO_FPS * 2)
    return render_frames


def sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return cleaned or "retrieval_clip"


BROWSER_PLAYABLE_VIDEO_TAGS = {
    ".mp4": {"avc1", "AVC1", "avc3", "h264", "H264", "x264", "X264"},
    ".webm": {"VP80", "VP90", "vp80", "vp90"},
    ".ogg": {"theora", "THEO"},
}

VIDEO_CODEC_CANDIDATES = [
    (".mp4", "avc1"),
    (".mp4", "H264"),
    (".mp4", "X264"),
    (".webm", "VP90"),
    (".webm", "VP80"),
    (".mp4", "mp4v"),
]


def probe_video_codec_tag(video_path: Path) -> str:
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            return ""
        raw_fourcc = int(capture.get(cv2.CAP_PROP_FOURCC) or 0)
        return "".join(chr((raw_fourcc >> (8 * index)) & 0xFF) for index in range(4)).strip()
    finally:
        capture.release()



def is_browser_playable_clip(video_path: Path) -> bool:
    suffix = video_path.suffix.lower()
    playable_tags = BROWSER_PLAYABLE_VIDEO_TAGS.get(suffix)
    if not playable_tags or not video_path.exists() or video_path.stat().st_size == 0:
        return False
    codec_tag = probe_video_codec_tag(video_path)
    return codec_tag in playable_tags



def render_video_candidate(output_path: Path, frame_paths: list[Path], codec: str) -> Path | None:
    render_frames = prepare_video_render_frames(frame_paths)
    if not render_frames:
        return None

    height, width = render_frames[0].shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*codec), VIDEO_FPS, (width, height))
    if not writer.isOpened():
        writer.release()
        return None

    wrote_frames = 0
    try:
        for frame in render_frames:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
            wrote_frames += 1
    finally:
        writer.release()

    if wrote_frames == 0 or not output_path.exists() or output_path.stat().st_size == 0:
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        return None
    return output_path



def write_video_clip(anchor_record: dict, frame_paths: list[Path]) -> Path | None:
    if not frame_paths:
        return None

    cache_key = build_video_clip_cache_key(anchor_record, frame_paths)
    cached_clip = get_cached_video_clip(cache_key)
    if cached_clip is not None:
        return cached_clip

    GENERATED_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    clip_stem = anchor_record.get("scene_name") or anchor_record.get("scene_token") or Path(frame_paths[0]).stem
    camera = anchor_record.get("camera") or PRIMARY_CAMERA
    base_name = sanitize_filename(f'{clip_stem}_{camera}_{Path(frame_paths[0]).stem}')

    fallback_candidate: Path | None = None
    for suffix, codec in VIDEO_CODEC_CANDIDATES:
        candidate_path = GENERATED_VIDEO_DIR / f"{base_name}_{codec}{suffix}"
        if candidate_path.exists() and is_browser_playable_clip(candidate_path):
            remember_video_clip(cache_key, candidate_path)
            return candidate_path

        clip_path = render_video_candidate(candidate_path, frame_paths, codec)
        if clip_path is None:
            continue
        if is_browser_playable_clip(clip_path):
            remember_video_clip(cache_key, clip_path)
            return clip_path
        if ffmpeg_installed() and clip_path.suffix.lower() != ".mp4":
            converted_path = Path(convert_video_to_playable_mp4(str(clip_path)))
            if converted_path.exists() and is_browser_playable_clip(converted_path):
                remember_video_clip(cache_key, converted_path)
                return converted_path
        fallback_candidate = clip_path

    if fallback_candidate and fallback_candidate.exists():
        fallback_candidate.unlink(missing_ok=True)
    return None




def build_video_caption(
    record: dict,
    frame_paths: list[Path],
    clip_path: Path | None,
    parsed_query: dict | None = None,
    kg_status: str = "",
    model_name: str = "",
) -> str:
    caption_parts = [("锚点分数", f"{record['score']:.4f}")]
    reason_summary = " / ".join(build_result_reason_tags(record, parsed_query, kg_status, model_name))
    if reason_summary:
        caption_parts.append(("命中条件", reason_summary))
    if record.get("scene_name"):
        caption_parts.append(("场景", str(record["scene_name"])))
    elif record.get("scene_token"):
        caption_parts.append(("场景", str(record["scene_token"])))
    clip_summary = f"{len(frame_paths)} 帧"
    if record.get("camera"):
        clip_summary = f"{record['camera']} · {clip_summary}"
    caption_parts.append(("片段", clip_summary))
    return build_result_details_html(caption_parts[:4])


def retrieve_videos(text: str) -> tuple[list[tuple[str, str]], str, dict, str]:
    timings: dict[str, float] = {}

    stage_started_at = time.perf_counter()
    parsed_query = parse_query(text)
    timings["nlp"] = time.perf_counter() - stage_started_at

    stage_started_at = time.perf_counter()
    candidate_scene_tokens, kg_status = get_candidate_scene_tokens(parsed_query)
    timings["kg"] = time.perf_counter() - stage_started_at

    stage_started_at = time.perf_counter()
    query_vector, model_name = encode_text_query(text)
    timings["clip"] = time.perf_counter() - stage_started_at

    stage_started_at = time.perf_counter()
    hits = search_frame_hits(query_vector, VIDEO_SEARCH_LIMIT, candidate_scene_tokens)
    timings["milvus"] = time.perf_counter() - stage_started_at

    stage_started_at = time.perf_counter()
    anchors_by_sequence: dict[tuple[str, str], dict] = {}
    for hit in hits:
        sequence_key = derive_sequence_group_key(hit)
        if sequence_key not in anchors_by_sequence:
            anchors_by_sequence[sequence_key] = hit
    timings["group"] = time.perf_counter() - stage_started_at

    frame_collection_time = 0.0
    clip_generation_time = 0.0
    video_results: list[tuple[str, str]] = []
    skipped_unplayable = 0
    for anchor_record in anchors_by_sequence.values():
        stage_started_at = time.perf_counter()
        frame_paths = collect_video_frames(anchor_record)
        frame_collection_time += time.perf_counter() - stage_started_at

        stage_started_at = time.perf_counter()
        clip_path = write_video_clip(anchor_record, frame_paths)
        clip_generation_time += time.perf_counter() - stage_started_at

        if clip_path is None:
            skipped_unplayable += 1
            continue
        video_results.append(
            (
                str(clip_path),
                build_video_caption(anchor_record, frame_paths, clip_path, parsed_query, kg_status, model_name),
            )
        )
        if len(video_results) >= VIDEO_RESULT_COUNT:
            break

    timings["frames"] = frame_collection_time
    timings["clips"] = clip_generation_time

    if not video_results and anchors_by_sequence:
        kg_status = f"{kg_status} | 已命中候选帧，但未能生成浏览器可播放的视频片段。"
    elif skipped_unplayable:
        kg_status = f"{kg_status} | 因浏览器不可播放，已跳过 {skipped_unplayable} 个片段候选。"

    return video_results, model_name, parsed_query, append_timing_breakdown(kg_status, timings)




MODE_LABELS = {
    "text2image": "搜索图片",
    "text2video": "搜索视频片段",
}


def status_card(title: str, body: str, note: str = "", tone: str = "neutral") -> str:
    note_html = f"<div class='status-note'>{escape(note)}</div>" if note else ""
    return (
        f"<details class='status-card {tone}'>"
        "<summary>"
        f"<span class='status-title'>{escape(title)}</span>"
        "<span class='status-toggle'>检索详情</span>"
        "</summary>"
        "<div class='status-content'>"
        f"<div class='status-body'>{escape(body)}</div>"
        f"{note_html}</div></details>"
    )


def summarize_query_clauses(parsed_query: dict | None) -> list[str]:
    parsed_query = parsed_query or {}
    clauses: list[str] = []
    if parsed_query.get("weather"):
        clauses.append(f"天气 {parsed_query['weather']}")
    if parsed_query.get("time"):
        clauses.append(f"时段 {parsed_query['time']}")
    if parsed_query.get("location"):
        clauses.append(f"位置 {parsed_query['location']}")
    objects = parsed_query.get("objects") or []
    if objects:
        clauses.append("对象 " + "/".join(objects[:2]))
    return clauses or ["仅语义检索"]


def summarize_candidate_source(kg_status: str) -> str:
    status = kg_status or ""
    if "尚未执行" in status or "提交后确定候选来源" in status:
        return "待执行"
    if "本地知识图谱" in status or "本地 KG" in status or "local KG filtered" in status:
        return "本地知识图谱过滤"
    if "Neo4j 已过滤出" in status or "Neo4j filtered" in status:
        return "Neo4j 过滤"
    if "回退到全库搜索" in status or "falling back to full-collection search" in status:
        return "全库回退"
    if "Neo4j 不可用" in status or "Neo4j unavailable" in status:
        return "Neo4j 降级"
    if "未提取到知识图谱过滤条件" in status or "未提取到 KG 过滤条件" in status or "No KG filters extracted" in status:
        return "未使用知识图谱条件"
    return trim_text(status, 36) if status else "待检索"


def build_structured_condition_rows(parsed_query: dict | None) -> list[tuple[str, str]]:
    parsed_query = parsed_query or {}
    rows: list[tuple[str, str]] = []
    if parsed_query.get("weather"):
        rows.append(("天气", str(parsed_query["weather"])))
    if parsed_query.get("time"):
        rows.append(("时段", str(parsed_query["time"])))
    if parsed_query.get("location"):
        rows.append(("位置", str(parsed_query["location"])))
    objects = parsed_query.get("objects") or []
    if objects:
        rows.append(("对象", " / ".join(str(item) for item in objects[:3])))
    return rows


def build_kg_filter_rows(parsed_query: dict | None, kg_status: str) -> list[tuple[str, str]]:
    parsed_query = parsed_query or {}
    rows: list[tuple[str, str]] = [("执行路径", summarize_candidate_source(kg_status))]

    if parsed_query.get("weather"):
        rows.append(
            (
                "天气条件",
                f"实体 Scene / Weather · 关系 WEATHER · 属性 Scene.weather, Weather.name · 值 {parsed_query['weather']}",
            )
        )
    if parsed_query.get("time"):
        rows.append(
            (
                "时段条件",
                f"实体 Scene / TimeOfDay · 关系 TIMEOFDAY · 属性 Scene.timeofday, TimeOfDay.name · 值 {parsed_query['time']}",
            )
        )
    if parsed_query.get("location"):
        rows.append(
            (
                "位置条件",
                f"实体 Scene / Location · 关系 LOCTYPE · 属性 Scene.location_kind, Location.kind · 值 {parsed_query['location']}",
            )
        )
    objects = parsed_query.get("objects") or []
    if objects:
        rows.append(
            (
                "对象条件",
                f"实体 Scene / Object · 关系 CONTAINS · 属性 Object.name · 值 {' / '.join(str(item) for item in objects[:3])}",
            )
        )

    if kg_status:
        rows.append(("系统返回", trim_text(kg_status, 200)))
    return rows


def build_result_reason_tags(
    record: dict,
    parsed_query: dict | None = None,
    kg_status: str = "",
    model_name: str = "",
) -> list[str]:
    parsed_query = parsed_query or {}
    tags: list[str] = []

    if parsed_query.get("weather") and record.get("weather") == parsed_query["weather"]:
        tags.append(f"天气 {parsed_query['weather']}")
    if parsed_query.get("time") and record.get("timeofday") == parsed_query["time"]:
        tags.append(f"时段 {parsed_query['time']}")

    record_objects = {item.strip() for item in str(record.get("obj_types") or "").split(",") if item.strip()}
    matched_objects = [obj for obj in parsed_query.get("objects") or [] if obj in record_objects]
    if matched_objects:
        tags.append("对象 " + "/".join(matched_objects[:2]))

    record_location = str(record.get("location") or "")
    if parsed_query.get("location") and parsed_query["location"] in record_location:
        tags.append(f"位置 {parsed_query['location']}")

    candidate_source = summarize_candidate_source(kg_status)
    if candidate_source:
        tags.append(candidate_source)

    unique_tags: list[str] = []
    for tag in tags:
        if tag and tag not in unique_tags:
            unique_tags.append(tag)
    return unique_tags[:4] or ["视觉语义命中"]


def render_key_value_rows(rows: list[tuple[str, str]], empty_text: str) -> str:
    valid_rows = [(key, value) for key, value in rows if value]
    if not valid_rows:
        return f"<div class='detail-empty'>{escape(empty_text)}</div>"
    return (
        "<div class='detail-list'>"
        + "".join(
            f"<div class='detail-row'><div class='detail-key'>{escape(key)}</div><div class='detail-value'>{escape(value)}</div></div>"
            for key, value in valid_rows
        )
        + "</div>"
    )


def build_result_details_html(rows: list[tuple[str, str]]) -> str:
    return (
        "<details class='result-details'>"
        "<summary>查看详情</summary>"
        f"<div class='result-detail-list'>{render_key_value_rows(rows, '暂无结果详情')}</div>"
        "</details>"
    )


def build_explanation_html(
    parsed_query: dict | None,
    kg_status: str,
    model_name: str,
    mode: str,
    result_count: int,
) -> str:
    nlp_rows = build_structured_condition_rows(parsed_query)
    kg_rows = build_kg_filter_rows(parsed_query, kg_status)
    return (
        "<details class='detail-panel explain-shell'>"
        "<summary>"
        "<span class='detail-summary-title'>解析与图谱详情</span>"
        "<span class='detail-summary-meta'>查看详情</span>"
        "</summary>"
        "<div class='detail-stack'>"
        "<div class='module-card'>"
        "<div class='module-title'>自然语言解析模块</div>"
        f"{render_key_value_rows(nlp_rows, '尚未抽取到结构化条件。')}"
        "</div>"
        "<div class='module-card'>"
        "<div class='module-title'>知识图谱过滤模块</div>"
        f"{render_key_value_rows(kg_rows, '提交检索后展示图谱过滤信息。')}"
        "</div>"
        "</div></details>"
    )


def build_preview_explanation(text: str, mode: str) -> str:
    query = text.strip()
    if not query:
        return build_explanation_html({}, "尚未执行知识图谱过滤。", "待命", mode, 0)
    parsed_query = parse_query(query)
    return build_explanation_html(parsed_query, "尚未执行知识图谱过滤，提交后展示 Neo4j / 本地知识图谱 / 全库回退。", "待检索", mode, 0)


INITIAL_STATUS_HTML = status_card(
    "系统已就绪",
    "双语 CLIP、知识图谱过滤与 Milvus 帧检索已准备完成。",
    INITIAL_STATUS,
)

INITIAL_EXPLANATION_HTML = build_explanation_html({}, "尚未执行知识图谱过滤。", "待命", "text2image", 0)

STATS_HTML = f"""
<div class='stats-grid'>
  <div class='stat-card'><div class='stat-label'>帧</div><div class='stat-value'>{len(ID_TO_RAW_PATH):,}</div></div>
  <div class='stat-card'><div class='stat-label'>场景</div><div class='stat-value'>{len(SCENE_BY_TOKEN)}</div></div>
  <div class='stat-card'><div class='stat-label'>模型</div><div class='stat-value'>2</div></div>
  <div class='stat-card'><div class='stat-label'>视频</div><div class='stat-value'>按需生成</div></div>
</div>
"""


custom_css = """
:root {
    --surface: rgba(255,255,255,0.86);
    --surface-strong: rgba(255,255,255,0.94);
    --surface-soft: rgba(247,250,255,0.82);
    --line: rgba(32,33,36,0.08);
    --text: #202124;
    --muted: #5f6368;
    --blue: #1a73e8;
    --blue-deep: #174ea6;
    --blue-soft: #e8f0fe;
    --shadow: 0 22px 60px rgba(60,64,67,0.10);
    --shadow-soft: 0 14px 36px rgba(60,64,67,0.08);
    --radius-xl: 36px;
    --radius-lg: 24px;
}
body, .gradio-container, .gradio-container-5-44-1 {
    background:
        radial-gradient(circle at 12% 10%, rgba(66,133,244,0.12) 0%, rgba(66,133,244,0.00) 34%),
        radial-gradient(circle at 90% 14%, rgba(234,67,53,0.07) 0%, rgba(234,67,53,0.00) 24%),
        radial-gradient(circle at 84% 76%, rgba(251,188,5,0.08) 0%, rgba(251,188,5,0.00) 22%),
        radial-gradient(circle at 16% 84%, rgba(52,168,83,0.08) 0%, rgba(52,168,83,0.00) 24%),
        linear-gradient(180deg, #f9fbff 0%, #f3f7fd 50%, #f7f9fc 100%);
    color: var(--text);
    font-family: "MiSans", "HarmonyOS Sans SC", "PingFang SC", "Microsoft YaHei UI", "Segoe UI Variable", sans-serif;
}
body.lightbox-open,
body.lightbox-open .gradio-container,
body.lightbox-open .gradio-container-5-44-1 {
    overflow: hidden !important;
}
body.lightbox-open {
    overscroll-behavior: none !important;
}
.gradio-container, .gradio-container-5-44-1 {
    max-width: 100% !important;
    padding: 24px 28px 36px !important;
}
.app-shell {
    width: 100% !important;
    max-width: 1480px !important;
    margin: 0 auto !important;
    gap: 14px;
}
.topbar, .hero-card, .search-card, .results-shell, #status-panel, #query-detail-panel {
    background: var(--surface);
    border: 1px solid var(--line);
    box-shadow: var(--shadow);
    backdrop-filter: blur(22px);
}
.topbar {
    display:flex;
    justify-content:space-between;
    align-items:center;
    gap:18px;
    padding: 18px 24px;
    border-radius: 999px;
    color: var(--muted);
}
.brand-lockup { display:flex; align-items:center; gap:14px; min-width: 0; }
.brand-mark { display:grid; grid-template-columns: repeat(2, 10px); gap:6px; }
.brand-mark span { width:10px; height:10px; border-radius:999px; display:block; }
.brand-blue { background:#4285f4; }
.brand-red { background:#ea4335; }
.brand-yellow { background:#fbbc04; }
.brand-green { background:#34a853; }
.topbar-title { font-size: 1.08rem; font-weight: 800; color: var(--text); letter-spacing: -0.04em; }
.topbar-subtitle { margin-top: 2px; color: var(--muted); font-size: .88rem; }
.topbar-note {
    padding: 8px 14px;
    border-radius: 999px;
    background: rgba(255,255,255,0.72);
    border: 1px solid rgba(26,115,232,0.10);
    color: var(--blue-deep);
    font-size: .86rem;
    font-weight: 600;
}
.hero-card { position: relative; overflow: hidden; border-radius: var(--radius-xl); padding: 28px 28px 24px; background: linear-gradient(180deg, rgba(255,255,255,0.96) 0%, rgba(246,250,255,0.92) 100%); }
.hero-card::before { content:""; position:absolute; inset: 0; background: linear-gradient(135deg, rgba(255,255,255,0.30), rgba(255,255,255,0.04)); pointer-events:none; }
.hero-grid { position:relative; z-index:1; display:grid; grid-template-columns: 1fr; gap: 18px; align-items: stretch; }
.hero-copy { display:flex; flex-direction:column; justify-content:center; }
.hero-kicker, .section-kicker, .results-kicker {
    display:inline-flex;
    width: fit-content;
    padding:7px 12px;
    border-radius:999px;
    background: var(--blue-soft);
    color: var(--blue);
    font-size: .82rem;
    font-weight:700;
    letter-spacing: .02em;
}
.hero-title {
    margin: 14px 0 0;
    font-size: clamp(2.2rem, 3.9vw, 3.5rem);
    line-height: 1.04;
    font-weight: 900;
    letter-spacing: -0.07em;
}
.hero-title span { display:block; color: var(--blue-deep); }
.hero-subtitle, .section-subtitle, .results-subtitle, .footer-note, .search-note { color: var(--muted); line-height: 1.8; }
.hero-subtitle { max-width: 620px; font-size: .98rem; margin-top: 12px; }
.capability-row { display:flex; flex-wrap:wrap; gap:10px; margin-top: 18px; }
.capability-chip { padding: 8px 12px; border-radius:999px; background: rgba(255,255,255,0.82); border:1px solid rgba(32,33,36,0.08); font-size:.86rem; font-weight:600; color: #4c5664; box-shadow: 0 8px 20px rgba(66,133,244,0.06); }
.hero-visual { display: none; }
.stats-grid { display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:10px; margin-top: 18px; }
.stat-card { padding: 14px 16px; border-radius: 18px; background: var(--surface-strong); border: 1px solid var(--line); box-shadow: var(--shadow-soft); }
.stat-label { color: #6b7280; font-size:.78rem; letter-spacing:.08em; margin-bottom: 4px; }
.stat-value { color: var(--blue-deep); font-size: 1.18rem; font-weight: 800; letter-spacing: -0.04em; }
.stat-note { display:none; }
.search-card, .results-shell { border-radius: var(--radius-xl); padding: 20px 22px; }
.section-heading, .results-heading { margin: 10px 0 0; font-size: 1.18rem; font-weight: 800; letter-spacing: -0.04em; }
#query-box {
    border-radius: 32px;
    border:1px solid var(--line);
    background:#fff;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.8), 0 10px 28px rgba(26,115,232,0.08);
    transition: border-color .2s ease, box-shadow .2s ease;
}
#query-box:focus-within { border-color: rgba(26,115,232,0.36); box-shadow: inset 0 1px 0 rgba(255,255,255,0.8), 0 12px 34px rgba(26,115,232,0.12); }
#query-box textarea { padding: 16px 20px !important; min-height: 84px !important; font-size: 1rem !important; line-height:1.7 !important; color: var(--text) !important; }
#query-box textarea::placeholder { color: #9097a3 !important; }
.control-row { align-items: center; gap: 12px; margin-top: 12px; }
.mode-tabs { border-radius: 24px; border:1px solid var(--line); background:#fff; padding: 10px; box-shadow: var(--shadow-soft); }
.mode-tabs label { border-radius:999px !important; border:1px solid rgba(32,33,36,0.08) !important; background:#fff !important; color: var(--muted) !important; font-weight: 700 !important; min-height: 44px !important; }
.mode-tabs label[data-selected="true"] { background: var(--blue-soft) !important; color: var(--blue) !important; border-color: rgba(26,115,232,0.10) !important; }
#search-btn, #clear-btn { border-radius:999px !important; font-weight: 700 !important; min-height: 46px !important; }
#search-btn {
    min-width:144px;
    background: linear-gradient(135deg, var(--blue) 0%, #4f96ff 100%) !important;
    color:#fff !important;
    box-shadow: 0 12px 26px rgba(26,115,232,0.22);
}
#clear-btn { min-width:104px; background:#fff !important; color: var(--text) !important; border:1px solid var(--line) !important; }
#status-panel, #query-detail-panel { border-radius: 26px; padding: 4px; }
.status-card, .detail-panel {
    border-radius: 22px;
    padding: 18px 20px;
    background: linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(248,250,255,0.96) 100%);
    border: 1px solid rgba(32,33,36,0.06);
}
.status-card.success { border-color: rgba(52,168,83,0.18); background: linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(243,250,245,0.98) 100%); }
.status-card.warning { border-color: rgba(251,188,5,0.22); background: linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(255,249,236,0.96) 100%); }
.status-card summary,
.detail-panel summary,
.result-details summary {
    list-style: none;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    cursor: pointer;
}
.status-card summary::-webkit-details-marker,
.detail-panel summary::-webkit-details-marker,
.result-details summary::-webkit-details-marker {
    display: none;
}
.status-title,
.detail-summary-title {
    font-size: 1rem;
    font-weight: 800;
    color: var(--text);
}
.status-toggle,
.detail-summary-meta {
    color: var(--blue-deep);
    font-size: .84rem;
    font-weight: 700;
}
.status-content,
.detail-stack,
.result-detail-list {
    margin-top: 14px;
    padding-top: 14px;
    border-top: 1px solid rgba(32,33,36,0.06);
}
.status-body { color: var(--muted); font-size:.98rem; line-height:1.7; }
.status-note { margin-top: 10px; color: var(--muted); font-size:.92rem; line-height:1.7; }
.detail-stack {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
}
.module-card {
    padding: 14px;
    border-radius: 18px;
    background: rgba(255,255,255,0.92);
    border: 1px solid rgba(32,33,36,0.06);
    box-shadow: var(--shadow-soft);
}
.module-title {
    margin-bottom: 10px;
    font-size: .94rem;
    font-weight: 800;
    color: var(--text);
}
.detail-list { display: grid; gap: 10px; }
.detail-row,
.result-detail-row {
    display: grid;
    grid-template-columns: 84px minmax(0, 1fr);
    gap: 12px;
    align-items: start;
}
.detail-key,
.result-detail-key {
    color: #6b7280;
    font-size: .82rem;
    font-weight: 700;
    letter-spacing: .04em;
}
.detail-value,
.result-detail-value {
    color: var(--text);
    font-size: .92rem;
    line-height: 1.65;
    word-break: break-word;
}
.detail-empty {
    color: var(--muted);
    font-size: .92rem;
    line-height: 1.7;
}
#image-grid {
    display: grid !important;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 12px !important;
    width: 100%;
}
#video-grid {
    display: grid !important;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px !important;
    width: 100%;
}
#image-grid > div,
#video-grid > div {
    min-width: 0 !important;
}
.image-card,
.video-card {
    padding: 12px;
    border-radius: var(--radius-lg);
    background: rgba(255,255,255,0.96);
    border:1px solid var(--line);
    box-shadow: 0 14px 30px rgba(31,41,55,0.08);
    transition: transform .22s ease, box-shadow .22s ease;
    min-width: 0 !important;
}
.image-card:hover,
.video-card:hover { transform: translateY(-4px); box-shadow: 0 22px 56px rgba(26,115,232,0.12); }
.media-frame { border-radius: 16px; min-width: 0; }
.media-frame img, .media-frame video {
    border-radius: 16px !important;
    object-fit: cover !important;
    width: 100% !important;
    height: 100% !important;
}
.image-card-actions {
    display: flex;
    justify-content: flex-end;
    margin-top: 10px;
}
.preview-trigger {
    min-height: 38px !important;
    border-radius: 999px !important;
    border: 1px solid rgba(26,115,232,0.14) !important;
    background: rgba(232,240,254,0.72) !important;
    color: var(--blue-deep) !important;
    font-weight: 700 !important;
}
.preview-trigger:hover {
    transform: none !important;
    background: rgba(232,240,254,0.92) !important;
}
.media-frame button {
    z-index: 3 !important;
    transition: none !important;
    transform: none !important;
}
.media-frame button:hover {
    transform: none !important;
}
.result-zone-title { margin-bottom: 10px; font-size: .98rem; font-weight: 800; color: var(--text); }
.result-meta { margin-top: 10px; }
.result-details {
    border: 1px solid rgba(32,33,36,0.06);
    border-radius: 16px;
    background: rgba(255,255,255,0.9);
    padding: 2px 12px 12px;
}
.result-details summary {
    padding-top: 8px;
    color: var(--blue-deep);
    font-size: .9rem;
    font-weight: 700;
}
.result-detail-list .detail-list {
    gap: 8px;
}
#image-lightbox {
    position: fixed !important;
    inset: 0 !important;
    z-index: 1200 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 24px !important;
    background: rgba(248, 250, 252, 0.74) !important;
    backdrop-filter: blur(14px) saturate(1.05) !important;
    overflow: hidden !important;
}
.image-lightbox-card {
    width: min(960px, calc(100vw - 48px));
    max-height: calc(100dvh - 48px);
    margin: 0 auto;
    padding: 20px 20px 18px;
    border-radius: 32px;
    background: rgba(255,255,255,0.98);
    border: 1px solid rgba(255,255,255,0.5);
    box-shadow: 0 32px 96px rgba(15,23,42,0.34);
    display: flex;
    flex-direction: column;
    gap: 10px;
    overflow: hidden;
}
.lightbox-toolbar {
    display: flex;
    align-items: center;
    justify-content: flex-start;
    gap: 12px;
    margin-bottom: 2px;
}
.lightbox-heading {
    color: var(--text);
    font-size: 1rem;
    font-weight: 800;
    letter-spacing: -0.03em;
}
.lightbox-close {
    position: fixed !important;
    top: 24px !important;
    right: 28px !important;
    z-index: 1210 !important;
    min-height: 42px !important;
    border-radius: 999px !important;
    min-width: 88px !important;
    width: auto !important;
    flex: 0 0 auto !important;
    background: rgba(255,255,255,0.94) !important;
    color: var(--text) !important;
    border: 1px solid rgba(32,33,36,0.08) !important;
    box-shadow: 0 12px 30px rgba(15,23,42,0.12) !important;
    font-weight: 700 !important;
}
.lightbox-media {
    border-radius: 24px;
    overflow: hidden;
    background: linear-gradient(180deg, #f8fbff 0%, #eef3fb 100%);
    border: 1px solid rgba(32,33,36,0.06);
    padding: 12px;
    flex: 1 1 auto;
    min-height: 0;
    display: flex;
    align-items: center;
    justify-content: center;
}
.lightbox-media img {
    border-radius: 20px !important;
    object-fit: contain !important;
    width: 100% !important;
    height: auto !important;
    max-height: calc(100dvh - 320px) !important;
}
.lightbox-meta {
    margin-top: 0;
    padding: 0 4px 4px;
    max-height: none;
    overflow: visible;
}
.footer-note { text-align:center; font-size:.88rem; padding-top: 2px; }
#search-btn, #clear-btn {
    transition: transform .18s ease, box-shadow .18s ease;
}
#search-btn:hover, #clear-btn:hover {
    transform: translateY(-1px);
}
@media (max-width: 1380px) {
    #image-grid { grid-template-columns: repeat(4, minmax(0, 1fr)); }
}
@media (max-width: 1120px) {
    #image-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    #video-grid,
    .detail-stack { grid-template-columns: 1fr 1fr; }
}
@media (max-width: 780px) {
    .hero-card, .search-card, .results-shell { padding: 18px; }
    .hero-title { font-size: 2.2rem; }
    .topbar { flex-direction:column; align-items:flex-start; border-radius:30px; }
    .topbar-note { width: 100%; text-align: center; }
    .control-row { flex-direction: column; align-items: stretch; }
    #image-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    #video-grid,
    .stats-grid,
    .detail-stack { grid-template-columns: 1fr; }
    .detail-row,
    .result-detail-row { grid-template-columns: 1fr; gap: 4px; }
}
@media (max-width: 640px) {
    .gradio-container, .gradio-container-5-44-1 { padding: 14px 12px 24px !important; }
    #image-grid { grid-template-columns: 1fr; }
    .stats-grid { grid-template-columns: 1fr 1fr; }
    #image-lightbox { padding: 12px !important; }
    .image-lightbox-card { width: 100%; }
    .lightbox-close { top: 16px !important; right: 16px !important; }
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Base(), fill_width=True) as iface:
    with gr.Column(elem_id="app-shell", elem_classes="app-shell"):
        gr.HTML(
            """
            <div class='topbar'>
              <div class='brand-lockup'>
                <div class='brand-mark'>
                  <span class='brand-blue'></span>
                  <span class='brand-red'></span>
                  <span class='brand-yellow'></span>
                  <span class='brand-green'></span>
                </div>
                <div>
                  <div class='topbar-title'>知识图谱场景检索</div>
                  <div class='topbar-subtitle'>更简洁的自动驾驶场景检索工作台</div>
                </div>
              </div>
              <div class='topbar-note'>搜索图片 / 搜索视频片段</div>
            </div>
            """
        )
        with gr.Column(elem_classes="hero-card"):
            gr.HTML(
                """
                <div class='hero-grid'>
                    <div class='hero-copy'>
                        <div class='hero-kicker'>知识图谱场景检索</div>
                        <div class='hero-title'>更快定位驾驶场景</div>
                        <div class='hero-subtitle'>
                            支持中文、英文、知识图谱过滤与结果命中解释。Milvus 主库存储的是图片帧实体，视频片段按需派生生成。
                        </div>
                        <div class='capability-row'>
                            <span class='capability-chip'>中英双语自动切换</span>
                            <span class='capability-chip'>知识图谱 + CLIP 融合检索</span>
                            <span class='capability-chip'>命中解释可见</span>
                        </div>
                    </div>
                    <div class='hero-visual'>
                        <div class='visual-orb orb-blue'></div>
                        <div class='visual-orb orb-red'></div>
                        <div class='visual-orb orb-yellow'></div>
                        <div class='visual-orb orb-green'></div>
                        <div class='visual-stack'>
                            <div class='visual-panel-primary'>
                                <div class='visual-label'>检索链路</div>
                                <div class='visual-value'>NLP 解析 -> 知识图谱过滤 -> CLIP 向量检索</div>
                            </div>
                            <div class='visual-panel-group'>
                                <div class='visual-panel'>
                                    <div class='visual-label'>查询模型</div>
                                    <div class='visual-value'>Chinese-CLIP 与 English-CLIP 自动路由</div>
                                </div>
                                <div class='visual-panel'>
                                    <div class='visual-label'>视频输出</div>
                                    <div class='visual-value'>先检索关键帧，再导出浏览器可播放的 mp4 片段</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                """
            )
            gr.HTML(STATS_HTML)
        with gr.Column(elem_classes="search-card"):
            gr.HTML(
                """
                <div class='section-kicker'>开始检索</div>
                <div class='section-heading'>输入一句场景描述</div>
                """
            )
            text_input = gr.Textbox(
                show_label=False,
                placeholder="例如：夜间路口有行人；或 urban road with cars in daytime",
                lines=2,
                container=False,
                elem_id="query-box",
            )
            with gr.Row(elem_classes="control-row"):
                with gr.Column(scale=5, min_width=280):
                    mode_select = gr.Radio(
                        choices=[("搜索图片", "text2image"), ("搜索视频片段", "text2video")],
                        value="text2image",
                        show_label=False,
                        container=False,
                        elem_classes="mode-tabs",
                    )
                with gr.Column(scale=1, min_width=140):
                    submit_btn = gr.Button("开始检索", variant="primary", elem_id="search-btn")
                with gr.Column(scale=1, min_width=110):
                    clear_btn = gr.Button("清空", variant="secondary", elem_id="clear-btn")
            query_detail_panel = gr.HTML(value=INITIAL_EXPLANATION_HTML, elem_id="query-detail-panel")
        status_panel = gr.HTML(value=INITIAL_STATUS_HTML, elem_id="status-panel")
        with gr.Column(elem_classes="results-shell"):
            gr.HTML(
                """
                <div class='results-heading'>检索结果</div>
                """
            )
            with gr.Column(visible=True, elem_id="image_result_zone") as image_result_zone:
                gr.HTML("<div class='result-zone-title'>图片结果</div>")
                image_outputs = []
                image_caption_outputs = []
                image_preview_buttons = []
                with gr.Row(elem_id="image-grid"):
                    for index in range(IMAGE_RESULT_COUNT):
                        with gr.Column(scale=1, min_width=0, elem_classes="image-card"):
                            image_component = gr.Image(
                                show_label=False,
                                type="pil",
                                height=250,
                                container=False,
                                interactive=False,
                                show_download_button=False,
                                show_fullscreen_button=False,
                                elem_classes="media-frame",
                            )
                            with gr.Row(elem_classes="image-card-actions"):
                                preview_button = gr.Button("放大查看", elem_classes="preview-trigger")
                            image_caption = gr.HTML(elem_classes="result-meta")
                            image_outputs.append(image_component)
                            image_caption_outputs.append(image_caption)
                            image_preview_buttons.append(preview_button)
                with gr.Column(visible=False, elem_id="image-lightbox") as image_lightbox:
                    with gr.Column(elem_classes="image-lightbox-card"):
                        with gr.Row(elem_classes="lightbox-toolbar"):
                            gr.HTML("<div class='lightbox-heading'>图片预览</div>")
                            image_lightbox_close = gr.Button("关闭", elem_classes="lightbox-close")
                        image_lightbox_image = gr.Image(
                            show_label=False,
                            type="pil",
                            height=560,
                            container=False,
                            interactive=False,
                            show_download_button=False,
                            show_fullscreen_button=False,
                            elem_classes="lightbox-media",
                        )
                        image_lightbox_caption = gr.HTML(elem_classes="result-meta lightbox-meta")
                image_preview_outputs = [image_lightbox_image, image_lightbox_caption, image_lightbox]
            with gr.Column(visible=False, elem_id="video_result_zone") as video_result_zone:
                gr.HTML("<div class='result-zone-title'>视频结果</div>")
                video_outputs = []
                video_caption_outputs = []
                with gr.Row(elem_id="video-grid"):
                    for index in range(VIDEO_RESULT_COUNT):
                        with gr.Column(scale=1, min_width=0, elem_classes="video-card"):
                            video_component = gr.Video(
                                show_label=False,
                                height=260,
                                container=False,
                                show_download_button=False,
                                elem_classes="media-frame",
                            )
                            video_caption = gr.HTML(elem_classes="result-meta")
                            video_outputs.append(video_component)
                            video_caption_outputs.append(video_caption)
        gr.HTML(
            """
            <div class='footer-note'>
                Milvus 仅存图片帧实体；视频片段由命中帧序列即时生成。
            </div>
            """
        )

    def update_progress(text: str, mode: str):
        query = text.strip()
        if not query:
            return [INITIAL_STATUS_HTML, build_preview_explanation("", mode)]
        return [
            status_card(
                "正在准备检索",
                "正在解析结构化条件并准备检索链路。",
                f"当前查询：{trim_text(query, 120)}",
            ),
            build_preview_explanation(query, mode),
        ]

    def switch_result_zone(mode: str, text: str):
        return [
            gr.update(visible=(mode == "text2image")),
            gr.update(visible=(mode == "text2video")),
            build_preview_explanation(text, mode),
        ]

    def build_output_values(
        image_results: list[tuple[Image.Image, str]] | None = None,
        video_results: list[tuple[str, str]] | None = None,
        status: str = INITIAL_STATUS_HTML,
        explanation: str = INITIAL_EXPLANATION_HTML,
    ) -> list:
        image_results = image_results or []
        video_results = video_results or []
        values: list = []
        for index in range(IMAGE_RESULT_COUNT):
            values.append(image_results[index][0] if index < len(image_results) else None)
        for index in range(IMAGE_RESULT_COUNT):
            values.append(image_results[index][1] if index < len(image_results) else "")
        for index in range(VIDEO_RESULT_COUNT):
            values.append(video_results[index][0] if index < len(video_results) else None)
        for index in range(VIDEO_RESULT_COUNT):
            values.append(video_results[index][1] if index < len(video_results) else "")
        values.append(None)
        values.append("")
        values.append(gr.update(visible=False))
        values.append(status)
        values.append(explanation)
        return values

    def open_image_preview(image: Image.Image | None, caption: str):
        if image is None:
            return [None, "", gr.update(visible=False)]
        return [image, caption or "", gr.update(visible=True)]

    def close_image_preview():
        return [None, "", gr.update(visible=False)]

    OPEN_IMAGE_PREVIEW_JS = """
    (image, caption) => {
        document.body.classList.add('lightbox-open');
        document.documentElement.classList.add('lightbox-open');
        return [image, caption];
    }
    """

    CLOSE_IMAGE_PREVIEW_JS = """
    () => {
        document.body.classList.remove('lightbox-open');
        document.documentElement.classList.remove('lightbox-open');
        return [];
    }
    """

    def dynamic_retrieve(text: str, mode: str):
        query = text.strip()
        if not query:
            return build_output_values(
                status=status_card(
                    "请先输入查询内容",
                    "请先输入一段中文或英文场景描述，再开始检索。",
                    "支持天气、时段、对象与位置等结构化条件。",
                    tone="warning",
                ),
                explanation=build_preview_explanation("", mode),
            )
        started_at = time.time()
        try:
            if mode == "text2image":
                image_results, model_name, parsed_query, kg_status = retrieve_images(query)
                elapsed = time.time() - started_at
                parsed_summary = format_parsed_query(parsed_query) or "仅语义匹配"
                status = status_card(
                    "图片检索完成",
                    f"{model_name} 返回 {len(image_results)} 张图片结果，用时 {elapsed:.2f}s。",
                    f"{parsed_summary} | {kg_status}",
                    tone="success" if image_results else "warning",
                )
                explanation = build_explanation_html(parsed_query, kg_status, model_name, mode, len(image_results))
                return build_output_values(image_results=image_results, status=status, explanation=explanation)

            video_results, model_name, parsed_query, kg_status = retrieve_videos(query)
            elapsed = time.time() - started_at
            parsed_summary = format_parsed_query(parsed_query) or "仅语义匹配"
            note = f"{parsed_summary} | {kg_status} | 视频片段由命中帧序列派生生成。"
            status = status_card(
                "视频检索完成",
                f"{model_name} 返回 {len(video_results)} 个视频片段，用时 {elapsed:.2f}s。",
                note,
                tone="success" if video_results else "warning",
            )
            explanation = build_explanation_html(parsed_query, kg_status, model_name, mode, len(video_results))
            return build_output_values(video_results=video_results, status=status, explanation=explanation)
        except Exception as exc:
            return build_output_values(
                status=status_card(
                    "检索失败",
                    "执行检索时发生异常，请检查模型、Milvus、Neo4j 与本地资源。",
                    str(exc),
                    tone="warning",
                ),
                explanation=build_preview_explanation(query, mode),
            )

    def clear_all():
        return [""] + build_output_values(status=INITIAL_STATUS_HTML, explanation=INITIAL_EXPLANATION_HTML)

    mode_select.change(
        fn=switch_result_zone,
        inputs=[mode_select, text_input],
        outputs=[image_result_zone, video_result_zone, query_detail_panel],
    )
    text_input.change(fn=update_progress, inputs=[text_input, mode_select], outputs=[status_panel, query_detail_panel])
    submit_btn.click(
        fn=dynamic_retrieve,
        inputs=[text_input, mode_select],
        outputs=image_outputs
        + image_caption_outputs
        + video_outputs
        + video_caption_outputs
        + image_preview_outputs
        + [status_panel, query_detail_panel],
    )
    text_input.submit(
        fn=dynamic_retrieve,
        inputs=[text_input, mode_select],
        outputs=image_outputs
        + image_caption_outputs
        + video_outputs
        + video_caption_outputs
        + image_preview_outputs
        + [status_panel, query_detail_panel],
    )
    clear_btn.click(
        fn=clear_all,
        outputs=[text_input]
        + image_outputs
        + image_caption_outputs
        + video_outputs
        + video_caption_outputs
        + image_preview_outputs
        + [status_panel, query_detail_panel],
        js=CLOSE_IMAGE_PREVIEW_JS,
    )
    for preview_button, image_component, image_caption in zip(image_preview_buttons, image_outputs, image_caption_outputs):
        preview_button.click(
            fn=open_image_preview,
            inputs=[image_component, image_caption],
            outputs=image_preview_outputs,
            js=OPEN_IMAGE_PREVIEW_JS,
        )
    image_lightbox_close.click(fn=close_image_preview, outputs=image_preview_outputs, js=CLOSE_IMAGE_PREVIEW_JS)


def env_flag(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name, "")
    if not raw_value:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


def write_launch_info(local_url: str, share_url: str, share_enabled: bool, status: str | None = None, detail: str | None = None) -> None:
    info_path_raw = os.getenv("KGSR_LAUNCH_INFO_FILE", "").strip()
    if not info_path_raw:
        return

    if status is None:
        if share_enabled and share_url:
            status = "AVAILABLE"
            detail = detail or "Public share link is ready."
        elif share_enabled:
            status = "UNAVAILABLE"
            detail = detail or "Gradio launched locally but did not return a public share URL."
        else:
            status = "DISABLED"
            detail = detail or "Public share link disabled by user."

    info_lines = [
        f'set "GRADIO_LOCAL_URL={local_url}"',
        f'set "GRADIO_SHARE_URL={share_url}"',
        f'set "GRADIO_SHARE_STATUS={status}"',
        f'set "GRADIO_SHARE_STATUS_DETAIL={detail}"',
    ]
    Path(info_path_raw).write_text("\n".join(info_lines), encoding="utf-8")


def launch_interface() -> None:
    share_enabled = env_flag("KGSR_ENABLE_SHARE")
    server_name = (os.getenv("GRADIO_SERVER_NAME", "") or "").strip() or None
    server_port_raw = (os.getenv("GRADIO_SERVER_PORT", "") or "").strip()
    server_port = int(server_port_raw) if server_port_raw.isdigit() else None

    try:
        _app, local_url, share_url = iface.launch(
            share=share_enabled,
            show_error=True,
            server_name=server_name,
            server_port=server_port,
            prevent_thread_lock=True,
        )
        write_launch_info(local_url, share_url or "", share_enabled)
    except Exception as exc:
        write_launch_info("", "", share_enabled, status="FAILED", detail=f"Gradio launch failed: {exc}")
        raise

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    launch_interface()

