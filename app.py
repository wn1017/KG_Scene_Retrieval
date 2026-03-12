from __future__ import annotations

import json
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
    ENGCLIP_MODEL_DIR,
    GENERATED_VIDEO_DIR,
    IMAGE_CSV_PATH,
    IMAGE_ID_MIN,
    NUSCENES_META_DIR,
    NUSCENES_ROOT,
    NUSCENES_SAMPLES_DIR,
    NUSCENES_SWEEPS_DIR,
    PRIMARY_CAMERA,
    VIDEO_FPS,
    VIDEO_FRAME_STRIDE,
    VIDEO_MAX_FRAMES,
    VIDEO_RESULT_COUNT,
    VIDEO_SEARCH_LIMIT,
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


def get_live_collection(force_refresh: bool = False):
    global COLLECTION, MILVUS_ERROR
    if force_refresh or COLLECTION is None:
        COLLECTION, MILVUS_ERROR = load_collection_state()
    return COLLECTION


STARTUP_MESSAGES = []
if MODEL_LOAD_ERROR:
    STARTUP_MESSAGES.append(f"Model load failed: {MODEL_LOAD_ERROR}")
else:
    STARTUP_MESSAGES.append(f"模型已加载到 {DEVICE.type}")
STARTUP_MESSAGES.append(f"图像索引: {len(ID_TO_RAW_PATH)} 条")
STARTUP_MESSAGES.append(f"nuScenes 元数据: {len(SCENE_BY_TOKEN)} 个场景, {len(FILENAME_TO_SAMPLE_DATA)} 条图像记录")
STARTUP_MESSAGES.append(f"知识图谱场景记录: {len(KG_SCENE_RECORDS)}")
if COLLECTION is None:
    STARTUP_MESSAGES.append(f"Milvus 不可用: {MILVUS_ERROR or '连接失败'}")
else:
    STARTUP_MESSAGES.append("Milvus collection 已连接")
    if schema_needs_rebuild(COLLECTION):
        STARTUP_MESSAGES.append("当前 collection 仍是旧 schema，元数据过滤能力会受限，需重建后完全生效")
INITIAL_STATUS = " | ".join(STARTUP_MESSAGES)


def format_parsed_query(parsed_query: dict) -> str:
    parts = []
    if parsed_query.get("weather"):
        parts.append(f"天气={parsed_query['weather']}")
    if parsed_query.get("time"):
        parts.append(f"时段={parsed_query['time']}")
    if parsed_query.get("location"):
        parts.append(f"地点={parsed_query['location']}")
    if parsed_query.get("objects"):
        parts.append("目标物=" + ",".join(parsed_query["objects"]))
    if not parts:
        return "未提取到结构化条件"
    return "；".join(parts)


def get_candidate_scene_tokens(parsed_query: dict) -> tuple[list[str], str]:
    weather = parsed_query.get("weather")
    timeofday = parsed_query.get("time")
    object_types = parsed_query.get("objects") or []
    location_kind = parsed_query.get("location")

    if not any([weather, timeofday, object_types, location_kind]):
        return [], "未提取到知识图谱过滤条件，转为全库检索。"

    try:
        neo4j_tokens = query_scene_tokens(
            weather=weather,
            timeofday=timeofday,
            object_types=object_types,
            location_kind=location_kind,
        )
        if neo4j_tokens:
            return neo4j_tokens, f"知识图谱经 Neo4j 筛出 {len(neo4j_tokens)} 个场景。"
        return [], "知识图谱未筛到候选场景，回退到全库检索。"
    except RuntimeError:
        local_tokens = filter_scene_records(
            KG_SCENE_RECORDS,
            weather=weather,
            timeofday=timeofday,
            object_types=object_types,
            location_kind=location_kind,
        )
        if local_tokens:
            return local_tokens, f"Neo4j 不可用，本地图谱筛出 {len(local_tokens)} 个场景。"
        return [], "Neo4j 不可用，且本地图谱未命中，转为全库检索。"


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
    raw_path = record.get("frame_path") or ID_TO_RAW_PATH.get(record["id"], "")
    resolved_path = resolve_frame_path(raw_path)
    sample_data = get_sample_data_for_frame(resolved_path, raw_path)

    record["raw_frame_path"] = raw_path
    record["resolved_frame_path"] = str(resolved_path) if resolved_path else ""
    record.setdefault("camera", infer_camera_from_path(raw_path) or infer_camera_from_path(resolved_path))

    if sample_data:
        record.setdefault("sample_token", sample_data.get("sample_token", ""))
        record.setdefault("scene_token", sample_data.get("scene_token", ""))
        record.setdefault("camera", sample_data.get("camera", ""))
        record["source_filename"] = sample_data.get("filename", "")
        record["sample_data_token"] = sample_data.get("sample_data_token", "")

    scene_token = record.get("scene_token", "")
    scene_record = SCENE_BY_TOKEN.get(scene_token, {}) if scene_token else {}
    kg_scene_record = KG_RECORD_BY_SCENE_TOKEN.get(scene_token, {}) if scene_token else {}

    record["scene_name"] = scene_record.get("name") or kg_scene_record.get("scene_name", "")
    record["scene_description"] = scene_record.get("description") or kg_scene_record.get("description", "")
    if kg_scene_record:
        record.setdefault("weather", kg_scene_record.get("weather", ""))
        record.setdefault("timeofday", kg_scene_record.get("timeofday", ""))
        if not record.get("location"):
            record["location"] = f"{kg_scene_record.get('location_area', '')}:{kg_scene_record.get('location_kind', '')}".strip(':')
        if not record.get("obj_types"):
            record["obj_types"] = ", ".join(kg_scene_record.get("objects", {}).keys())
    return record

def search_frame_hits(query_vector: np.ndarray, limit: int, candidate_scene_tokens: list[str] | None = None) -> list[dict]:
    global COLLECTION, MILVUS_ERROR

    candidate_scene_tokens = list(dict.fromkeys(candidate_scene_tokens or []))

    def _run_search(active_collection) -> list[dict]:
        active_collection.load()
        if active_collection.num_entities == 0:
            return []

        use_scene_filter = bool(candidate_scene_tokens) and has_field(active_collection, "scene_token")
        output_fields = get_search_output_fields(active_collection)
        expr = f"id >= {IMAGE_ID_MIN}"
        search_limit = limit
        if use_scene_filter:
            expr = f"{expr} and {build_scene_filter_expr(candidate_scene_tokens)}"
        elif candidate_scene_tokens:
            search_limit = max(limit * 8, 80)

        search_result = active_collection.search(
            data=[query_vector.tolist()],
            anns_field="embedding",
            param=SEARCH_PARAMS,
            limit=search_limit,
            expr=expr,
            output_fields=output_fields,
        )

        enriched_hits = []
        allowed_scene_tokens = set(candidate_scene_tokens)
        for hit in search_result[0]:
            enriched_record = enrich_hit_record(extract_hit_record(hit, output_fields))
            if candidate_scene_tokens and not use_scene_filter and enriched_record.get("scene_token") not in allowed_scene_tokens:
                continue
            enriched_hits.append(enriched_record)
            if len(enriched_hits) >= limit:
                break
        return enriched_hits

    active_collection = get_live_collection()
    if active_collection is None:
        raise RuntimeError(MILVUS_ERROR or "Milvus is unavailable")

    try:
        return _run_search(active_collection)
    except Exception as first_exc:
        COLLECTION = None
        MILVUS_ERROR = str(first_exc)
        refreshed_collection = get_live_collection(force_refresh=True)
        if refreshed_collection is None:
            raise RuntimeError(MILVUS_ERROR or "Milvus is unavailable") from first_exc
        try:
            return _run_search(refreshed_collection)
        except Exception as second_exc:
            COLLECTION = None
            MILVUS_ERROR = str(second_exc)
            raise RuntimeError(f"Milvus collection unavailable: {second_exc}") from second_exc

def build_image_caption(record: dict) -> str:
    caption_parts = [f"相似度: {record['score']:.4f}"]
    if record.get("camera"):
        caption_parts.append(f"相机: {record['camera']}")
    if record.get("scene_name"):
        caption_parts.append(f"场景: {record['scene_name']}")
    elif record.get("scene_token"):
        caption_parts.append(f"场景标识: {record['scene_token']}")
    if record.get("weather"):
        caption_parts.append(f"天气: {record['weather']}")
    if record.get("timeofday"):
        caption_parts.append(f"时段: {record['timeofday']}")
    if record.get("obj_types"):
        caption_parts.append(f"目标物: {trim_text(record['obj_types'], 70)}")
    frame_name = Path(record.get("resolved_frame_path") or record.get("raw_frame_path") or "").name
    if frame_name:
        caption_parts.append(f"帧文件: {frame_name}")
    return "<br>".join(caption_parts)


def retrieve_images(text: str) -> tuple[list[tuple[Image.Image, str]], str, dict, str]:
    parsed_query = parse_query(text)
    candidate_scene_tokens, kg_status = get_candidate_scene_tokens(parsed_query)
    query_vector, model_name = encode_text_query(text)
    hits = search_frame_hits(query_vector, IMAGE_RESULT_COUNT, candidate_scene_tokens)

    image_results: list[tuple[Image.Image, str]] = []
    for hit in hits:
        resolved_path = hit.get("resolved_frame_path", "")
        if not resolved_path:
            continue
        image = Image.open(resolved_path).convert("RGB")
        image_results.append((image, build_image_caption(hit)))
        if len(image_results) >= IMAGE_RESULT_COUNT:
            break

    return image_results, model_name, parsed_query, kg_status

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
        return [resolved_path] * max(8, VIDEO_FPS * 2) if resolved_path and resolved_path.exists() else []

    scene_token = anchor_record.get("scene_token") or sample_data.get("scene_token", "")
    camera = anchor_record.get("camera") or sample_data.get("camera", "") or PRIMARY_CAMERA
    sequence = CAMERA_SEQUENCES.get((scene_token, camera), [])
    if not sequence:
        return [resolved_path] * max(8, VIDEO_FPS * 2) if resolved_path and resolved_path.exists() else []

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
    for item in sequence[start_index:end_index:VIDEO_FRAME_STRIDE]:
        frame_path = NUSCENES_ROOT / item["filename"]
        if frame_path.exists():
            frame_paths.append(frame_path)

    if len(frame_paths) == 1:
        frame_paths = frame_paths * max(8, VIDEO_FPS * 2)
    return frame_paths


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
    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        return None

    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*codec), VIDEO_FPS, (width, height))
    if not writer.isOpened():
        writer.release()
        return None

    wrote_frames = 0
    try:
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
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

    GENERATED_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    clip_stem = anchor_record.get("scene_name") or anchor_record.get("scene_token") or Path(frame_paths[0]).stem
    camera = anchor_record.get("camera") or PRIMARY_CAMERA
    base_name = sanitize_filename(f'{clip_stem}_{camera}_{Path(frame_paths[0]).stem}')

    fallback_candidate: Path | None = None
    for suffix, codec in VIDEO_CODEC_CANDIDATES:
        candidate_path = GENERATED_VIDEO_DIR / f"{base_name}_{codec}{suffix}"
        clip_path = render_video_candidate(candidate_path, frame_paths, codec)
        if clip_path is None:
            continue
        if is_browser_playable_clip(clip_path):
            return clip_path
        if ffmpeg_installed() and clip_path.suffix.lower() != ".mp4":
            converted_path = Path(convert_video_to_playable_mp4(str(clip_path)))
            if converted_path.exists() and is_browser_playable_clip(converted_path):
                return converted_path
        fallback_candidate = clip_path

    if fallback_candidate and fallback_candidate.exists():
        fallback_candidate.unlink(missing_ok=True)
    return None


def build_video_caption(record: dict, frame_paths: list[Path], clip_path: Path | None) -> str:
    caption_parts = [f"最佳帧相似度: {record['score']:.4f}"]
    if record.get("scene_name"):
        caption_parts.append(f"场景: {record['scene_name']}")
    elif record.get("scene_token"):
        caption_parts.append(f"场景标识: {record['scene_token']}")
    if record.get("camera"):
        caption_parts.append(f"相机: {record['camera']}")
    if record.get("scene_description"):
        caption_parts.append(f"场景描述: {trim_text(record['scene_description'], 110)}")
    caption_parts.append(f"片段帧数: {len(frame_paths)}")
    if clip_path:
        caption_parts.append(f"生成文件: {clip_path.name}")
    return "<br>".join(caption_parts)


def retrieve_videos(text: str) -> tuple[list[tuple[str, str]], str, dict, str]:
    parsed_query = parse_query(text)
    candidate_scene_tokens, kg_status = get_candidate_scene_tokens(parsed_query)
    query_vector, model_name = encode_text_query(text)
    hits = search_frame_hits(query_vector, VIDEO_SEARCH_LIMIT, candidate_scene_tokens)

    anchors_by_sequence: dict[tuple[str, str], dict] = {}
    for hit in hits:
        sequence_key = derive_sequence_group_key(hit)
        if sequence_key not in anchors_by_sequence:
            anchors_by_sequence[sequence_key] = hit

    video_results: list[tuple[str, str]] = []
    skipped_unplayable = 0
    for anchor_record in anchors_by_sequence.values():
        frame_paths = collect_video_frames(anchor_record)
        clip_path = write_video_clip(anchor_record, frame_paths)
        if clip_path is None:
            skipped_unplayable += 1
            continue
        video_results.append((str(clip_path), build_video_caption(anchor_record, frame_paths, clip_path)))
        if len(video_results) >= VIDEO_RESULT_COUNT:
            break

    if not video_results and anchors_by_sequence:
        kg_status = f"{kg_status} | 已命中候选帧，但当前环境未生成浏览器兼容的视频片段。"
    elif skipped_unplayable:
        kg_status = f"{kg_status} | {skipped_unplayable} 个候选片段因浏览器兼容性被跳过。"

    return video_results, model_name, parsed_query, kg_status


MODE_LABELS = {
    "text2image": "文搜图",
    "text2video": "文搜视频片段",
}


def status_card(title: str, body: str, note: str = "", tone: str = "neutral") -> str:
    note_html = f"<div class='status-note'>{escape(note)}</div>" if note else ""
    return (
        f"<div class='status-card {tone}'>"
        f"<div class='status-title'>{escape(title)}</div>"
        f"<div class='status-body'>{escape(body)}</div>"
        f"{note_html}</div>"
    )


INITIAL_STATUS_HTML = status_card(
    "系统已就绪",
    "中英双语 CLIP 检索、Neo4j 图谱过滤与 Milvus 帧级搜索均已在线。",
    f"{INITIAL_STATUS} | 视频模式会根据命中帧序列即时拼接片段。",
)

STATS_HTML = f"""
<div class='stats-grid'>
  <div class='stat-card'><div class='stat-label'>已索引帧</div><div class='stat-value'>{len(ID_TO_RAW_PATH):,}</div><div class='stat-note'>Milvus 中存储的关键帧实体。</div></div>
  <div class='stat-card'><div class='stat-label'>场景数</div><div class='stat-value'>{len(SCENE_BY_TOKEN)}</div><div class='stat-note'>已对齐官方 nuScenes 场景与图谱结构。</div></div>
  <div class='stat-card'><div class='stat-label'>查询模型</div><div class='stat-value'>2</div><div class='stat-note'>中文 Chinese-CLIP 与 English-CLIP 自动切换。</div></div>
  <div class='stat-card'><div class='stat-label'>视频模式</div><div class='stat-value'>动态生成</div><div class='stat-note'>根据命中帧序列即时合成短片段。</div></div>
</div>
"""

custom_css = """
:root {
    --panel: rgba(255,255,255,0.88);
    --panel-strong: rgba(255,255,255,0.94);
    --line: rgba(32,33,36,0.08);
    --text: #202124;
    --muted: #5f6368;
    --blue: #1a73e8;
    --blue-soft: #e8f0fe;
    --shadow: 0 22px 54px rgba(32,33,36,0.10);
    --radius-xl: 34px;
    --radius-lg: 24px;
}
body, .gradio-container, .gradio-container-5-44-1 {
    background:
        radial-gradient(circle at 12% 14%, rgba(66,133,244,0.16) 0%, rgba(66,133,244,0.00) 32%),
        radial-gradient(circle at 88% 18%, rgba(234,67,53,0.10) 0%, rgba(234,67,53,0.00) 26%),
        radial-gradient(circle at 82% 78%, rgba(251,188,5,0.11) 0%, rgba(251,188,5,0.00) 28%),
        radial-gradient(circle at 18% 80%, rgba(52,168,83,0.10) 0%, rgba(52,168,83,0.00) 26%),
        linear-gradient(180deg, #fbfdff 0%, #f2f6ff 54%, #f7f9fc 100%);
    color: var(--text);
    font-family: "Microsoft YaHei UI", "PingFang SC", "Segoe UI Variable", sans-serif;
}
.gradio-container, .gradio-container-5-44-1 { max-width: 1680px !important; padding: 24px 24px 44px !important; }
.app-shell { gap: 18px; }
.topbar, .hero-card, .search-card, .results-shell, #status-panel { background: var(--panel); border: 1px solid var(--line); box-shadow: var(--shadow); backdrop-filter: blur(18px); }
.topbar { display:flex; justify-content:space-between; align-items:center; gap:16px; padding: 16px 22px; border-radius: 999px; color: var(--muted); }
.topbar-title { font-size: 1.24rem; font-weight: 800; color: var(--text); letter-spacing: -0.03em; }
.hero-card { position: relative; overflow: hidden; border-radius: var(--radius-xl); padding: 40px 42px 36px; }
.hero-card::before { content:""; position:absolute; inset: 0; background: linear-gradient(135deg, rgba(255,255,255,0.24), rgba(255,255,255,0.02)); pointer-events:none; }
.hero-grid { position:relative; z-index:1; display:grid; grid-template-columns: minmax(0, 1.18fr) minmax(320px, 0.82fr); gap: 26px; align-items: stretch; }
.hero-copy { display:flex; flex-direction:column; justify-content:center; }
.hero-kicker, .section-kicker, .results-kicker { display:inline-flex; width: fit-content; padding:8px 14px; border-radius:999px; background: var(--blue-soft); color: var(--blue); font-size: .86rem; font-weight:700; }
.hero-title { margin: 18px 0 14px; font-size: clamp(3.05rem, 5.3vw, 5rem); line-height: .95; font-weight: 900; letter-spacing: -0.06em; }
.hero-title span { display:block; color: var(--blue); }
.hero-subtitle, .section-subtitle, .results-subtitle, .footer-note, .search-note { color: var(--muted); line-height: 1.8; }
.capability-row { display:flex; flex-wrap:wrap; gap:10px; margin-top: 20px; }
.capability-chip { padding: 10px 15px; border-radius:999px; background: rgba(255,255,255,0.84); border:1px solid var(--line); font-size:.92rem; font-weight:600; color: var(--muted); }
.hero-visual { position:relative; min-height: 340px; border-radius: 30px; overflow: hidden; background: linear-gradient(160deg, rgba(255,255,255,0.80) 0%, rgba(232,240,254,0.72) 32%, rgba(255,255,255,0.70) 100%); border:1px solid rgba(26,115,232,0.10); }
.hero-visual::before { content:""; position:absolute; inset:-12% auto auto -8%; width: 220px; height:220px; background: radial-gradient(circle, rgba(66,133,244,0.26), rgba(66,133,244,0)); }
.hero-visual::after { content:""; position:absolute; inset:auto -10% -14% auto; width: 260px; height:260px; background: radial-gradient(circle, rgba(251,188,5,0.22), rgba(251,188,5,0)); }
.visual-orb { position:absolute; border-radius: 999px; filter: blur(2px); opacity: .82; }
.orb-blue { width: 140px; height: 140px; top: 34px; right: 44px; background: radial-gradient(circle, rgba(66,133,244,0.28), rgba(66,133,244,0.05)); }
.orb-red { width: 88px; height: 88px; top: 140px; right: 170px; background: radial-gradient(circle, rgba(234,67,53,0.22), rgba(234,67,53,0.04)); }
.orb-yellow { width: 96px; height: 96px; bottom: 42px; left: 58px; background: radial-gradient(circle, rgba(251,188,5,0.24), rgba(251,188,5,0.04)); }
.orb-green { width: 78px; height: 78px; bottom: 94px; right: 92px; background: radial-gradient(circle, rgba(52,168,83,0.20), rgba(52,168,83,0.04)); }
.visual-stack { position:relative; z-index:2; display:flex; flex-direction:column; justify-content:flex-end; gap:14px; height:100%; padding: 26px; }
.visual-panel, .visual-panel-primary { background: rgba(255,255,255,0.86); border:1px solid rgba(32,33,36,0.08); border-radius: 22px; box-shadow: 0 18px 44px rgba(66,133,244,0.10); }
.visual-panel-primary { padding: 18px 18px 16px; }
.visual-panel-group { display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:12px; }
.visual-panel { padding: 16px; }
.visual-label { color:#6b7280; font-size:.82rem; letter-spacing:.08em; text-transform:uppercase; }
.visual-value { margin-top: 8px; font-size:1rem; font-weight:700; line-height:1.55; color: var(--text); }
.stats-grid { display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:14px; margin-top: 28px; }
.stat-card { padding: 18px; border-radius: 22px; background: var(--panel-strong); border: 1px solid var(--line); }
.stat-label { color: #6b7280; font-size:.82rem; letter-spacing:.08em; text-transform:uppercase; margin-bottom: 8px; }
.stat-value { color: var(--blue); font-size: 2rem; font-weight: 800; letter-spacing: -0.04em; }
.stat-note { color: var(--muted); line-height: 1.7; }
.search-card, .results-shell { border-radius: var(--radius-xl); padding: 28px 30px; }
.section-heading, .results-heading { margin: 14px 0 10px; font-size: 1.24rem; font-weight: 800; letter-spacing: -0.03em; }
.search-row { align-items: stretch; gap: 14px; }
#query-box { border-radius: 30px; border:1px solid var(--line); background:#fff; box-shadow: inset 0 1px 0 rgba(255,255,255,0.8), 0 10px 24px rgba(26,115,232,0.07); }
#query-box textarea { padding: 18px 22px !important; min-height: 88px !important; font-size: 1.04rem !important; line-height:1.72 !important; color: var(--text) !important; }
.mode-tabs { border-radius: 26px; border:1px solid var(--line); background:#fff; padding: 14px; }
.mode-tabs label { border-radius:999px !important; border:1px solid var(--line) !important; background:#fff !important; color: var(--muted) !important; font-weight: 700 !important; }
.mode-tabs label[data-selected="true"] { background: var(--blue-soft) !important; color: var(--blue) !important; }
.action-row { gap: 12px; margin-top: 10px; }
#search-btn, #clear-btn { border-radius:999px !important; }
#search-btn { min-width:150px; background: linear-gradient(135deg, var(--blue) 0%, #4f96ff 100%) !important; color:#fff !important; }
#clear-btn { min-width:120px; background:#fff !important; color: var(--text) !important; border:1px solid var(--line) !important; }
#status-panel { border-radius: 26px; padding: 4px; }
.status-card { border-radius: 22px; padding: 20px 22px; background: linear-gradient(180deg, rgba(255,255,255,0.96) 0%, rgba(248,250,255,0.96) 100%); }
.status-title { font-size:1.04rem; font-weight:800; color: var(--text); }
.status-body { margin-top: 10px; color: var(--muted); font-size:.98rem; line-height:1.7; }
.status-note { margin-top: 12px; color: var(--muted); font-size:.92rem; line-height:1.7; }
.image-grid, .video-grid { gap: 14px; }
.image-card, .video-card { padding: 14px; border-radius: var(--radius-lg); background: rgba(255,255,255,0.95); border:1px solid var(--line); box-shadow: 0 14px 34px rgba(31,41,55,0.08); transition: transform .2s ease, box-shadow .2s ease; }
.image-card:hover, .video-card:hover { transform: translateY(-4px); box-shadow: 0 20px 52px rgba(26,115,232,0.12); }
.media-frame { border-radius: 16px; overflow:hidden; }
.media-frame img, .media-frame video { border-radius: 16px !important; object-fit: cover !important; }
.result-zone-title { margin-bottom: 12px; font-size: 1.02rem; font-weight: 800; color: var(--text); }
.result-meta { margin-top: 12px; font-size: .94rem; }
.footer-note { text-align:center; font-size:.92rem; padding-top: 4px; }
@media (max-width: 1180px) { .hero-grid { grid-template-columns: 1fr; } .stats-grid { grid-template-columns: repeat(2, minmax(0,1fr)); } .hero-visual { min-height: 280px; } }
@media (max-width: 780px) { .hero-card, .search-card, .results-shell { padding: 22px 18px; } .hero-title { font-size: 2.8rem; } .topbar { flex-direction:column; align-items:flex-start; border-radius:28px; } .visual-panel-group { grid-template-columns: 1fr; } }
@media (max-width: 640px) { .gradio-container, .gradio-container-5-44-1 { padding: 14px 12px 26px !important; } .stats-grid { grid-template-columns: 1fr; } }
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Base(), fill_width=True) as iface:
    with gr.Column(elem_classes="app-shell"):
        gr.HTML("<div class='topbar'><div class='topbar-title'>驾驶场景检索工作台</div><div>知识图谱 + 跨模态驾驶场景检索系统</div></div>")
        with gr.Column(elem_classes="hero-card"):
            gr.HTML("""
                <div class='hero-grid'>
                    <div class='hero-copy'>
                        <div class='hero-kicker'>知识图谱 × CLIP × Milvus</div>
                        <div class='hero-title'>用自然语言检索<span>驾驶场景图像与视频片段</span></div>
                        <div class='hero-subtitle'>融合 Neo4j 候选过滤、Milvus 向量检索与本地中英双语 CLIP 模型，支持中文与英文查询。当前系统提供文搜图，以及基于命中帧序列即时拼接的视频片段检索。</div>
                        <div class='capability-row'>
                            <span class='capability-chip'>中文 / English 自动切换</span>
                            <span class='capability-chip'>知识图谱候选过滤与回退</span>
                            <span class='capability-chip'>对齐 nuScenes 官方元数据</span>
                            <span class='capability-chip'>文搜图与文搜视频片段</span>
                        </div>
                    </div>
                    <div class='hero-visual'>
                        <div class='visual-orb orb-blue'></div>
                        <div class='visual-orb orb-red'></div>
                        <div class='visual-orb orb-yellow'></div>
                        <div class='visual-orb orb-green'></div>
                        <div class='visual-stack'>
                            <div class='visual-panel-primary'>
                                <div class='visual-label'>检索路径</div>
                                <div class='visual-value'>NLP 解析 → KG 过滤 → CLIP 检索</div>
                            </div>
                            <div class='visual-panel-group'>
                                <div class='visual-panel'>
                                    <div class='visual-label'>查询语言</div>
                                    <div class='visual-value'>中文 / English</div>
                                </div>
                                <div class='visual-panel'>
                                    <div class='visual-label'>视频结果</div>
                                    <div class='visual-value'>由命中帧动态拼接</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            """)
            gr.HTML(STATS_HTML)
        with gr.Column(elem_classes="search-card"):
            gr.HTML("<div class='section-kicker'>检索工作台</div><div class='section-heading'>自然语言查询 + 图谱感知排序</div><div class='section-subtitle'>输入中文或英文场景描述，系统会自动选择编码器，并在提取到结构化条件时优先进行知识图谱过滤。</div>")
            with gr.Row(elem_classes="search-row"):
                with gr.Column(scale=6):
                    text_input = gr.Textbox(show_label=False, placeholder="例如：雨天夜晚十字路口有行人，或 night driving near a bus stop", lines=2, container=False, elem_id="query-box")
                with gr.Column(scale=3, min_width=240):
                    mode_select = gr.Radio(choices=[("文搜图", "text2image"), ("文搜视频片段", "text2video")], value="text2image", show_label=False, container=False, elem_classes="mode-tabs")
            with gr.Row(elem_classes="action-row"):
                submit_btn = gr.Button("开始检索", variant="primary", elem_id="search-btn")
                clear_btn = gr.Button("清空", variant="secondary", elem_id="clear-btn")
            gr.HTML("<div class='search-note'>视频模式返回的是根据命中帧序列即时拼接的短片段。Milvus 中存储的是帧级实体，不是独立视频向量。</div>")
        status_panel = gr.HTML(value=INITIAL_STATUS_HTML, elem_id="status-panel")
        with gr.Column(elem_classes="results-shell"):
            gr.HTML("<div class='results-kicker'>检索结果</div><div class='results-heading'>带结构化场景信息的高置信度命中</div><div class='results-subtitle'>图像结果展示关键帧与元数据；视频结果展示围绕高分命中帧自动拼接出的短片段。</div>")
            with gr.Column(visible=True, elem_id="image_result_zone") as image_result_zone:
                gr.HTML("<div class='result-zone-title'>图像结果</div>")
                image_outputs = []
                image_caption_outputs = []
                with gr.Row(elem_classes="image-grid"):
                    for index in range(IMAGE_RESULT_COUNT):
                        with gr.Column(min_width=220, elem_classes="image-card"):
                            image_component = gr.Image(show_label=False, type="pil", height=270, container=False, elem_classes="media-frame")
                            image_caption = gr.Markdown(elem_classes="result-meta")
                            image_outputs.append(image_component)
                            image_caption_outputs.append(image_caption)
            with gr.Column(visible=False, elem_id="video_result_zone") as video_result_zone:
                gr.HTML("<div class='result-zone-title'>视频片段结果</div>")
                video_outputs = []
                video_caption_outputs = []
                with gr.Row(elem_classes="video-grid"):
                    for index in range(VIDEO_RESULT_COUNT):
                        with gr.Column(min_width=300, elem_classes="video-card"):
                            video_component = gr.Video(show_label=False, height=270, container=False, elem_classes="media-frame")
                            video_caption = gr.Markdown(elem_classes="result-meta")
                            video_outputs.append(video_component)
                            video_caption_outputs.append(video_caption)
        gr.HTML("<div class='footer-note'>本地演示系统。模型从磁盘加载，Milvus 存储帧级实体，视频片段按需由 nuScenes 连续帧即时生成。</div>")

    def update_progress(text: str) -> str:
        query = text.strip()
        if not query:
            return INITIAL_STATUS_HTML
        return status_card("准备检索", "提交后将执行中英双语 CLIP 检索与知识图谱候选过滤。", f"当前草稿：{trim_text(query, 120)}")

    def switch_result_zone(mode: str):
        return [gr.update(visible=(mode == "text2image")), gr.update(visible=(mode == "text2video"))]

    def build_output_values(image_results: list[tuple[Image.Image, str]] | None = None, video_results: list[tuple[str, str]] | None = None, status: str = INITIAL_STATUS_HTML) -> list:
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
        values.append(status)
        return values

    def dynamic_retrieve(text: str, mode: str):
        query = text.strip()
        if not query:
            return build_output_values(status=status_card("请输入查询", "请先输入中文或英文的驾驶场景描述。", "当前支持文搜图和文搜视频片段。", tone="warning"))
        started_at = time.time()
        try:
            if mode == "text2image":
                image_results, model_name, parsed_query, kg_status = retrieve_images(query)
                elapsed = time.time() - started_at
                parsed_summary = format_parsed_query(parsed_query) or "semantic only"
                status = status_card("图像检索完成", f"{model_name} 返回 {len(image_results)} 张关键帧，用时 {elapsed:.2f}s。", f"{parsed_summary} | {kg_status}", tone="success" if image_results else "warning")
                return build_output_values(image_results=image_results, status=status)
            video_results, model_name, parsed_query, kg_status = retrieve_videos(query)
            elapsed = time.time() - started_at
            parsed_summary = format_parsed_query(parsed_query) or "semantic only"
            note = f"{parsed_summary} | {kg_status} | 视频结果由命中帧序列即时拼接。"
            status = status_card("视频片段检索完成", f"{model_name} 返回 {len(video_results)} 个可播放片段，用时 {elapsed:.2f}s。", note, tone="success" if video_results else "warning")
            return build_output_values(video_results=video_results, status=status)
        except Exception as exc:
            return build_output_values(status=status_card("检索失败", "当前本地服务未能完成这次请求。", str(exc), tone="warning"))

    def clear_all():
        return [""] + build_output_values(status=INITIAL_STATUS_HTML)

    mode_select.change(fn=switch_result_zone, inputs=mode_select, outputs=[image_result_zone, video_result_zone])
    text_input.change(fn=update_progress, inputs=text_input, outputs=status_panel)
    submit_btn.click(fn=dynamic_retrieve, inputs=[text_input, mode_select], outputs=image_outputs + image_caption_outputs + video_outputs + video_caption_outputs + [status_panel])
    text_input.submit(fn=dynamic_retrieve, inputs=[text_input, mode_select], outputs=image_outputs + image_caption_outputs + video_outputs + video_caption_outputs + [status_panel])
    clear_btn.click(fn=clear_all, outputs=[text_input] + image_outputs + image_caption_outputs + video_outputs + video_caption_outputs + [status_panel])


if __name__ == "__main__":
    iface.launch()
