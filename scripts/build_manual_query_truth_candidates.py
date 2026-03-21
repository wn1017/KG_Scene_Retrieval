from __future__ import annotations

import argparse
import json
import pathlib
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PRIMARY_CAMERA
from scripts.evaluate import (
    DEFAULT_FRAME_SEARCH_LIMIT,
    DEFAULT_MAP_DEPTH,
    RetrievalRun,
    run_kg_clip_strict,
    run_pure_clip,
    strict_candidate_scene_tokens,
)
from src.kg_builder import build_scene_records
from src.nuscenes_metadata import load_nuscenes_metadata, resolve_frame_path


INPUT_BENCHMARK_PATH = PROJECT_ROOT / "benchmark" / "manual_query_benchmark.csv"
OUTPUT_CANDIDATE_DETAIL_PATH = PROJECT_ROOT / "benchmark" / "manual_query_truth_candidates.csv"
OUTPUT_BENCHMARK_PATH = PROJECT_ROOT / "benchmark" / "manual_query_benchmark_candidates.csv"

PURE_SCENE_DEPTH = 10
KG_SCENE_DEPTH = 10
MAX_TRUTH_CANDIDATES = 15
MAX_FULL_STRICT_POOL = 12
TOP_HIT_LIMIT = 80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build truth-candidate tables for the manual 20-query benchmark.")
    parser.add_argument("--input", default=str(INPUT_BENCHMARK_PATH), help="Input benchmark CSV path.")
    parser.add_argument("--detail-output", default=str(OUTPUT_CANDIDATE_DETAIL_PATH), help="Detail candidate CSV path.")
    parser.add_argument("--benchmark-output", default=str(OUTPUT_BENCHMARK_PATH), help="Benchmark summary CSV path.")
    return parser.parse_args()


def object_summary(record: dict) -> str:
    ranked_objects = sorted(record.get("objects", {}).items(), key=lambda item: (-int(item[1]), str(item[0])))
    return ",".join(f"{name}:{count}" for name, count in ranked_objects[:10])


def load_scene_camera_index() -> tuple[dict[tuple[str, str], str], dict[str, tuple[str, str]]]:
    metadata_index = load_nuscenes_metadata()
    by_scene_camera: dict[tuple[str, str], tuple[int, str]] = {}
    fallback_by_scene: dict[str, tuple[int, str, str]] = {}

    for record in metadata_index["filename_to_sample_data"].values():
        scene_token = str(record.get("scene_token") or "")
        camera = str(record.get("camera") or "")
        filename = str(record.get("filename") or "")
        timestamp = int(record.get("timestamp", 0) or 0)
        if not scene_token or not camera or not filename:
            continue

        key = (scene_token, camera)
        existing = by_scene_camera.get(key)
        if existing is None or timestamp < existing[0]:
            by_scene_camera[key] = (timestamp, filename)

        fallback = fallback_by_scene.get(scene_token)
        if fallback is None or timestamp < fallback[0]:
            fallback_by_scene[scene_token] = (timestamp, camera, filename)

    resolved_by_scene_camera = {key: value for key, (_timestamp, value) in by_scene_camera.items()}
    resolved_fallback = {scene_token: (camera, value) for scene_token, (_timestamp, camera, value) in fallback_by_scene.items()}
    return resolved_by_scene_camera, resolved_fallback


def choose_scene_image_path(
    scene_token: str,
    preferred_camera: str,
    by_scene_camera: dict[tuple[str, str], str],
    fallback_by_scene: dict[str, tuple[str, str]],
) -> tuple[str, str]:
    camera_candidates = [preferred_camera, PRIMARY_CAMERA]
    seen_cameras: set[str] = set()
    for camera in camera_candidates:
        if not camera or camera in seen_cameras:
            continue
        seen_cameras.add(camera)
        relative_path = by_scene_camera.get((scene_token, camera))
        if relative_path:
            resolved = resolve_frame_path(relative_path)
            return str(resolved) if resolved else relative_path, camera

    fallback = fallback_by_scene.get(scene_token)
    if fallback:
        fallback_camera, relative_path = fallback
        resolved = resolve_frame_path(relative_path)
        return str(resolved) if resolved else relative_path, fallback_camera
    return "", ""


def scene_rankings_from_hits(hits: list[dict], scene_depth: int) -> tuple[list[str], dict[str, int], dict[str, float]]:
    ranked_scene_tokens: list[str] = []
    rank_map: dict[str, int] = {}
    score_map: dict[str, float] = {}
    seen: set[str] = set()

    for hit in hits:
        scene_token = str(hit.get("scene_token") or "").strip()
        if not scene_token or scene_token in seen:
            continue
        seen.add(scene_token)
        ranked_scene_tokens.append(scene_token)
        rank_map[scene_token] = len(ranked_scene_tokens)
        score_map[scene_token] = float(hit.get("score", 0.0) or 0.0)
        if len(ranked_scene_tokens) >= scene_depth:
            break

    return ranked_scene_tokens, rank_map, score_map


def scene_matches_structured_query(scene_record: dict, parsed_query: dict) -> bool:
    if parsed_query.get("weather") and scene_record.get("weather") != parsed_query["weather"]:
        return False
    if parsed_query.get("time") and scene_record.get("timeofday") != parsed_query["time"]:
        return False
    if parsed_query.get("location") and scene_record.get("location_kind") != parsed_query["location"]:
        return False
    required_objects = parsed_query.get("objects") or []
    if required_objects and not all(obj in scene_record.get("objects", {}) for obj in required_objects):
        return False
    return True


def empty_retrieval_run(strategy: str, query_text: str, parsed_query: dict, message: str) -> RetrievalRun:
    return RetrievalRun(
        strategy=strategy,
        query_text=query_text,
        model_name="",
        parsed_query=parsed_query or {},
        kg_status=message,
        ranked_scene_tokens=[],
        top_hits=[],
        candidate_scene_count=None,
        strict_zero_candidate=False,
        used_full_collection=False,
        elapsed_seconds=0.0,
    )


def safe_run_retrieval(strategy: str, query_text: str) -> tuple[RetrievalRun, str]:
    try:
        if strategy == "pure_clip":
            return (
                run_pure_clip(
                    query_text,
                    frame_search_limit=DEFAULT_FRAME_SEARCH_LIMIT,
                    map_depth=DEFAULT_MAP_DEPTH,
                    k=TOP_HIT_LIMIT,
                ),
                "ok",
            )
        if strategy == "kg_clip_strict":
            return (
                run_kg_clip_strict(
                    query_text,
                    frame_search_limit=DEFAULT_FRAME_SEARCH_LIMIT,
                    map_depth=DEFAULT_MAP_DEPTH,
                    k=TOP_HIT_LIMIT,
                ),
                "ok",
            )
        raise ValueError(f"Unsupported strategy: {strategy}")
    except Exception as exc:
        return empty_retrieval_run(strategy, query_text, {}, f"{strategy}_unavailable: {exc}"), str(exc)


def build_candidate_reason(
    scene_token: str,
    seed_scene_token: str,
    strict_pool: set[str],
    pure_rank_map: dict[str, int],
    kg_rank_map: dict[str, int],
    structured_match: bool,
) -> str:
    reasons: list[str] = []
    if scene_token == seed_scene_token:
        reasons.append("seed_scene")
    if scene_token in strict_pool:
        reasons.append("strict_kg_pool")
    if scene_token in pure_rank_map:
        reasons.append(f"pure_clip_rank={pure_rank_map[scene_token]}")
    if scene_token in kg_rank_map:
        reasons.append(f"kg_clip_rank={kg_rank_map[scene_token]}")
    if scene_token in pure_rank_map and scene_token in kg_rank_map:
        reasons.append("appears_in_both_rankings")
    if structured_match:
        reasons.append("matches_structured_conditions")
    return "; ".join(reasons)


def collect_candidate_scene_tokens(
    seed_scene_token: str,
    strict_pool_tokens: list[str],
    pure_ranked_scenes: list[str],
    kg_ranked_scenes: list[str],
) -> list[str]:
    ordered: list[str] = []

    def add(tokens: list[str]) -> None:
        for token in tokens:
            token = str(token or "").strip()
            if token and token not in ordered:
                ordered.append(token)

    add([seed_scene_token])

    both_ranked = [token for token in kg_ranked_scenes if token in set(pure_ranked_scenes)]
    add(both_ranked)
    add(kg_ranked_scenes[:KG_SCENE_DEPTH])
    add(pure_ranked_scenes[:PURE_SCENE_DEPTH])

    if strict_pool_tokens and len(strict_pool_tokens) <= MAX_FULL_STRICT_POOL:
        add(strict_pool_tokens)

    if len(ordered) > MAX_TRUTH_CANDIDATES:
        ordered = ordered[:MAX_TRUTH_CANDIDATES]
    return ordered


def build_outputs(benchmark_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scene_records = build_scene_records()
    scene_record_by_token = {record["scene_token"]: record for record in scene_records}
    by_scene_camera, fallback_by_scene = load_scene_camera_index()

    detail_rows: list[dict] = []
    benchmark_rows: list[dict] = []

    for _, row in benchmark_df.iterrows():
        query_id = str(row["query_id"])
        query_text = str(row["query_text"])
        seed_scene_token = str(row.get("scene_token") or "")
        preferred_camera = str(row.get("camera") or PRIMARY_CAMERA)

        strict_pool_tokens_raw, strict_status, strict_zero = strict_candidate_scene_tokens(
            {
                "weather": row.get("parsed_weather") if pd.notna(row.get("parsed_weather")) else None,
                "time": row.get("parsed_time") if pd.notna(row.get("parsed_time")) else None,
                "location": row.get("parsed_location") if pd.notna(row.get("parsed_location")) else None,
                "objects": json.loads(row.get("parsed_objects") or "[]"),
            }
        )
        if strict_pool_tokens_raw is None:
            strict_pool_tokens = []
        else:
            strict_pool_tokens = list(strict_pool_tokens_raw)
        strict_pool_set = set(strict_pool_tokens)

        pure_run, pure_error = safe_run_retrieval("pure_clip", query_text)
        kg_run, kg_error = safe_run_retrieval("kg_clip_strict", query_text)

        parsed_query = kg_run.parsed_query or pure_run.parsed_query or {
            "weather": row.get("parsed_weather") if pd.notna(row.get("parsed_weather")) else None,
            "time": row.get("parsed_time") if pd.notna(row.get("parsed_time")) else None,
            "location": row.get("parsed_location") if pd.notna(row.get("parsed_location")) else None,
            "objects": json.loads(row.get("parsed_objects") or "[]"),
        }

        pure_ranked_scenes, pure_rank_map, pure_score_map = scene_rankings_from_hits(pure_run.top_hits, PURE_SCENE_DEPTH)
        kg_ranked_scenes, kg_rank_map, kg_score_map = scene_rankings_from_hits(kg_run.top_hits, KG_SCENE_DEPTH)

        candidate_scene_tokens = collect_candidate_scene_tokens(
            seed_scene_token=seed_scene_token,
            strict_pool_tokens=strict_pool_tokens,
            pure_ranked_scenes=pure_ranked_scenes,
            kg_ranked_scenes=kg_ranked_scenes,
        )

        if seed_scene_token and seed_scene_token not in candidate_scene_tokens:
            candidate_scene_tokens.insert(0, seed_scene_token)
            candidate_scene_tokens = candidate_scene_tokens[:MAX_TRUTH_CANDIDATES]

        for candidate_rank, scene_token in enumerate(candidate_scene_tokens, start=1):
            scene_record = scene_record_by_token.get(scene_token, {})
            candidate_image_path, candidate_image_camera = choose_scene_image_path(
                scene_token, preferred_camera, by_scene_camera, fallback_by_scene
            )
            structured_match = scene_matches_structured_query(scene_record, parsed_query)
            detail_rows.append(
                {
                    "query_id": query_id,
                    "query_text": query_text,
                    "language": str(row.get("language") or ""),
                    "query_group": str(row.get("query_group") or ""),
                    "seed_image_path": str(row.get("resolved_image_path") or row.get("image_path") or ""),
                    "seed_scene_token": seed_scene_token,
                    "candidate_rank": candidate_rank,
                    "candidate_scene_token": scene_token,
                    "candidate_scene_name": scene_record.get("scene_name", ""),
                    "candidate_scene_description": scene_record.get("description", ""),
                    "candidate_weather": scene_record.get("weather", ""),
                    "candidate_timeofday": scene_record.get("timeofday", ""),
                    "candidate_location_area": scene_record.get("location_area", ""),
                    "candidate_location_kind": scene_record.get("location_kind", ""),
                    "candidate_num_samples": int(scene_record.get("num_samples", 0) or 0),
                    "candidate_object_summary": object_summary(scene_record),
                    "candidate_image_path": candidate_image_path,
                    "candidate_image_camera": candidate_image_camera,
                    "is_seed_scene": scene_token == seed_scene_token,
                    "in_strict_candidate_pool": scene_token in strict_pool_set,
                    "strict_candidate_scene_count": len(strict_pool_tokens),
                    "pure_clip_rank": pure_rank_map.get(scene_token, ""),
                    "pure_clip_score": pure_score_map.get(scene_token, ""),
                    "pure_clip_status": "ok" if not pure_error else pure_error,
                    "kg_clip_rank": kg_rank_map.get(scene_token, ""),
                    "kg_clip_score": kg_score_map.get(scene_token, ""),
                    "kg_clip_status": "ok" if not kg_error else kg_error,
                    "appears_in_both_rankings": scene_token in pure_rank_map and scene_token in kg_rank_map,
                    "structured_match": structured_match,
                    "candidate_reason": build_candidate_reason(
                        scene_token, seed_scene_token, strict_pool_set, pure_rank_map, kg_rank_map, structured_match
                    ),
                    "manual_relevant": "",
                    "review_note": "",
                }
            )

        benchmark_row = row.to_dict()
        benchmark_row.update(
            {
                "strict_candidate_scene_count": len(strict_pool_tokens),
                "strict_zero_candidate": bool(strict_zero),
                "strict_candidate_status": strict_status,
                "pure_clip_top_scene_tokens": json.dumps(pure_ranked_scenes[:PURE_SCENE_DEPTH], ensure_ascii=True),
                "pure_clip_status": "ok" if not pure_error else pure_error,
                "kg_clip_top_scene_tokens": json.dumps(kg_ranked_scenes[:KG_SCENE_DEPTH], ensure_ascii=True),
                "kg_clip_status": "ok" if not kg_error else kg_error,
                "candidate_scene_count": len(candidate_scene_tokens),
                "candidate_scene_tokens": json.dumps(candidate_scene_tokens, ensure_ascii=True),
                "candidate_generation_note": (
                    "Union of seed scene, strict KG candidate pool, Pure CLIP top scenes, and Strict KG+CLIP top scenes."
                ),
                "relevant_scene_tokens": "",
            }
        )
        benchmark_rows.append(benchmark_row)

    detail_df = pd.DataFrame(detail_rows)
    benchmark_out_df = pd.DataFrame(benchmark_rows)
    return detail_df, benchmark_out_df


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    detail_output_path = Path(args.detail_output)
    benchmark_output_path = Path(args.benchmark_output)

    benchmark_df = pd.read_csv(input_path)
    detail_df, benchmark_out_df = build_outputs(benchmark_df)

    detail_output_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_output_path.parent.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(detail_output_path, index=False, encoding="utf-8-sig")
    benchmark_out_df.to_csv(benchmark_output_path, index=False, encoding="utf-8-sig")

    summary = {
        "queries": int(len(benchmark_out_df)),
        "detail_rows": int(len(detail_df)),
        "detail_output": str(detail_output_path),
        "benchmark_output": str(benchmark_output_path),
        "avg_candidate_scene_count": float(benchmark_out_df["candidate_scene_count"].mean()) if not benchmark_out_df.empty else 0.0,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
