from __future__ import annotations

import json
import pathlib
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app
from scripts.evaluate import strict_candidate_scene_tokens


DEFAULT_INPUT = PROJECT_ROOT / "benchmark" / "manual_query_benchmark.csv"
DEFAULT_QUERY_OUTPUT = PROJECT_ROOT / "benchmark" / "manual_query_candidate_queries.csv"
DEFAULT_SCENE_OUTPUT = PROJECT_ROOT / "benchmark" / "manual_query_candidate_scenes.csv"
DEFAULT_FRAME_SEARCH_LIMIT = 120
DEFAULT_SCENE_DEPTH = 20
DEFAULT_OPEN_SEMANTIC_DEPTH = 12


def unique_scene_tokens_from_hits(hits: list[dict], depth: int) -> list[str]:
    ordered_tokens: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        scene_token = str(hit.get("scene_token") or "").strip()
        if not scene_token or scene_token in seen:
            continue
        seen.add(scene_token)
        ordered_tokens.append(scene_token)
        if len(ordered_tokens) >= depth:
            break
    return ordered_tokens


def best_scores_by_scene(hits: list[dict]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for hit in hits:
        scene_token = str(hit.get("scene_token") or "").strip()
        if not scene_token:
            continue
        score = float(hit.get("score", 0.0) or 0.0)
        scores[scene_token] = max(score, scores.get(scene_token, float("-inf")))
    return scores


def build_scene_summary(scene_token: str) -> dict[str, object]:
    record = app.KG_RECORD_BY_SCENE_TOKEN.get(scene_token, {})
    objects = record.get("objects", {}) if isinstance(record, dict) else {}
    return {
        "scene_token": scene_token,
        "scene_name": record.get("scene_name", ""),
        "scene_description": record.get("description", ""),
        "scene_weather": record.get("weather", ""),
        "scene_timeofday": record.get("timeofday", ""),
        "scene_location_area": record.get("location_area", ""),
        "scene_location_kind": record.get("location_kind", ""),
        "scene_objects": ",".join(objects.keys()) if objects else "",
        "scene_object_counts": json.dumps(objects, ensure_ascii=False) if objects else "{}",
    }


def run_candidate_expansion(query_text: str, query_group: str) -> dict[str, object]:
    parsed_query = app.parse_query(query_text)
    if app.has_kg_filter_conditions(parsed_query):
        candidate_scene_tokens, kg_status, strict_zero = strict_candidate_scene_tokens(parsed_query)
        if strict_zero:
            return {
                "parsed_query": parsed_query,
                "candidate_source": "strict_kg_zero",
                "kg_status": kg_status,
                "candidate_scene_tokens": [],
                "ranked_scene_tokens": [],
                "top_hits": [],
                "best_scene_scores": {},
            }

        query_vector, model_name = app.encode_text_query(query_text)
        del model_name
        hits = app.search_frame_hits(
            query_vector,
            DEFAULT_FRAME_SEARCH_LIMIT,
            candidate_scene_tokens if candidate_scene_tokens is not None else None,
        )
        ranked_scene_tokens = unique_scene_tokens_from_hits(hits, DEFAULT_SCENE_DEPTH)
        return {
            "parsed_query": parsed_query,
            "candidate_source": "strict_kg_ranked",
            "kg_status": kg_status,
            "candidate_scene_tokens": list(candidate_scene_tokens or []),
            "ranked_scene_tokens": ranked_scene_tokens,
            "top_hits": hits,
            "best_scene_scores": best_scores_by_scene(hits),
        }

    query_vector, model_name = app.encode_text_query(query_text)
    del model_name
    hits = app.search_frame_hits(query_vector, DEFAULT_FRAME_SEARCH_LIMIT)
    ranked_scene_tokens = unique_scene_tokens_from_hits(hits, DEFAULT_OPEN_SEMANTIC_DEPTH)
    return {
        "parsed_query": parsed_query,
        "candidate_source": "open_semantic_clip_ranked",
        "kg_status": "No structured KG filters; using top CLIP-ranked scenes as manual candidate pool.",
        "candidate_scene_tokens": ranked_scene_tokens,
        "ranked_scene_tokens": ranked_scene_tokens,
        "top_hits": hits,
        "best_scene_scores": best_scores_by_scene(hits),
    }


def build_outputs(input_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    benchmark_df = pd.read_csv(input_path)
    query_rows: list[dict[str, object]] = []
    scene_rows: list[dict[str, object]] = []

    for _, row in benchmark_df.iterrows():
        query_id = str(row.get("query_id", "") or "")
        query_text = str(row.get("query_text", "") or "")
        query_group = str(row.get("query_group", "") or "")
        seed_scene_token = str(row.get("scene_token", "") or "")

        expansion = run_candidate_expansion(query_text, query_group)
        parsed_query = expansion["parsed_query"]
        candidate_scene_tokens = list(expansion["candidate_scene_tokens"])
        ranked_scene_tokens = list(expansion["ranked_scene_tokens"])
        best_scene_scores = dict(expansion["best_scene_scores"])

        suggested_relevant_scene_tokens = ranked_scene_tokens if ranked_scene_tokens else candidate_scene_tokens
        seed_scene_in_candidates = seed_scene_token in set(candidate_scene_tokens)
        seed_scene_in_ranked = seed_scene_token in set(ranked_scene_tokens)

        query_rows.append(
            {
                **row.to_dict(),
                "candidate_source": expansion["candidate_source"],
                "kg_status_for_candidates": expansion["kg_status"],
                "candidate_scene_count": len(candidate_scene_tokens),
                "candidate_scene_tokens": json.dumps(candidate_scene_tokens, ensure_ascii=False),
                "ranked_candidate_scene_tokens": json.dumps(ranked_scene_tokens, ensure_ascii=False),
                "suggested_relevant_scene_tokens": json.dumps(suggested_relevant_scene_tokens, ensure_ascii=False),
                "seed_scene_in_candidates": seed_scene_in_candidates,
                "seed_scene_in_ranked": seed_scene_in_ranked,
                "candidate_preview_top5": json.dumps(ranked_scene_tokens[:5], ensure_ascii=False),
                "parsed_query_json": json.dumps(parsed_query, ensure_ascii=False),
            }
        )

        reference_tokens: list[tuple[str, str]]
        reference_tokens = []
        if seed_scene_token:
            reference_tokens.append(("seed_scene", seed_scene_token))
        for scene_token in ranked_scene_tokens:
            if scene_token == seed_scene_token:
                continue
            reference_tokens.append(("candidate_scene", scene_token))

        seen_reference_tokens: set[str] = set()
        for rank, (row_type, scene_token) in enumerate(reference_tokens, start=1):
            if not scene_token or scene_token in seen_reference_tokens:
                continue
            seen_reference_tokens.add(scene_token)
            scene_rows.append(
                {
                    "query_id": query_id,
                    "query_text": query_text,
                    "query_group": query_group,
                    "row_type": row_type,
                    "rank": rank if row_type == "candidate_scene" else 0,
                    "is_seed_scene": scene_token == seed_scene_token,
                    "in_candidate_scene_tokens": scene_token in set(candidate_scene_tokens),
                    "in_ranked_candidate_scene_tokens": scene_token in set(ranked_scene_tokens),
                    "best_scene_score": best_scene_scores.get(scene_token, ""),
                    **build_scene_summary(scene_token),
                }
            )

    return pd.DataFrame(query_rows), pd.DataFrame(scene_rows)


def main() -> None:
    query_df, scene_df = build_outputs(DEFAULT_INPUT)
    DEFAULT_QUERY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    query_df.to_csv(DEFAULT_QUERY_OUTPUT, index=False, encoding="utf-8-sig")
    scene_df.to_csv(DEFAULT_SCENE_OUTPUT, index=False, encoding="utf-8-sig")

    print(f"queries={len(query_df)}")
    print(f"query_output={DEFAULT_QUERY_OUTPUT}")
    print(f"scene_output={DEFAULT_SCENE_OUTPUT}")
    print(query_df["candidate_source"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
