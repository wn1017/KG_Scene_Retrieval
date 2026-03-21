from __future__ import annotations

import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import ast
import json
import statistics
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import pandas as pd

import app

from config import DEFAULT_TOP_K


DEFAULT_MAP_DEPTH = 20
DEFAULT_FRAME_SEARCH_LIMIT = 80
DEFAULT_TIMING_WARMUP = 1
DEFAULT_TIMING_RUNS = 5
DEFAULT_STRATEGIES = ("pure_clip", "kg_clip_strict")
SUPPORTED_STRATEGIES = (*DEFAULT_STRATEGIES, "kg_clip_engineering")
NO_KG_FILTER_CANDIDATES = None


@dataclass
class RetrievalRun:
    strategy: str
    query_text: str
    model_name: str
    parsed_query: dict
    kg_status: str
    ranked_scene_tokens: list[str]
    top_hits: list[dict]
    candidate_scene_count: int | None
    strict_zero_candidate: bool
    used_full_collection: bool
    elapsed_seconds: float


def parse_relevant_scene_tokens(value) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    text = str(value).strip()
    if not text:
        return []

    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(text)
        except Exception:
            continue
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]

    normalized = text.replace(";", ",").replace("|", ",")
    return [item.strip() for item in normalized.split(",") if item.strip()]


def unique_scene_tokens_from_hits(hits: Iterable[dict], depth: int) -> list[str]:
    scene_tokens: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        scene_token = (hit.get("scene_token") or "").strip()
        if not scene_token or scene_token in seen:
            continue
        seen.add(scene_token)
        scene_tokens.append(scene_token)
        if len(scene_tokens) >= depth:
            break
    return scene_tokens


def precision_at_k(ranked_scene_tokens: list[str], relevant_scene_tokens: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    relevant = set(relevant_scene_tokens)
    hits = sum(1 for token in ranked_scene_tokens[:k] if token in relevant)
    return hits / float(k)


def recall_at_k(ranked_scene_tokens: list[str], relevant_scene_tokens: list[str], k: int) -> float:
    relevant = set(relevant_scene_tokens)
    if not relevant:
        return 0.0
    hits = sum(1 for token in ranked_scene_tokens[:k] if token in relevant)
    return hits / float(len(relevant))


def average_precision(ranked_scene_tokens: list[str], relevant_scene_tokens: list[str], depth: int) -> float:
    relevant = set(relevant_scene_tokens)
    if not relevant:
        return 0.0

    hit_count = 0
    cumulative_precision = 0.0
    for rank, scene_token in enumerate(ranked_scene_tokens[:depth], start=1):
        if scene_token in relevant:
            hit_count += 1
            cumulative_precision += hit_count / float(rank)
    return cumulative_precision / float(len(relevant))


def normalize_object_types(raw_value: str | None) -> set[str]:
    if not raw_value:
        return set()
    return {item.strip() for item in str(raw_value).split(",") if item.strip()}


def extract_location_kind(record: dict) -> str:
    location_value = str(record.get("location") or "").strip()
    if not location_value:
        return ""
    if ":" in location_value:
        return location_value.split(":")[-1].strip()
    return location_value


def hit_matches_structured_conditions(hit: dict, parsed_query: dict) -> bool:
    parsed_query = parsed_query or {}

    if parsed_query.get("weather") and str(hit.get("weather") or "").strip() != str(parsed_query["weather"]):
        return False
    if parsed_query.get("time") and str(hit.get("timeofday") or "").strip() != str(parsed_query["time"]):
        return False
    if parsed_query.get("location") and extract_location_kind(hit) != str(parsed_query["location"]):
        return False

    required_objects = parsed_query.get("objects") or []
    if required_objects:
        hit_objects = normalize_object_types(hit.get("obj_types"))
        if not all(object_name in hit_objects for object_name in required_objects):
            return False

    return True


def constraint_consistency_at_k(hits: list[dict], parsed_query: dict, k: int) -> float:
    if k <= 0:
        return 0.0

    top_hits = list(hits[:k])
    if not top_hits:
        return 0.0

    consistent_hits = sum(1 for hit in top_hits if hit_matches_structured_conditions(hit, parsed_query))
    return consistent_hits / float(len(top_hits))


def serialize_top_hits(hits: list[dict], k: int) -> str:
    preview = []
    for hit in hits[:k]:
        preview.append(
            {
                "id": int(hit.get("id", 0) or 0),
                "scene_token": str(hit.get("scene_token") or ""),
                "score": float(hit.get("score", 0.0) or 0.0),
                "weather": str(hit.get("weather") or ""),
                "timeofday": str(hit.get("timeofday") or ""),
                "location": str(hit.get("location") or ""),
                "obj_types": str(hit.get("obj_types") or ""),
            }
        )
    return json.dumps(preview, ensure_ascii=True)


def strict_candidate_scene_tokens(parsed_query: dict) -> tuple[list[str] | None, str, bool]:
    weather = parsed_query.get("weather")
    timeofday = parsed_query.get("time")
    object_types = parsed_query.get("objects") or []
    location_kind = parsed_query.get("location")

    if not app.has_kg_filter_conditions(parsed_query):
        return NO_KG_FILTER_CANDIDATES, "Strict KG+CLIP found no structured KG filters; searching the full collection.", False

    try:
        neo4j_tokens = app.sanitize_candidate_scene_tokens(
            app.query_scene_tokens(
                weather=weather,
                timeofday=timeofday,
                object_types=object_types,
                location_kind=location_kind,
            )
        )
        valid_neo4j_tokens = [token for token in neo4j_tokens if token in app.KNOWN_SCENE_TOKENS]
        if valid_neo4j_tokens:
            return valid_neo4j_tokens, f"Strict KG+CLIP kept {len(valid_neo4j_tokens)} candidate scenes from Neo4j.", False

        if neo4j_tokens:
            local_tokens = app.get_local_candidate_scene_tokens(
                weather=weather,
                timeofday=timeofday,
                object_types=object_types,
                location_kind=location_kind,
            )
            if local_tokens:
                return local_tokens, (
                    f"Strict KG+CLIP ignored out-of-universe Neo4j tokens and kept {len(local_tokens)} local KG scenes."
                ), False
            return [], "Strict KG+CLIP found 0 valid candidate scenes after Neo4j returned out-of-universe tokens.", True

        local_tokens = app.get_local_candidate_scene_tokens(
            weather=weather,
            timeofday=timeofday,
            object_types=object_types,
            location_kind=location_kind,
        )
        if local_tokens:
            return local_tokens, f"Strict KG+CLIP kept {len(local_tokens)} candidate scenes from the local KG fallback.", False
        return [], (
            "Strict KG+CLIP found 0 candidate scenes and stopped without similarity fallback. "
            f"Parsed conditions: {app.format_parsed_query(parsed_query)}"
        ), True
    except RuntimeError:
        local_tokens = app.get_local_candidate_scene_tokens(
            weather=weather,
            timeofday=timeofday,
            object_types=object_types,
            location_kind=location_kind,
        )
        if local_tokens:
            return local_tokens, f"Strict KG+CLIP used the local KG and kept {len(local_tokens)} candidate scenes.", False
        return [], (
            "Neo4j was unavailable, the local KG also returned 0 candidate scenes, and Strict KG+CLIP stopped without "
            f"similarity fallback. Parsed conditions: {app.format_parsed_query(parsed_query)}"
        ), True


def run_pure_clip(query_text: str, frame_search_limit: int, map_depth: int, k: int) -> RetrievalRun:
    started_at = time.perf_counter()
    parsed_query = app.parse_query(query_text)
    query_vector, model_name = app.encode_text_query(query_text)
    hits = app.search_frame_hits(query_vector, frame_search_limit)
    ranked_scene_tokens = unique_scene_tokens_from_hits(hits, map_depth)
    elapsed = time.perf_counter() - started_at
    return RetrievalRun(
        strategy="pure_clip",
        query_text=query_text,
        model_name=model_name,
        parsed_query=parsed_query,
        kg_status="Pure CLIP baseline without KG filtering.",
        ranked_scene_tokens=ranked_scene_tokens,
        top_hits=hits[:k],
        candidate_scene_count=None,
        strict_zero_candidate=False,
        used_full_collection=True,
        elapsed_seconds=elapsed,
    )


def run_kg_clip_strict(query_text: str, frame_search_limit: int, map_depth: int, k: int) -> RetrievalRun:
    started_at = time.perf_counter()
    parsed_query = app.parse_query(query_text)
    candidate_scene_tokens, kg_status, strict_zero_candidate = strict_candidate_scene_tokens(parsed_query)

    if strict_zero_candidate:
        elapsed = time.perf_counter() - started_at
        return RetrievalRun(
            strategy="kg_clip_strict",
            query_text=query_text,
            model_name=app.KG_ZERO_CANDIDATE_MODEL_NAME,
            parsed_query=parsed_query,
            kg_status=kg_status,
            ranked_scene_tokens=[],
            top_hits=[],
            candidate_scene_count=0,
            strict_zero_candidate=True,
            used_full_collection=False,
            elapsed_seconds=elapsed,
        )

    query_vector, model_name = app.encode_text_query(query_text)
    if candidate_scene_tokens is NO_KG_FILTER_CANDIDATES:
        hits = app.search_frame_hits(query_vector, frame_search_limit)
        used_full_collection = True
        candidate_scene_count = None
    else:
        hits = app.search_frame_hits(query_vector, frame_search_limit, candidate_scene_tokens)
        used_full_collection = False
        candidate_scene_count = len(candidate_scene_tokens)

    ranked_scene_tokens = unique_scene_tokens_from_hits(hits, map_depth)
    elapsed = time.perf_counter() - started_at
    return RetrievalRun(
        strategy="kg_clip_strict",
        query_text=query_text,
        model_name=model_name,
        parsed_query=parsed_query,
        kg_status=kg_status,
        ranked_scene_tokens=ranked_scene_tokens,
        top_hits=hits[:k],
        candidate_scene_count=candidate_scene_count,
        strict_zero_candidate=False,
        used_full_collection=used_full_collection,
        elapsed_seconds=elapsed,
    )


def run_kg_clip_engineering(query_text: str, frame_search_limit: int, map_depth: int, k: int) -> RetrievalRun:
    started_at = time.perf_counter()
    parsed_query = app.parse_query(query_text)
    candidate_scene_tokens, kg_status, _should_stop = app.get_candidate_scene_tokens(parsed_query)
    query_vector, model_name = app.encode_text_query(query_text)

    hits = app.search_frame_hits(query_vector, frame_search_limit, candidate_scene_tokens)
    ranked_scene_tokens = unique_scene_tokens_from_hits(hits, map_depth)
    used_full_collection = not bool(candidate_scene_tokens)

    if candidate_scene_tokens and not ranked_scene_tokens:
        hits = app.search_frame_hits(query_vector, frame_search_limit)
        ranked_scene_tokens = unique_scene_tokens_from_hits(hits, map_depth)
        kg_status = kg_status + " Filter returned no indexed scenes; fallback to full collection."
        used_full_collection = True

    elapsed = time.perf_counter() - started_at
    return RetrievalRun(
        strategy="kg_clip_engineering",
        query_text=query_text,
        model_name=model_name,
        parsed_query=parsed_query,
        kg_status=kg_status,
        ranked_scene_tokens=ranked_scene_tokens,
        top_hits=hits[:k],
        candidate_scene_count=len(candidate_scene_tokens) if candidate_scene_tokens else None,
        strict_zero_candidate=False,
        used_full_collection=used_full_collection,
        elapsed_seconds=elapsed,
    )


def measure_retrieval_run(
    run_fn,
    query_text: str,
    frame_search_limit: int,
    map_depth: int,
    k: int,
    timing_warmup: int,
    timing_runs: int,
) -> RetrievalRun:
    if timing_runs <= 0:
        raise ValueError("timing_runs must be >= 1")
    if timing_warmup < 0:
        raise ValueError("timing_warmup must be >= 0")

    for _ in range(timing_warmup):
        run_fn(query_text, frame_search_limit=frame_search_limit, map_depth=map_depth, k=k)

    timed_runs: list[RetrievalRun] = []
    for _ in range(timing_runs):
        timed_runs.append(run_fn(query_text, frame_search_limit=frame_search_limit, map_depth=map_depth, k=k))

    reference_run = timed_runs[0]
    median_elapsed = float(statistics.median(run.elapsed_seconds for run in timed_runs))
    return replace(reference_run, elapsed_seconds=median_elapsed)


def evaluate_strategy(
    dataframe: pd.DataFrame,
    strategy: str,
    query_col: str,
    relevant_col: str,
    k: int,
    map_depth: int,
    frame_search_limit: int,
    timing_warmup: int,
    timing_runs: int,
) -> tuple[dict, pd.DataFrame]:
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(f"Unsupported strategy: {strategy}")

    run_functions = {
        "pure_clip": run_pure_clip,
        "kg_clip_strict": run_kg_clip_strict,
        "kg_clip_engineering": run_kg_clip_engineering,
    }
    run_fn = run_functions[strategy]
    rows: list[dict] = []

    for row_index, row in dataframe.iterrows():
        query_text = str(row[query_col]).strip()
        relevant_scene_tokens = parse_relevant_scene_tokens(row[relevant_col])
        run = measure_retrieval_run(
            run_fn,
            query_text,
            frame_search_limit=frame_search_limit,
            map_depth=map_depth,
            k=k,
            timing_warmup=timing_warmup,
            timing_runs=timing_runs,
        )
        consistency_value = constraint_consistency_at_k(run.top_hits, run.parsed_query, k)

        rows.append(
            {
                "row_index": int(row_index),
                "query_id": str(row.get("query_id", row_index)),
                "query_group": str(row.get("query_group", "")),
                "strategy": run.strategy,
                "query_text": query_text,
                "model_name": run.model_name,
                "parsed_query": json.dumps(run.parsed_query, ensure_ascii=True),
                "kg_status": run.kg_status,
                "candidate_scene_count": run.candidate_scene_count,
                "strict_zero_candidate": run.strict_zero_candidate,
                "used_full_collection": run.used_full_collection,
                "relevant_scene_tokens": json.dumps(relevant_scene_tokens, ensure_ascii=True),
                "ranked_scene_tokens": json.dumps(run.ranked_scene_tokens, ensure_ascii=True),
                "top_hits": serialize_top_hits(run.top_hits, k),
                "returned_scene_count": len(run.ranked_scene_tokens),
                "returned_hit_count": len(run.top_hits),
                f"precision@{k}": precision_at_k(run.ranked_scene_tokens, relevant_scene_tokens, k),
                f"recall@{k}": recall_at_k(run.ranked_scene_tokens, relevant_scene_tokens, k),
                "average_precision": average_precision(run.ranked_scene_tokens, relevant_scene_tokens, map_depth),
                f"constraint_consistency@{k}": consistency_value,
                "response_time_seconds": run.elapsed_seconds,
                "response_time_protocol": "end_to_end_median",
                "timing_warmup": timing_warmup,
                "timing_runs": timing_runs,
            }
        )

    result_df = pd.DataFrame(rows)
    summary = {
        "strategy": strategy,
        "queries": int(len(result_df)),
        f"precision@{k}": float(result_df[f"precision@{k}"].mean()) if not result_df.empty else 0.0,
        f"recall@{k}": float(result_df[f"recall@{k}"].mean()) if not result_df.empty else 0.0,
        "mAP": float(result_df["average_precision"].mean()) if not result_df.empty else 0.0,
        f"constraint_consistency@{k}": float(result_df[f"constraint_consistency@{k}"].mean()) if not result_df.empty else 0.0,
        "avg_response_time_seconds": float(result_df["response_time_seconds"].mean()) if not result_df.empty else 0.0,
        "response_time_protocol": "query_level_median_then_macro_average",
        "timing_warmup": timing_warmup,
        "timing_runs": timing_runs,
        "strict_zero_candidate_queries": int(result_df["strict_zero_candidate"].sum()) if not result_df.empty else 0,
    }
    return summary, result_df


def load_eval_dataframe(csv_path: Path, query_col: str, relevant_col: str, max_queries: int | None) -> pd.DataFrame:
    dataframe = pd.read_csv(csv_path)
    missing_cols = [column for column in (query_col, relevant_col) if column not in dataframe.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    dataframe = dataframe.copy()
    dataframe[query_col] = dataframe[query_col].astype(str)
    dataframe = dataframe[dataframe[query_col].str.strip() != ""]
    dataframe = dataframe.reset_index(drop=True)
    if max_queries is not None:
        dataframe = dataframe.head(max_queries)
    return dataframe


def build_group_summary(table: pd.DataFrame, k: int) -> pd.DataFrame:
    if "query_group" not in table.columns:
        return pd.DataFrame()

    grouped_rows: list[dict] = []
    for query_group, group_df in table.groupby("query_group", dropna=False):
        grouped_rows.append(
            {
                "strategy": group_df["strategy"].iloc[0] if not group_df.empty else "",
                "query_group": str(query_group or ""),
                "queries": int(len(group_df)),
                f"precision@{k}": float(group_df[f"precision@{k}"].mean()),
                f"recall@{k}": float(group_df[f"recall@{k}"].mean()),
                "mAP": float(group_df["average_precision"].mean()),
                f"constraint_consistency@{k}": float(group_df[f"constraint_consistency@{k}"].mean()),
                "avg_response_time_seconds": float(group_df["response_time_seconds"].mean()),
                "response_time_protocol": str(group_df["response_time_protocol"].iloc[0]),
                "timing_warmup": int(group_df["timing_warmup"].iloc[0]),
                "timing_runs": int(group_df["timing_runs"].iloc[0]),
                "strict_zero_candidate_queries": int(group_df["strict_zero_candidate"].sum()),
            }
        )
    return pd.DataFrame(grouped_rows)


def save_outputs(output_dir: Path, summaries: list[dict], result_tables: list[pd.DataFrame], k: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(summaries).to_csv(output_dir / "evaluation_summary.csv", index=False, encoding="utf-8-sig")

    for table in result_tables:
        strategy = table["strategy"].iloc[0] if not table.empty else "empty"
        table.to_csv(output_dir / f"{strategy}_details.csv", index=False, encoding="utf-8-sig")
        group_summary = build_group_summary(table, k)
        if not group_summary.empty:
            group_summary.to_csv(output_dir / f"{strategy}_group_summary.csv", index=False, encoding="utf-8-sig")


def print_summary_table(summaries: list[dict], k: int) -> None:
    summary_df = pd.DataFrame(summaries)
    print("Evaluation summary")
    print(summary_df.to_string(index=False, justify="left"))
    print()
    print(
        f"Metrics reported: Precision@{k}, Recall@{k}, mAP, ConstraintConsistency@{k}, and end-to-end response time "
        "reported as the query-level median over repeated runs."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Pure CLIP vs Strict KG+CLIP scene retrieval")
    parser.add_argument("csv_path", help="CSV containing query_text and relevant_scene_tokens")
    parser.add_argument("--query-col", default="query_text")
    parser.add_argument("--relevant-col", default="relevant_scene_tokens")
    parser.add_argument("--k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--map-depth", type=int, default=DEFAULT_MAP_DEPTH)
    parser.add_argument("--frame-search-limit", type=int, default=DEFAULT_FRAME_SEARCH_LIMIT)
    parser.add_argument("--timing-warmup", type=int, default=DEFAULT_TIMING_WARMUP)
    parser.add_argument("--timing-runs", type=int, default=DEFAULT_TIMING_RUNS)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=list(DEFAULT_STRATEGIES),
        choices=SUPPORTED_STRATEGIES,
        help="Strategies to evaluate. Default keeps the academic comparison strict.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Evaluation CSV not found: {csv_path}")

    if app.get_live_collection() is None:
        raise RuntimeError(
            "Milvus is unavailable. Start Milvus first, rebuild the collection if needed, then rerun evaluation."
        )

    eval_df = load_eval_dataframe(
        csv_path=csv_path,
        query_col=args.query_col,
        relevant_col=args.relevant_col,
        max_queries=args.max_queries,
    )

    summaries: list[dict] = []
    result_tables: list[pd.DataFrame] = []
    for strategy in args.strategies:
        summary, result_df = evaluate_strategy(
            dataframe=eval_df,
            strategy=strategy,
            query_col=args.query_col,
            relevant_col=args.relevant_col,
            k=args.k,
            map_depth=args.map_depth,
            frame_search_limit=args.frame_search_limit,
            timing_warmup=args.timing_warmup,
            timing_runs=args.timing_runs,
        )
        summaries.append(summary)
        result_tables.append(result_df)

    print_summary_table(summaries, args.k)

    if args.output_dir:
        save_outputs(Path(args.output_dir), summaries, result_tables, args.k)
        print(f"Saved evaluation outputs to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
