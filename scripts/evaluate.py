from __future__ import annotations

import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import ast
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

import app
from config import DEFAULT_TOP_K


DEFAULT_MAP_DEPTH = 20
DEFAULT_FRAME_SEARCH_LIMIT = 80


@dataclass
class RetrievalRun:
    strategy: str
    query_text: str
    model_name: str
    parsed_query: dict
    kg_status: str
    ranked_scene_tokens: list[str]
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


def run_pure_clip(query_text: str, frame_search_limit: int, map_depth: int) -> RetrievalRun:
    started_at = time.perf_counter()
    query_vector, model_name = app.encode_text_query(query_text)
    hits = app.search_frame_hits(query_vector, frame_search_limit)
    ranked_scene_tokens = unique_scene_tokens_from_hits(hits, map_depth)
    elapsed = time.perf_counter() - started_at
    return RetrievalRun(
        strategy="pure_clip",
        query_text=query_text,
        model_name=model_name,
        parsed_query={},
        kg_status="Pure CLIP baseline without KG filtering.",
        ranked_scene_tokens=ranked_scene_tokens,
        elapsed_seconds=elapsed,
    )


def run_kg_clip(query_text: str, frame_search_limit: int, map_depth: int) -> RetrievalRun:
    started_at = time.perf_counter()
    parsed_query = app.parse_query(query_text)
    candidate_scene_tokens, kg_status = app.get_candidate_scene_tokens(parsed_query)
    query_vector, model_name = app.encode_text_query(query_text)

    hits = app.search_frame_hits(query_vector, frame_search_limit, candidate_scene_tokens)
    ranked_scene_tokens = unique_scene_tokens_from_hits(hits, map_depth)

    if candidate_scene_tokens and not ranked_scene_tokens:
        hits = app.search_frame_hits(query_vector, frame_search_limit)
        ranked_scene_tokens = unique_scene_tokens_from_hits(hits, map_depth)
        kg_status = kg_status + " Filter returned no indexed scenes; fallback to full collection."

    elapsed = time.perf_counter() - started_at
    return RetrievalRun(
        strategy="kg_clip",
        query_text=query_text,
        model_name=model_name,
        parsed_query=parsed_query,
        kg_status=kg_status,
        ranked_scene_tokens=ranked_scene_tokens,
        elapsed_seconds=elapsed,
    )


def evaluate_strategy(
    dataframe: pd.DataFrame,
    strategy: str,
    query_col: str,
    relevant_col: str,
    k: int,
    map_depth: int,
    frame_search_limit: int,
) -> tuple[dict, pd.DataFrame]:
    if strategy not in {"pure_clip", "kg_clip"}:
        raise ValueError(f"Unsupported strategy: {strategy}")

    run_fn = run_pure_clip if strategy == "pure_clip" else run_kg_clip
    rows: list[dict] = []

    for row_index, row in dataframe.iterrows():
        query_text = str(row[query_col]).strip()
        relevant_scene_tokens = parse_relevant_scene_tokens(row[relevant_col])
        run = run_fn(query_text, frame_search_limit=frame_search_limit, map_depth=map_depth)

        rows.append(
            {
                "row_index": int(row_index),
                "strategy": run.strategy,
                "query_text": query_text,
                "model_name": run.model_name,
                "parsed_query": json.dumps(run.parsed_query, ensure_ascii=True),
                "kg_status": run.kg_status,
                "relevant_scene_tokens": json.dumps(relevant_scene_tokens, ensure_ascii=True),
                "ranked_scene_tokens": json.dumps(run.ranked_scene_tokens, ensure_ascii=True),
                f"precision@{k}": precision_at_k(run.ranked_scene_tokens, relevant_scene_tokens, k),
                f"recall@{k}": recall_at_k(run.ranked_scene_tokens, relevant_scene_tokens, k),
                "average_precision": average_precision(run.ranked_scene_tokens, relevant_scene_tokens, map_depth),
                "response_time_seconds": run.elapsed_seconds,
            }
        )

    result_df = pd.DataFrame(rows)
    summary = {
        "strategy": strategy,
        "queries": int(len(result_df)),
        f"precision@{k}": float(result_df[f"precision@{k}"].mean()) if not result_df.empty else 0.0,
        f"recall@{k}": float(result_df[f"recall@{k}"].mean()) if not result_df.empty else 0.0,
        "mAP": float(result_df["average_precision"].mean()) if not result_df.empty else 0.0,
        "avg_response_time_seconds": float(result_df["response_time_seconds"].mean()) if not result_df.empty else 0.0,
    }
    return summary, result_df


def load_eval_dataframe(csv_path: Path, query_col: str, relevant_col: str, max_queries: int | None) -> pd.DataFrame:
    dataframe = pd.read_csv(csv_path)
    missing_cols = [column for column in (query_col, relevant_col) if column not in dataframe.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    dataframe = dataframe[[query_col, relevant_col]].copy()
    dataframe[query_col] = dataframe[query_col].astype(str)
    dataframe = dataframe[dataframe[query_col].str.strip() != ""]
    dataframe = dataframe.reset_index(drop=True)
    if max_queries is not None:
        dataframe = dataframe.head(max_queries)
    return dataframe


def save_outputs(output_dir: Path, summaries: list[dict], result_tables: list[pd.DataFrame]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    for table in result_tables:
        strategy = table["strategy"].iloc[0] if not table.empty else "empty"
        table.to_csv(output_dir / f"{strategy}_details.csv", index=False, encoding="utf-8-sig")


def print_summary_table(summaries: list[dict], k: int) -> None:
    summary_df = pd.DataFrame(summaries)
    print("Evaluation summary")
    print(summary_df.to_string(index=False, justify="left"))
    print()
    print(f"Metrics reported: Precision@{k}, Recall@{k}, mAP, average response time")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Pure CLIP vs KG+CLIP scene retrieval")
    parser.add_argument("csv_path", help="CSV containing query_text and relevant_scene_tokens")
    parser.add_argument("--query-col", default="query_text")
    parser.add_argument("--relevant-col", default="relevant_scene_tokens")
    parser.add_argument("--k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--map-depth", type=int, default=DEFAULT_MAP_DEPTH)
    parser.add_argument("--frame-search-limit", type=int, default=DEFAULT_FRAME_SEARCH_LIMIT)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
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
    for strategy in ("pure_clip", "kg_clip"):
        summary, result_df = evaluate_strategy(
            dataframe=eval_df,
            strategy=strategy,
            query_col=args.query_col,
            relevant_col=args.relevant_col,
            k=args.k,
            map_depth=args.map_depth,
            frame_search_limit=args.frame_search_limit,
        )
        summaries.append(summary)
        result_tables.append(result_df)

    print_summary_table(summaries, args.k)

    if args.output_dir:
        save_outputs(Path(args.output_dir), summaries, result_tables)
        print(f"Saved evaluation outputs to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
