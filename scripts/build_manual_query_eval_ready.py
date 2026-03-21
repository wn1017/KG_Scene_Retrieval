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


DEFAULT_DETAIL_INPUT = PROJECT_ROOT / "benchmark" / "manual_query_truth_candidates.csv"
DEFAULT_BENCHMARK_INPUT = PROJECT_ROOT / "benchmark" / "manual_query_benchmark_candidates.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "benchmark" / "manual_query_benchmark_eval_ready.csv"

POSITIVE_LABELS = {
    "1",
    "true",
    "t",
    "yes",
    "y",
    "relevant",
    "positive",
    "hit",
    "\u662f",
    "\u76f8\u5173",
    "\u547d\u4e2d",
}
NEGATIVE_LABELS = {
    "0",
    "false",
    "f",
    "no",
    "n",
    "irrelevant",
    "negative",
    "miss",
    "\u5426",
    "\u4e0d\u76f8\u5173",
    "\u672a\u547d\u4e2d",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate manual relevant labels into an evaluation-ready manual benchmark CSV."
    )
    parser.add_argument("--detail-input", default=str(DEFAULT_DETAIL_INPUT), help="manual_query_truth_candidates.csv path.")
    parser.add_argument(
        "--benchmark-input",
        default=str(DEFAULT_BENCHMARK_INPUT),
        help="manual_query_benchmark_candidates.csv path.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output CSV path with relevant_scene_tokens filled from manual labels.",
    )
    return parser.parse_args()


def safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_manual_relevant(value: object) -> tuple[bool | None, str]:
    text = str(value or "").strip()
    if not text:
        return None, ""
    normalized = text.casefold()
    if normalized in POSITIVE_LABELS:
        return True, text
    if normalized in NEGATIVE_LABELS:
        return False, text
    return None, text


def ordered_unique(tokens: list[str]) -> list[str]:
    ordered: list[str] = []
    for token in tokens:
        token = str(token or "").strip()
        if token and token not in ordered:
            ordered.append(token)
    return ordered


def build_eval_ready_dataframe(
    detail_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    label_source_name: str,
) -> tuple[pd.DataFrame, dict]:
    detail_df = detail_df.copy()
    benchmark_df = benchmark_df.copy()

    if "query_id" not in detail_df.columns or "manual_relevant" not in detail_df.columns:
        raise ValueError("Detail CSV must contain query_id and manual_relevant columns.")
    if "query_id" not in benchmark_df.columns:
        raise ValueError("Benchmark CSV must contain query_id column.")

    detail_df["query_id"] = detail_df["query_id"].astype(str)
    benchmark_df["query_id"] = benchmark_df["query_id"].astype(str)

    output_rows: list[dict] = []
    ready_queries = 0
    awaiting_queries = 0
    labeled_no_positive_queries = 0
    missing_candidate_queries = 0
    unknown_label_queries = 0
    extra_detail_query_ids = sorted(set(detail_df["query_id"]) - set(benchmark_df["query_id"]))

    for _, row in benchmark_df.iterrows():
        query_id = str(row["query_id"])
        query_detail_df = detail_df[detail_df["query_id"] == query_id].copy()
        if not query_detail_df.empty and "candidate_rank" in query_detail_df.columns:
            query_detail_df = query_detail_df.sort_values(
                by="candidate_rank",
                key=lambda series: series.map(lambda value: safe_int(value, default=10**9)),
            )

        relevant_scene_tokens: list[str] = []
        labeled_count = 0
        positive_count = 0
        unknown_labels: list[str] = []

        for _, detail_row in query_detail_df.iterrows():
            normalized_value, raw_label = normalize_manual_relevant(detail_row.get("manual_relevant", ""))
            if normalized_value is None:
                if raw_label:
                    unknown_labels.append(raw_label)
                continue

            labeled_count += 1
            if normalized_value:
                positive_count += 1
                relevant_scene_tokens.append(str(detail_row.get("candidate_scene_token") or "").strip())

        relevant_scene_tokens = ordered_unique(relevant_scene_tokens)
        candidate_count = int(len(query_detail_df))

        if relevant_scene_tokens:
            truth_status = "ready"
            ready_queries += 1
        elif labeled_count > 0:
            truth_status = "labeled_no_positive"
            labeled_no_positive_queries += 1
        elif candidate_count > 0:
            truth_status = "awaiting_manual_review"
            awaiting_queries += 1
        else:
            truth_status = "missing_candidates"
            missing_candidate_queries += 1

        unknown_labels = ordered_unique(unknown_labels)
        if unknown_labels:
            unknown_label_queries += 1

        output_row = row.to_dict()
        output_row.update(
            {
                "relevant_scene_tokens": json.dumps(relevant_scene_tokens, ensure_ascii=True),
                "relevant_scene_count": len(relevant_scene_tokens),
                "truth_candidate_count": candidate_count,
                "truth_labeled_count": labeled_count,
                "truth_positive_count": positive_count,
                "truth_status": truth_status,
                "truth_unknown_labels": "|".join(unknown_labels),
                "truth_label_source": label_source_name,
            }
        )
        output_rows.append(output_row)

    output_df = pd.DataFrame(output_rows)
    summary = {
        "queries": int(len(output_df)),
        "ready_queries": int(ready_queries),
        "awaiting_manual_review_queries": int(awaiting_queries),
        "labeled_no_positive_queries": int(labeled_no_positive_queries),
        "missing_candidate_queries": int(missing_candidate_queries),
        "queries_with_unknown_labels": int(unknown_label_queries),
        "avg_relevant_scene_count": float(output_df["relevant_scene_count"].mean()) if not output_df.empty else 0.0,
        "extra_detail_query_ids": extra_detail_query_ids,
    }
    return output_df, summary


def main() -> None:
    args = parse_args()
    detail_input = Path(args.detail_input)
    benchmark_input = Path(args.benchmark_input)
    output_path = Path(args.output)

    if not detail_input.exists():
        raise FileNotFoundError(f"Detail input not found: {detail_input}")
    if not benchmark_input.exists():
        raise FileNotFoundError(f"Benchmark input not found: {benchmark_input}")

    detail_df = pd.read_csv(detail_input, keep_default_na=False)
    benchmark_df = pd.read_csv(benchmark_input, keep_default_na=False)

    output_df, summary = build_eval_ready_dataframe(
        detail_df=detail_df,
        benchmark_df=benchmark_df,
        label_source_name=detail_input.name,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    summary["output"] = str(output_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
