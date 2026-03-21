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


DEFAULT_INPUT = PROJECT_ROOT / "benchmark" / "manual_query_benchmark.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "benchmark" / "manual_query_benchmark_seed_scene_eval.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a seed-scene evaluation CSV for the manual query benchmark."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input manual_query_benchmark.csv path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output seed-scene eval CSV path.")
    return parser.parse_args()


def build_seed_scene_eval_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    if "query_id" not in dataframe.columns or "scene_token" not in dataframe.columns:
        raise ValueError("Input CSV must contain query_id and scene_token columns.")
    if "query_text" not in dataframe.columns:
        raise ValueError("Input CSV must contain query_text column.")

    output_df = dataframe.copy()
    output_df["scene_token"] = output_df["scene_token"].astype(str).str.strip()

    missing_scene_mask = output_df["scene_token"] == ""
    if bool(missing_scene_mask.any()):
        missing_queries = output_df.loc[missing_scene_mask, "query_id"].tolist()
        raise ValueError(f"Missing scene_token for queries: {missing_queries}")

    output_df["relevant_scene_tokens"] = output_df["scene_token"].map(
        lambda token: json.dumps([token], ensure_ascii=True)
    )
    output_df["relevant_scene_count"] = 1
    output_df["truth_definition"] = "seed_scene_anchor"
    output_df["truth_status"] = "ready"
    output_df["truth_note"] = (
        "Scene-level ground truth anchored to the author-selected seed scene. "
        "A result counts as relevant if it belongs to the same scene token."
    )
    return output_df


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input benchmark not found: {input_path}")

    dataframe = pd.read_csv(input_path, keep_default_na=False)
    output_df = build_seed_scene_eval_dataframe(dataframe)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    summary = {
        "queries": int(len(output_df)),
        "output": str(output_path),
        "truth_definition": "seed_scene_anchor",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
