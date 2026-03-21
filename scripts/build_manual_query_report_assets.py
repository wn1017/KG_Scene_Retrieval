from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from pathlib import Path

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import PercentFormatter


DEFAULT_RUN_DIR = PROJECT_ROOT / "benchmark" / "runs" / "manual_query_seed_scene_main_e2e_median"

STRATEGY_LABELS = {
    "pure_clip": "Pure CLIP",
    "kg_clip_strict": "Strict KG+CLIP",
}

METRIC_COLUMNS = {
    "Precision@5": "precision@5",
    "Recall@5": "recall@5",
    "ConstraintConsistency@5": "constraint_consistency@5",
    "mAP": "mAP",
    "AvgResponseTime(s)": "avg_response_time_seconds",
}

PURE_COLOR = "#8A99A8"
STRICT_COLOR = "#3B82B8"
GRID_COLOR = "#E7EBF0"
TEXT_COLOR = "#1F2933"
SUBTLE_TEXT_COLOR = "#52606D"


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    return pd.read_csv(path)


def build_metrics_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    summary_df = summary_df.copy()
    pure_row = summary_df.loc[summary_df["strategy"] == "pure_clip"].iloc[0]
    strict_row = summary_df.loc[summary_df["strategy"] == "kg_clip_strict"].iloc[0]

    rows: list[dict] = []
    for metric_label, metric_column in METRIC_COLUMNS.items():
        pure_value = float(pure_row[metric_column])
        strict_value = float(strict_row[metric_column])
        rows.append(
            {
                "metric": metric_label,
                "pure_clip": pure_value,
                "strict_kg_clip": strict_value,
                "strict_minus_pure": strict_value - pure_value,
            }
        )
    return pd.DataFrame(rows)


def clear_existing_figures(figure_dir: Path) -> None:
    for pattern in ("*.png", "*.svg", "*.pdf"):
        for path in figure_dir.glob(pattern):
            path.unlink()


def save_metric_bar_chart(summary_df: pd.DataFrame, metric_column: str, metric_label: str, output_path: Path) -> None:
    chart_df = summary_df.copy()
    chart_df["strategy_label"] = chart_df["strategy"].map(STRATEGY_LABELS)
    labels = chart_df["strategy_label"].tolist()
    values = chart_df[metric_column].astype(float).tolist()
    delta = values[1] - values[0]
    upper = max(values) * 1.22 + 0.01

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.1, 3.8))
    y_positions = np.arange(len(labels))
    bars = ax.barh(y_positions, values, color=[PURE_COLOR, STRICT_COLOR], height=0.52)

    ax.set_title(metric_label, loc="left", fontsize=16, fontweight="bold", color=TEXT_COLOR, pad=8)
    ax.text(
        1.0,
        1.02,
        f"Strict KG+CLIP +{delta:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color=SUBTLE_TEXT_COLOR,
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, upper)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(axis="x", color=GRID_COLOR, linestyle="--", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#CBD2D9")

    for label, color in zip(ax.get_yticklabels(), [PURE_COLOR, STRICT_COLOR]):
        label.set_color(color)
        label.set_fontsize(11)

    for bar, value in zip(bars, values):
        ax.text(
            min(value + upper * 0.02, upper * 0.97),
            bar.get_y() + bar.get_height() / 2.0,
            f"{value:.3f}",
            ha="left",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=TEXT_COLOR,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def extract_score_matrix(details_df: pd.DataFrame, top_k: int = 5) -> tuple[list[str], np.ndarray]:
    details_df = details_df.copy()
    details_df["query_id"] = details_df["query_id"].astype(str)
    details_df = details_df.sort_values("query_id").reset_index(drop=True)

    score_rows: list[list[float]] = []
    query_ids: list[str] = []
    for _, row in details_df.iterrows():
        top_hits = json.loads(row["top_hits"]) if isinstance(row["top_hits"], str) and row["top_hits"].strip() else []
        scores = [float(hit.get("score", math.nan)) for hit in top_hits[:top_k]]
        if len(scores) < top_k:
            scores.extend([math.nan] * (top_k - len(scores)))
        query_ids.append(str(row["query_id"]))
        score_rows.append(scores)

    return query_ids, np.array(score_rows, dtype=float)


def save_response_time_line_chart(pure_df: pd.DataFrame, strict_df: pd.DataFrame, output_path: Path) -> None:
    pure_times = pure_df[["query_id", "response_time_seconds"]].copy()
    strict_times = strict_df[["query_id", "response_time_seconds"]].copy()
    pure_times["query_id"] = pure_times["query_id"].astype(str)
    strict_times["query_id"] = strict_times["query_id"].astype(str)

    merged = pure_times.merge(strict_times, on="query_id", suffixes=("_pure", "_strict"))
    merged = merged.sort_values("query_id").reset_index(drop=True)

    x = np.arange(len(merged))
    pure_ms = merged["response_time_seconds_pure"].astype(float).to_numpy() * 1000.0
    strict_ms = merged["response_time_seconds_strict"].astype(float).to_numpy() * 1000.0
    query_labels = merged["query_id"].tolist()

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11.8, 4.9))
    ax.plot(x, pure_ms, marker="o", linewidth=2.0, markersize=4.8, color=PURE_COLOR)
    ax.plot(x, strict_ms, marker="s", linewidth=2.2, markersize=5.1, color=STRICT_COLOR)
    ax.axhline(pure_ms.mean(), linestyle="--", linewidth=1.1, color=PURE_COLOR, alpha=0.35)
    ax.axhline(strict_ms.mean(), linestyle="--", linewidth=1.1, color=STRICT_COLOR, alpha=0.35)

    ax.set_title("End-to-End Response Time by Query", loc="left", fontsize=16, fontweight="bold", color=TEXT_COLOR)
    ax.text(
        1.0,
        1.02,
        f"Average delta: +{(strict_ms.mean() - pure_ms.mean()):.1f} ms",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color=SUBTLE_TEXT_COLOR,
    )
    ax.set_ylabel("Response Time (ms)")
    ax.set_xlabel("Query ID")
    ax.set_xticks(x)
    ax.set_xticklabels(query_labels, rotation=45, ha="right")
    ax.grid(axis="y", color=GRID_COLOR, linestyle="--", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD2D9")
    ax.spines["bottom"].set_color("#CBD2D9")
    ax.set_ylim(0.0, max(strict_ms.max(), pure_ms.max()) * 1.12)
    ax.set_xlim(-0.6, len(x) + 1.25)
    ax.text(len(x) + 0.35, pure_ms[-1], f"Pure CLIP\n{pure_ms.mean():.1f} ms avg", color=PURE_COLOR, fontsize=10, va="center")
    ax.text(len(x) + 0.35, strict_ms[-1], f"Strict KG+CLIP\n{strict_ms.mean():.1f} ms avg", color=STRICT_COLOR, fontsize=10, va="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_similarity_heatmap(pure_df: pd.DataFrame, strict_df: pd.DataFrame, output_path: Path, top_k: int = 5) -> None:
    pure_query_ids, pure_scores = extract_score_matrix(pure_df, top_k=top_k)
    strict_query_ids, strict_scores = extract_score_matrix(strict_df, top_k=top_k)

    if pure_query_ids != strict_query_ids:
        raise ValueError("Pure CLIP and Strict KG+CLIP query order is inconsistent.")

    combined_scores = np.concatenate([pure_scores.flatten(), strict_scores.flatten()])
    valid_scores = combined_scores[~np.isnan(combined_scores)]
    vmin = float(valid_scores.min()) if valid_scores.size else 0.0
    vmax = float(valid_scores.max()) if valid_scores.size else 1.0

    delta_scores = strict_scores - pure_scores
    valid_delta_scores = delta_scores[~np.isnan(delta_scores)]
    delta_abs_max = float(np.nanmax(np.abs(valid_delta_scores))) if valid_delta_scores.size else 1.0

    plt.style.use("seaborn-v0_8-white")
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.8, 8.2),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.05]},
        constrained_layout=True,
    )
    score_tables = [
        ("Pure CLIP", pure_scores, "YlGnBu", vmin, vmax),
        ("Strict KG+CLIP", strict_scores, "YlOrRd", vmin, vmax),
        ("Strict - Pure", delta_scores, "RdBu_r", -delta_abs_max, delta_abs_max),
    ]

    for ax, (title, scores, cmap, local_vmin, local_vmax) in zip(axes, score_tables):
        masked_scores = np.ma.masked_invalid(scores)
        if title == "Strict - Pure":
            heatmap = ax.imshow(
                masked_scores,
                cmap=cmap,
                aspect="auto",
                norm=TwoSlopeNorm(vmin=local_vmin, vcenter=0.0, vmax=local_vmax),
            )
        else:
            heatmap = ax.imshow(masked_scores, cmap=cmap, aspect="auto", vmin=local_vmin, vmax=local_vmax)

        ax.set_title(f"{title}\nTop-{top_k} Similarity", fontsize=12, fontweight="bold", color=TEXT_COLOR)
        ax.set_xticks(range(top_k))
        ax.set_xticklabels([f"Rank {idx}" for idx in range(1, top_k + 1)])
        ax.set_yticks(range(len(pure_query_ids)))
        ax.set_yticklabels(pure_query_ids, fontsize=8)
        ax.set_xlabel("Returned hit rank")
        if ax is axes[0]:
            ax.set_ylabel("Query ID")
        ax.set_xticks(np.arange(-0.5, top_k, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(pure_query_ids), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.4)
        ax.tick_params(which="minor", bottom=False, left=False)

    colorbar_left = fig.colorbar(axes[1].images[0], ax=axes[:2], shrink=0.92)
    colorbar_left.set_label("Frame similarity score")
    colorbar_right = fig.colorbar(axes[2].images[0], ax=[axes[2]], shrink=0.92)
    colorbar_right.set_label("Score delta")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_report_outline(summary_df: pd.DataFrame, output_path: Path, figure_dir: Path) -> None:
    pure_row = summary_df.loc[summary_df["strategy"] == "pure_clip"].iloc[0]
    strict_row = summary_df.loc[summary_df["strategy"] == "kg_clip_strict"].iloc[0]

    outline = f"""# 主实验分析报告提纲

## 1. 实验边界与任务定义
- 检索模态仅包含 text2image 与 text2video。
- Milvus 存储的是帧级实体，视频结果来自命中帧后的应用层派生，不直接检索视频向量。
- Neo4j 当前活库为 scene 级知识图谱，核心节点与关系围绕 Scene / Weather / TimeOfDay / Location / Object。
- 本轮主实验采用 seed-scene scene-level truth：每条 query 的真值 scene 为其原始编写时对应的 seed_scene_token。

## 2. 对比方法
- Pure CLIP：不做 KG 过滤，直接对全量帧向量做文本相似度检索，再映射为 scene 级结果。
- Strict KG+CLIP：先基于结构化条件在 scene 级 KG 中筛选候选 scene，再在候选 scene 对应的帧级向量中做相似度检索。

## 3. 评测指标与响应时间协议
- 指标：Precision@5、Recall@5、ConstraintConsistency@5、mAP、响应时间。
- 响应时间协议：end-to-end，包含 query parsing；每条 query 先 warmup 1 次，再正式运行 5 次，取中位数，最后对 20 条 query 做宏平均。

## 4. 总体结果
- Precision@5：Pure CLIP = {float(pure_row['precision@5']):.3f}，Strict KG+CLIP = {float(strict_row['precision@5']):.3f}。
- Recall@5：Pure CLIP = {float(pure_row['recall@5']):.3f}，Strict KG+CLIP = {float(strict_row['recall@5']):.3f}。
- ConstraintConsistency@5：Pure CLIP = {float(pure_row['constraint_consistency@5']):.3f}，Strict KG+CLIP = {float(strict_row['constraint_consistency@5']):.3f}。
- mAP：Pure CLIP = {float(pure_row['mAP']):.3f}，Strict KG+CLIP = {float(strict_row['mAP']):.3f}。
- 平均响应时间：Pure CLIP = {float(pure_row['avg_response_time_seconds']):.4f}s，Strict KG+CLIP = {float(strict_row['avg_response_time_seconds']):.4f}s。

## 5. 图表解读建议
- Precision@5 与 Recall@5 柱状图：突出 Strict KG+CLIP 在 scene-level 命中上的增益。
- ConstraintConsistency@5 与 mAP 柱状图：说明 KG 过滤在结构化条件满足度和整体排序质量上的优势。
- 响应时间折线图：展示 20 条 query 的端到端耗时变化，并报告平均绝对延迟差值。
- 相似度热力图：比较两种方法 top-5 返回帧的相似度分布差异，并补充 Strict KG+CLIP 相对 Pure CLIP 的分数增量视图。

## 6. 结果分析要点
- 从检索效果看，Strict KG+CLIP 在 Precision@5、Recall@5、ConstraintConsistency@5 与 mAP 上整体优于 Pure CLIP。
- 从约束一致性看，KG 过滤显著降低了与天气、时段、地点、目标类别不一致的返回结果比例。
- 从效率看，Strict KG+CLIP 的绝对延迟高于 Pure CLIP，但仍维持在亚秒级。
- 由于当前真值定义为 seed-scene anchor，该评测更适合作为“目标 scene 命中能力”分析，而非“所有语义相关 scene 的完整覆盖”分析。

## 7. 局限性与返工预案
- 若后续需要更完整的 scene-level 相关性定义，可回到 manual_query_truth_candidates.csv 做候选 scene 的人工审核。
- 当前实验不会把“语义合理但非 seed scene”的结果视为命中，因此对开放语义 query 的评价偏保守。
- 后续可补充分组分析，例如中文/英文、three_plus/two_condition/open_semantic 等维度。

## 8. 论文插图与表格引用建议
- 指标汇总表：{(output_path.parent / 'metrics_summary_for_paper.csv').as_posix()}
- Precision@5 图：{(figure_dir / 'precision_at_5_bar.png').as_posix()}
- Recall@5 图：{(figure_dir / 'recall_at_5_bar.png').as_posix()}
- ConstraintConsistency@5 图：{(figure_dir / 'constraint_consistency_at_5_bar.png').as_posix()}
- mAP 图：{(figure_dir / 'map_bar.png').as_posix()}
- 响应时间折线图：{(figure_dir / 'response_time_line.png').as_posix()}
- 相似度热力图：{(figure_dir / 'similarity_heatmap_top5.png').as_posix()}
"""
    output_path.write_text(outline, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build figures and report assets for manual query comparison results.")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    clear_existing_figures(figure_dir)

    summary_df = load_table(run_dir / "evaluation_summary.csv")
    pure_details_df = load_table(run_dir / "pure_clip_details.csv")
    strict_details_df = load_table(run_dir / "kg_clip_strict_details.csv")

    metrics_summary_df = build_metrics_summary(summary_df)
    metrics_summary_df.to_csv(run_dir / "metrics_summary_for_paper.csv", index=False, encoding="utf-8-sig")

    save_metric_bar_chart(summary_df, "precision@5", "Precision@5", figure_dir / "precision_at_5_bar.png")
    save_metric_bar_chart(summary_df, "recall@5", "Recall@5", figure_dir / "recall_at_5_bar.png")
    save_metric_bar_chart(
        summary_df,
        "constraint_consistency@5",
        "ConstraintConsistency@5",
        figure_dir / "constraint_consistency_at_5_bar.png",
    )
    save_metric_bar_chart(summary_df, "mAP", "mAP", figure_dir / "map_bar.png")
    save_response_time_line_chart(pure_details_df, strict_details_df, figure_dir / "response_time_line.png")
    save_similarity_heatmap(pure_details_df, strict_details_df, figure_dir / "similarity_heatmap_top5.png", top_k=5)
    build_report_outline(summary_df, run_dir / "analysis_report_outline.md", figure_dir)

    print(f"Saved report assets to {run_dir}")


if __name__ == "__main__":
    main()
