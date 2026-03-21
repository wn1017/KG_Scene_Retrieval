# 主实验分析报告提纲

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
- Precision@5：Pure CLIP = 0.080，Strict KG+CLIP = 0.110。
- Recall@5：Pure CLIP = 0.400，Strict KG+CLIP = 0.550。
- ConstraintConsistency@5：Pure CLIP = 0.320，Strict KG+CLIP = 0.755。
- mAP：Pure CLIP = 0.289，Strict KG+CLIP = 0.490。
- 平均响应时间：Pure CLIP = 0.0164s，Strict KG+CLIP = 0.1156s。

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
- 指标汇总表：D:/KG_Scene_Retrieval/benchmark/runs/manual_query_seed_scene_main_e2e_median/metrics_summary_for_paper.csv
- Precision@5 图：D:/KG_Scene_Retrieval/benchmark/runs/manual_query_seed_scene_main_e2e_median/figures/precision_at_5_bar.png
- Recall@5 图：D:/KG_Scene_Retrieval/benchmark/runs/manual_query_seed_scene_main_e2e_median/figures/recall_at_5_bar.png
- ConstraintConsistency@5 图：D:/KG_Scene_Retrieval/benchmark/runs/manual_query_seed_scene_main_e2e_median/figures/constraint_consistency_at_5_bar.png
- mAP 图：D:/KG_Scene_Retrieval/benchmark/runs/manual_query_seed_scene_main_e2e_median/figures/map_bar.png
- 响应时间折线图：D:/KG_Scene_Retrieval/benchmark/runs/manual_query_seed_scene_main_e2e_median/figures/response_time_line.png
- 相似度热力图：D:/KG_Scene_Retrieval/benchmark/runs/manual_query_seed_scene_main_e2e_median/figures/similarity_heatmap_top5.png
