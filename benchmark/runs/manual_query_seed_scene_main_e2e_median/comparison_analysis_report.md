# Pure CLIP vs Strict KG+CLIP 对比实验分析报告

## 1. 实验目的
本实验围绕固定的 20 条人工 query，比较 `Pure CLIP` 与 `Strict KG+CLIP` 两种检索策略在 scene-level 命中任务上的表现。实验边界保持不变：最终检索模态仅包含 `text2image` 和 `text2video`；Milvus 存储的是帧级实体；视频结果由命中帧在应用层派生；Neo4j 当前使用的是 scene 级知识图谱。

## 2. 数据集与真值口径
主实验 benchmark 共 20 条 query，其中中文 14 条、英文 6 条；按查询复杂度可分为 `three_plus=16`、`two_condition=2`、`open_semantic=2`。本轮采用 `seed-scene scene-level truth`，即每条 query 的真值 scene 设为该 query 编写时对应的 `seed_scene_token`。因此，评测目标是“是否命中目标 scene”，而不是“是否覆盖所有可能语义相关的 scene”。

## 3. 参数设置
本次评测采用 `Top-K=5`、`map_depth=20`、`frame_search_limit=80`。响应时间按端到端口径统计，包含 query parsing；每条 query 先 warmup 1 次，再正式运行 5 次，取中位数，最终对 20 条 query 做宏平均。

## 4. 总体结果

| 方法 | Precision@5 | Recall@5 | ConstraintConsistency@5 | mAP | Avg Response Time (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pure CLIP | 0.080 | 0.400 | 0.320 | 0.289 | 0.0164 |
| Strict KG+CLIP | 0.110 | 0.550 | 0.755 | 0.490 | 0.1156 |

从总体结果看，`Strict KG+CLIP` 在 4 个效果指标上均优于 `Pure CLIP`。其中，`Precision@5` 提升 `0.030`，`Recall@5` 提升 `0.150`，`ConstraintConsistency@5` 提升 `0.435`，`mAP` 提升 `0.202`。这说明在 scene 级结构化约束参与后，系统更容易把检索范围收缩到与天气、时段、地点和目标类别更一致的候选 scene 中，从而改善最终排序质量。

## 5. 效果分析
`Precision@5` 和 `Recall@5` 的提升说明，`Strict KG+CLIP` 不仅能更稳定地把目标 scene 提前，还能在前 5 个返回结果中更高概率覆盖 seed scene。相比之下，`Pure CLIP` 在开放相似度检索中更容易受到视觉近邻 scene 的干扰，导致检索结果虽然“看起来相似”，但未必属于正确 scene。

`ConstraintConsistency@5` 的提升最为明显，从 `0.320` 提高到 `0.755`。这说明 KG 过滤的主要价值并不只是“缩小候选集合”，更关键的是显著降低了与查询结构化条件不一致的返回结果比例。对于包含多个约束条件的查询，这一优势尤其重要。

`mAP` 从 `0.289` 提高到 `0.490`，表明 `Strict KG+CLIP` 在整体排序顺序上更优，不仅仅是“Top-1 或 Top-5 命中一次”，而是更系统地把相关 scene 排在更前的位置。

## 6. 响应时间分析
响应时间方面，`Strict KG+CLIP` 的平均延迟为 `0.1156s`，高于 `Pure CLIP` 的 `0.0164s`，平均绝对差值约为 `99.2 ms`。这部分开销主要来自 scene 级 KG 条件解析与候选筛选。尽管如此，当前两种方法都仍处于亚秒级响应范围内，说明该开销在主实验规模下是可以接受的。

## 7. 相似度热力图解读
热力图横轴表示返回结果中的 `Top-5` 排名位置，纵轴表示 20 条 query。左图对应 `Pure CLIP`，中图对应 `Strict KG+CLIP`，颜色越暖表示该排名位置的帧相似度越高。右图显示两者差值：红色表示 `Strict KG+CLIP` 在该 query / rank 上的相似度更高，蓝色表示更低，接近白色表示差异较小。

阅读热力图时，优先看两个层面：
- 看中图相对左图是否在更多 query 上呈现更连续的高分带，这反映 `Strict KG+CLIP` 的前列结果是否更稳定。
- 看右图在前几列是否更多接近红色或白色，而不是深蓝色，这反映引入 KG 后是否没有明显破坏相似度排序。

需要注意的是，热力图展示的是“返回帧的相似度分布”，而不是直接展示 `Precision@5` 或 `Recall@5`。某些 query 即使相似度很高，只要返回 scene 不等于 seed scene，仍然不会被计为命中。因此，热力图更适合辅助解释“排序特征变化”，而不是单独作为效果优劣的最终结论。

## 8. 结论
在当前 20 条主实验 query 和 `seed-scene scene-level truth` 的评测口径下，`Strict KG+CLIP` 相比 `Pure CLIP` 取得了更好的 scene-level 检索效果，尤其在 `Recall@5`、`ConstraintConsistency@5` 和 `mAP` 上优势明显。其代价是平均增加约 `99.2 ms` 的端到端响应时间，但整体仍维持在可接受的亚秒级范围内。综合来看，`Strict KG+CLIP` 更适合作为本项目主实验中的核心检索方案。

## 9. 局限性
当前实验将每条 query 的真值限定为单个 `seed_scene_token`，因此不会把“语义上合理但不是 seed scene”的结果记为命中。这一口径有利于稳定开展主实验，但对开放语义 query 的评价偏保守。若后续需要扩展为“多相关 scene”评测，可回到 `manual_query_truth_candidates.csv` 做 scene-level 人工审核，以补充更完整的真值集合。
