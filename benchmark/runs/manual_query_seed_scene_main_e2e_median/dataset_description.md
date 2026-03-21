# 测试数据集说明

## 1. 数据来源
- 主实验查询集来自用户人工编写的 20 条 query。
- 原始整理文件为 `D:\KG_Scene_Retrieval\benchmark\manual_query_benchmark.csv`。
- 评测输入文件为 `D:\KG_Scene_Retrieval\benchmark\manual_query_benchmark_seed_scene_eval.csv`。

## 2. 实验边界
- 检索模态仅包含 `text2image` 与 `text2video`。
- Milvus 存储的是帧级实体，不直接存储视频实体。
- 视频结果来自命中帧后的应用层派生输出，不直接检索视频向量。
- Neo4j 当前活库为 scene 级知识图谱，核心节点与关系围绕 `Scene / Weather / TimeOfDay / Location / Object`。

## 3. 查询集规模与构成
- query 总数：20
- 中文 query：14
- 英文 query：6
- `three_plus`：16
- `two_condition`：2
- `open_semantic`：2

## 4. 真值定义
- 本轮主实验采用 `seed-scene scene-level truth`。
- 每条 query 的真值 scene 为该 query 编写时对应的 `seed_scene_token`。
- 若返回结果属于同一 `scene_token`，则记为命中。
- 该定义适用于 scene-level 命中能力评估，但不会把“语义合理但非 seed scene”的结果记为命中。

## 5. 评测对象
- `text2image`：以返回帧映射到所属 scene 后进行评测。
- `text2video`：以命中帧所属 scene 派生的视频结果进行评测，本质上仍以 scene-level 命中为准。

## 6. 评测输入字段
- `query_id`：主实验 query 编号，范围 `M001-M020`
- `query_text`：原始查询文本，不改写原句
- `scene_token`：对应 seed scene
- `relevant_scene_tokens`：评测真值集合，当前为单元素列表
- `truth_definition`：固定为 `seed_scene_anchor`
- `truth_status`：当前为 `ready`

## 7. 使用说明
- 本数据集适合作为本科毕设主实验中的固定 benchmark。
- 若后续需要更完整的 scene-level 相关性定义，可回到 `manual_query_truth_candidates.csv` 做候选 scene 的人工审核扩充真值。
