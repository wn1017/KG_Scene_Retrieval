# 实验参数设置

## 1. 运行环境
- Python 环境：`conda run -n kg python ...`
- 项目目录：`D:\KG_Scene_Retrieval`
- Milvus：`127.0.0.1:19530`
- Neo4j：`127.0.0.1:7687`

## 2. 对比方法
- `Pure CLIP`
  - 不做 KG 过滤
  - 直接对全量帧向量进行文本相似度检索
  - 返回帧再映射为 scene 级结果
- `Strict KG+CLIP`
  - 先在 scene 级 KG 中按结构化条件筛选候选 scene
  - 再在候选 scene 对应的帧向量中进行文本相似度检索
  - 若严格 KG 返回 0 个候选，则停止，不走相似度回退

## 3. 评测指标
- `Precision@5`
- `Recall@5`
- `ConstraintConsistency@5`
- `mAP`
- `Response Time`

## 4. 检索与评测参数
- `Top-K`：5
- `map_depth`：20
- `frame_search_limit`：80
- `timing_warmup`：1
- `timing_runs`：5

## 5. 响应时间协议
- 计时口径为 `end-to-end`
- 计时包含 `query parsing`
- 每条 query 先预热 1 次
- 再正式运行 5 次
- 对同一 query 取中位数作为 `response_time_seconds`
- 最后对 20 条 query 做宏平均

## 6. 评测输入与输出
- 输入文件：`D:\KG_Scene_Retrieval\benchmark\manual_query_benchmark_seed_scene_eval.csv`
- 输出目录：`D:\KG_Scene_Retrieval\benchmark\runs\manual_query_seed_scene_main_e2e_median`
- 原始汇总：`evaluation_summary.csv`
- 明细结果：
  - `pure_clip_details.csv`
  - `kg_clip_strict_details.csv`
- 分组汇总：
  - `pure_clip_group_summary.csv`
  - `kg_clip_strict_group_summary.csv`

## 7. 实际执行命令
```bash
conda run -n kg python D:\KG_Scene_Retrieval\scripts\evaluate.py D:\KG_Scene_Retrieval\benchmark\manual_query_benchmark_seed_scene_eval.csv --output-dir D:\KG_Scene_Retrieval\benchmark\runs\manual_query_seed_scene_main_e2e_median --strategies pure_clip kg_clip_strict
```

## 8. 图表生成命令
```bash
conda run -n kg python D:\KG_Scene_Retrieval\scripts\build_manual_query_report_assets.py --run-dir D:\KG_Scene_Retrieval\benchmark\runs\manual_query_seed_scene_main_e2e_median
```
