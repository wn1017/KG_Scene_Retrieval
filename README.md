# KG Scene Retrieval

知识图谱与自然语言驱动的跨模态驾驶场景检索系统。

当前项目的最终检索模态只有两种：

- `text2image`
- `text2video`

需要先统一 3 个核心事实：

- Milvus 当前存储的是帧级实体，不是视频级实体。
- `text2video` 不是直接检索视频，而是先检索命中帧，再在同一场景、同一相机序列中收集连续帧并动态生成视频片段。
- Neo4j 负责 scene 级结构化条件过滤，Milvus 负责帧级语义相似度检索。

## Current Runtime Scope

仓库里的历史说明有一部分仍然以 `nuScenes v1.0-mini` 为讲解示例，但当前实际运行配置应以 [config.py](/D:/KG_Scene_Retrieval/config.py) 为准。

当前默认配置的重点是：

- 数据版本：`v1.0-trainval`
- 当前实验子集：`trainval_camera_part06_part10`
- Milvus collection：`multimodal_search_trainval_camera_part06_part10`
- 主要相机：`CAM_FRONT`

因此，这个仓库现在更准确地说是：

```text
面向 nuScenes trainval 相机子集的 scene-level KG filtering + frame-level CLIP retrieval system
```

## What The System Does

用户输入一句中文或英文场景描述后，系统会：

1. 用 [src/nlp_parser.py](/D:/KG_Scene_Retrieval/src/nlp_parser.py) 解析天气、时段、地点、对象等结构化条件。
2. 用 [src/kg_builder.py](/D:/KG_Scene_Retrieval/src/kg_builder.py) 和 Neo4j 先筛选候选 `scene`。
3. 根据语言路由到 English-CLIP 或 Chinese-CLIP。
4. 用 Milvus 在候选 scene 对应的帧向量中检索相似帧。
5. 补全命中帧的元数据，并返回图片结果或生成视频片段。

主链路可概括为：

```text
query
-> parse_query
-> get_candidate_scene_tokens
-> encode_text_query
-> search_frame_hits
-> enrich_hit_record
-> render images or derive video clips
```

## Architecture

### 1. Natural Language Parsing

[src/nlp_parser.py](/D:/KG_Scene_Retrieval/src/nlp_parser.py) 负责：

- 识别中文/英文
- 提取 `weather`、`time`、`location`、`objects`
- 生成查询高亮信息
- 决定使用 `engclip` 还是 `chnclip`

### 2. Scene Knowledge Graph

[src/kg_builder.py](/D:/KG_Scene_Retrieval/src/kg_builder.py) 负责：

- 从 nuScenes 元数据构建 scene 级结构化记录
- 推断天气、时段、地点类型
- 统计 scene 中出现的对象
- 将 scene / weather / time / location / object 写入 Neo4j
- 根据结构化条件查询候选 `scene_token`

### 3. Frame Retrieval

[src/milvus_utils.py](/D:/KG_Scene_Retrieval/src/milvus_utils.py) 和 [scripts/insert_image.py](/D:/KG_Scene_Retrieval/scripts/insert_image.py) 负责：

- 定义 Milvus schema
- 写入图像帧 embedding
- 同步写入帧级业务元数据

当前 Milvus 中一条记录除了向量外，还包含：

- `scene_token`
- `sample_token`
- `camera`
- `frame_path`
- `weather`
- `timeofday`
- `location`
- `obj_types`

### 4. Online Orchestration and UI

[app.py](/D:/KG_Scene_Retrieval/app.py) 负责：

- 加载模型与集合
- 编排 KG 过滤、向量检索和结果补全
- 为 `text2image` 和 `text2video` 生成最终结果
- 构建 Gradio 界面

## Repository Layout

```text
KG_Scene_Retrieval/
├─ app.py
├─ config.py
├─ README.md
├─ requirements.txt
├─ start.bat
├─ docker-compose.services.yml
├─ embedEtcd.yaml
├─ user.yaml
├─ src/
│  ├─ kg_builder.py
│  ├─ milvus_utils.py
│  ├─ nlp_parser.py
│  ├─ nuscenes_metadata.py
│  └─ trainval_subset.py
├─ scripts/
│  ├─ prepare_trainval06_subset.py
│  ├─ insert_image.py
│  ├─ kg_builder.py
│  ├─ evaluate.py
│  ├─ build_manual_query_benchmark.py
│  └─ build_manual_query_report_assets.py
├─ tests/
├─ docs/
├─ benchmark/
├─ derived_data/
├─ generated_videos/
├─ csvdata/
└─ models/
```

## Environment

项目默认使用 Conda 环境 `kg`。

```bash
conda run -n kg python ...
conda run -n kg pip install -r requirements.txt
```

不要使用系统默认 `python` 或 `pip`。

## Configuration

所有可配置路径、端口和服务地址统一放在 [config.py](/D:/KG_Scene_Retrieval/config.py)。

当前关键配置包括：

- 数据集根目录与 metadata 目录
- trainval 子集名称与派生目录
- CLIP 模型目录
- Milvus / Neo4j 主机与端口
- Gradio 端口
- 视频生成参数

如果迁移到新机器，优先修改 `config.py`。

## Data Preparation

### 1. Prepare the trainval subset metadata

```bash
conda run -n kg python scripts\prepare_trainval06_subset.py
```

这一步会根据当前配置生成：

- 子集 metadata JSON
- 场景 token 列表
- 关键帧 CSV
- subset report

### 2. Build the Neo4j scene graph

```bash
conda run -n kg python scripts\kg_builder.py --write
```

### 3. Insert frame embeddings into Milvus

```bash
conda run -n kg python scripts\insert_image.py --drop-existing
```

只有在需要重建 collection 或 schema 时才加 `--drop-existing`。

## Run The App

Windows 下推荐直接运行：

```bat
start.bat
```

也可以先做自检：

```bat
start.bat --check
```

默认访问地址：

- Web UI: `http://127.0.0.1:7860`
- Attu: `http://127.0.0.1:8000`
- Neo4j Browser: `http://127.0.0.1:7474`

## Retrieval Modes

### `text2image`

- 返回高相关帧图像
- 展示分数、场景信息、相机、天气、时段、地点和对象元数据

### `text2video`

- 先检索锚点帧
- 再从同一 scene、同一 camera 的连续帧中抽取片段
- 最终返回动态生成的视频文件

注意：

- 返回的视频片段必须只来自单个完整 scene 的单个相机流
- 不会将多个 scene 混合成一个视频结果

## Evaluation

[scripts/evaluate.py](/D:/KG_Scene_Retrieval/scripts/evaluate.py) 用于离线评估。

常见策略包括：

- `pure_clip`
- `kg_clip_strict`
- `kg_clip_engineering`

需要区分两种口径：

- `kg_clip_strict` 更适合论文主实验，会严格执行 KG 条件，不满足时可以直接记为零候选。
- `kg_clip_engineering` 更接近当前 UI 工程行为，会在严格过滤后按规则做放宽或全库相似度回退。

示例：

```bash
conda run -n kg python scripts\evaluate.py benchmark\manual_query_benchmark_seed_scene_eval.csv --output-dir benchmark\runs\manual_query_seed_scene_main_e2e_median --strategies pure_clip kg_clip_strict
```

## Docs

文档分两层：

- [docs/项目解释-简版.md](/D:/KG_Scene_Retrieval/docs/项目解释-简版.md)：适合快速建立整体认识
- [docs/项目解释-详尽版.md](/D:/KG_Scene_Retrieval/docs/项目解释-详尽版.md)：适合系统性阅读源码和实验链路

为了避免历史信息混淆，这两份文档都以当前 trainval 子集配置为主，同时会在必要处说明 `v1.0-mini` 仅用于概念举例。

## Publishing To GitHub

如果你要把这个仓库当作自己的 GitHub 项目，建议上传“源码、配置、文档、轻量评估定义”，不要上传“本地大资源、数据集、模型、运行产物”。

建议上传：

- `app.py`
- `config.py`
- `src/`
- `scripts/`
- `tests/`
- `docs/`
- `benchmark/*.csv`
- `README.md`
- `requirements.txt`
- `start.bat`
- `docker-compose.services.yml`
- `embedEtcd.yaml`
- `user.yaml`
- `.gitignore`
- `AGENTS.md`

建议不要上传：

- `models/`
- `csvdata/`
- `generated_videos/`
- `derived_data/`
- `.gradio/`
- `__pycache__/`
- `*.log`
- 本地 nuScenes 数据集目录
- 大型 benchmark 结果图、临时导出和缓存文件

如果你希望仓库更适合公开发布，建议在上传前检查 [config.py](/D:/KG_Scene_Retrieval/config.py) 中的本地绝对路径，并按需要改成更通用的示例路径或在 README 中标注这些路径需要用户自行修改。

## Notes

- Neo4j 是 scene-level filtering，不是向量检索引擎。
- Milvus 是 frame-level similarity search，不是视频数据库。
- `text2video` 是帧检索后的应用层派生输出。
- 当前仓库既服务于在线演示，也服务于 benchmark 与论文实验。

## License

本仓库用于毕业设计与教学实验场景。若需公开发布或商用，请根据数据集、模型和第三方组件许可证补充对应说明。
