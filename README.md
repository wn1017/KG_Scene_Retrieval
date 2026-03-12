# KG Scene Retrieval

知识图谱与自然语言驱动的跨模态驾驶场景检索系统。

本项目以 `nuScenes v1.0-mini` 为实验数据，结合 `Neo4j`、`Milvus`、`English-CLIP` 和 `Chinese-CLIP`，实现面向驾驶场景的自然语言检索。当前系统的最终检索模式为：

- 文搜图：返回高相关关键帧图像
- 文搜视频片段：先检索高相关帧，再按场景与相机序列拼接短视频片段

注意：Milvus 中当前存储的是帧级实体，不是独立视频向量；视频结果属于应用层派生输出。

## Features

- 支持中文和英文自然语言查询，自动切换 `Chinese-CLIP` / `English-CLIP`
- 基于 `Neo4j` 的知识图谱候选过滤与降级策略
- 基于 `Milvus` 的帧级向量检索
- 支持文搜图与文搜视频片段两种展示模式
- 支持对比评估 `Pure CLIP` 与 `KG + CLIP`
- 使用 `Gradio` 构建本地演示界面

## Tech Stack

- 前端：`Gradio`
- 向量模型：`openai/clip-vit-base-patch16`、`OFA-Sys/chinese-clip-vit-base-patch16`
- 向量数据库：`Milvus`
- 知识图谱：`Neo4j`
- 数据集：`nuScenes v1.0-mini`
- 主要语言：`Python`

## Project Structure

```text
KG_Scene_Retrieval/
├─ app.py                         # Gradio 应用入口
├─ config.py                      # 全局配置入口
├─ start.bat                      # Windows 一键启动脚本
├─ README.md                      # GitHub 项目说明
├─ requirements.txt               # Python 依赖
├─ docker-compose.services.yml    # 本地服务编排
├─ embedEtcd.yaml                 # Milvus 配置文件
├─ user.yaml                      # Milvus 配置文件
├─ scripts/                       # 离线脚本与命令行入口
│  ├─ evaluate.py
│  ├─ insert_image.py
│  ├─ insert_text.py
│  └─ kg_builder.py
├─ src/                           # 内部业务模块
│  ├─ milvus_utils.py
│  ├─ kg_builder.py
│  ├─ nlp_parser.py
│  └─ nuscenes_metadata.py
├─ tests/                         # 单元测试
├─ csvdata/                       # 本地 CSV 数据（已在 .gitignore 中忽略）
├─ models/                        # 本地 CLIP 模型（已在 .gitignore 中忽略）
└─ generated_videos/              # 动态生成的视频片段目录
```

## Environment

本项目默认使用 Conda 环境 `kg`。

```bash
conda run -n kg python ...
conda run -n kg pip install -r requirements.txt
```

项目中的本地路径、端口和服务配置统一放在 `config.py` 中。迁移到其他机器时，优先修改该文件。

## Data and Models

默认配置如下：

- `nuScenes` 根目录：`D:\nuScenes_v1.0-mini`
- 英文模型目录：`models/engclip`
- 中文模型目录：`models/chnclip`
- Milvus collection：`multimodal_search`

仓库默认不提交以下本地资源：

- `models/`
- `csvdata/`
- `generated_videos/`

## Quick Start

### 1. 准备本地依赖

确保以下服务或依赖可用：

- Conda 环境 `kg`
- `Docker Desktop`
- `Milvus`
- `Neo4j`
- 本地 `engclip` 和 `chnclip` 模型目录
- 本地 `nuScenes v1.0-mini` 数据

### 2. 检查配置

根据你的机器环境修改 `config.py` 中的关键项，例如：

- 数据集路径
- 模型路径
- `Milvus` / `Neo4j` 端口
- 应用端口

### 3. 启动项目

Windows 下推荐直接双击：

```bat
start.bat
```

也可以先做自检：

```bat
start.bat --check
```

默认访问地址：

- Web UI：`http://127.0.0.1:7860`
- Attu：`http://127.0.0.1:8000`
- Neo4j Browser：`http://127.0.0.1:7474`

说明：首次启动时需要加载模型与元数据，`7860` 可能在几十秒后才可访问；如果第一次打开失败，稍等后刷新即可。

## Retrieval Workflow

### 文搜图

1. 用户输入中文或英文驾驶场景描述
2. `src/nlp_parser.py` 提取天气、时间、目标物、位置等结构化字段
3. `src/kg_builder.py` / `Neo4j` 过滤候选场景
4. CLIP 编码查询文本
5. `Milvus` 返回相关关键帧
6. 前端展示图像与结构化元数据

### 文搜视频片段

1. 先完成同样的候选过滤与帧检索
2. 选取高分命中帧作为锚点
3. 从 `nuScenes samples/sweeps` 中收集邻近连续帧
4. 动态拼接为短视频片段
5. 前端展示可播放片段与场景说明

## Knowledge Graph and Evaluation

### 知识图谱构建

```bash
conda run -n kg python scripts\kg_builder.py
```

### 评估脚本

```bash
conda run -n kg python scripts\evaluate.py path\to\eval.csv --output-dir eval_outputs
```

当前评估关注指标包括：

- `Precision@5`
- `Recall@5`
- `mAP`
- 平均响应时间

## Notes

- 当前视频检索是“检索帧，再拼接片段”，不是片段级向量库检索。
- `scripts/insert_text.py` 目前主要用于文本样本解析与实验辅助，不再向主检索 collection 写入文本实体。
- 如果 `start.bat` 启动后网页短时间不可访问，优先等待模型加载完成，再刷新 `7860` 页面。

## License

本仓库用于毕业设计与教学实验场景，若需对外发布，请根据数据集、模型与第三方组件的许可证要求进一步补充说明。
