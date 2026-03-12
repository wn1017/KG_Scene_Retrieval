# AGENTS.md

## General Rules

1. In every reply, address the user as "彦祖".
2. Before writing any code, describe the implementation plan and wait for approval. If requirements are unclear, ask clarifying questions before writing code.
3. If a task requires changes to more than 3 files, pause first and break it into smaller subtasks.
4. When a bug is found, first write a test that reproduces the bug, then fix it until the test passes.
5. Each time the user corrects you, add a new rule to this file to avoid the same issue.
6. Prefer writing AGENTS.md rules in English to avoid encoding or garbled text issues.
7. Before implementation, restate the final retrieval modalities explicitly and do not assume legacy modes still apply.
8. Distinguish stored entity types in Milvus from application-level derived outputs such as generated video clips.
9. When the user asks for example content outside the UI, provide it directly in the response and do not add it to the webpage by default.
10. After UI changes, verify the live page is served by the new process before claiming the update is visible.
11. For Windows batch launchers, validate the observable double-click behavior (or an equivalent detached launch), not only interactive terminal execution.
12. When the user asks to organize the repository, perform structural reorganization and documentation updates, not cleanup-only.

## Python Environment

- This project uses the conda environment `kg`.
- Python path: `D:\miniconda\envs\kg\python.exe`
- All Python commands must be executed using the following format:

```bash
conda run -n kg python ...
conda run -n kg pip install ...
```

- Do NOT use the system default `python` or `pip` unless explicitly instructed.

## Dataset

- nuScenes root: `D:\nuScenes_v1.0-mini\`
- Version: `v1.0-mini`
- Subdirectories:
  - `v1.0-mini\` - JSON metadata (`scene.json`, `sample.json`, etc.)
  - `samples\` - keyframe images (`CAM_FRONT`, `CAM_BACK`, etc.)
  - `sweeps\` - continuous frames for video retrieval
  - `maps\` - HD map files (optional, low priority)
- Primary camera: `CAM_FRONT`

## Project

- Project name: `KG_Scene_Retrieval`
- Root directory: `D:\KG_Scene_Retrieval\`

## Services

- Milvus: localhost (default port 19530)
- Neo4j: localhost (default port 7687)

## Config

- All configurable values (paths, hosts, ports, passwords) must be stored in `config.py` at the project root.
- If `config.py` does not exist, create it automatically before writing any other code.
- All other files must import from `config.py` instead of hardcoding values.
