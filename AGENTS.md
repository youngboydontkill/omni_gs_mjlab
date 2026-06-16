# Omni-GS-Playground Agent Guide

本文件是顶层协作规则，适用于仓库根目录下的工作。进入子目录时，如果存在更靠近目标文件的 `AGENTS.md`，以更具体的文件为准。

## 基本工作流

- 从仓库根目录执行项目命令。
- 所有 Python 相关命令使用 `uv run ...`，不要直接使用 `python` 或 `python3`。
- 文档默认使用中文，除非目标文件已有明确英文风格或任务要求英文。
- 小步修改，保持变更聚焦；不要顺手做无关重构。
- 不要把废弃代码注释掉，确认无用后直接删除。
- 命令示例要保持可复制，优先写真实存在的任务 ID、脚本路径和参数。

## 依赖与环境

- Python 版本保持 `>=3.11,<3.12`。
- 保持 `mjlab==1.4.0`。
- 保持 `rsl-rl-lib==5.2.0`，并确保它来自 workspace 成员 `packages/rsl_rl`。
- 不要用 PyPI 上的 `rsl-rl-lib` 替换本地 workspace 包。
- `uv.lock` 是当前依赖快照的权威来源。依赖声明有意变化时，运行：

```bash
uv lock
uv sync --locked
```

- 文档或小改动至少用以下命令确认 lock 文件仍一致：

```bash
uv lock --check
```

更多版本和 DeFM 相关约束见 `doc/uv_environment.md`。

## 代码边界

- 优先修改项目自有代码：
  - `src/omni_gs_playground/`
  - `src/sensor/`
  - `src/managers/`
  - `scripts/`
  - `doc/`
- `packages/rsl_rl` 是本项目使用的本地改造版 workspace 包，可以按任务需要谨慎修改，并同步验证依赖和相关测试。
- `packages/3dgeer` 和 `packages/GaussianRenderer` 默认视作 vendored / 上游移植代码。除非任务明确要求，不做无关格式化、重构或接口改动。
- `src/managers/omni_gs_manger.py` 当前为空；不要在文档或代码中把 GS manager 描述为已完成能力。
- `src/sensor/gs_camera_sensor.py` 属于实验性 GS camera sensor，改动前需要检查 MJLab sensor context 和 MuJoCo / MuJoCo Warp 渲染限制。

## 任务与脚本约定

- 新增环境任务必须通过 `mjlab.tasks.registry.register_mjlab_task` 注册。
- 注册后至少确认任务可被发现：

```bash
uv run python scripts/list_envs.py --keyword <name>
```

- `scripts/train.py` 和 `scripts/play.py` 的 help 需要先提供任务 ID：

```bash
uv run python scripts/train.py Kuavo-S54-Flat --help
uv run python scripts/play.py Kuavo-S54-Flat --help
```

- `scripts/play.py` 当前公开稳定的本地模型入口是 `--checkpoint-file`。不要在 README 或示例中推荐未出现在 help 中的 `wandb_run_path` / `registry_name` 参数。
- 训练日志默认写入 `logs/rsl_rl/`；不要把日志、checkpoint 或视频纳入源码改动。

## 验证要求

- 文档和依赖无关的小改动：

```bash
uv lock --check
```

- 任务注册、配置或脚本参数改动：

```bash
uv run python scripts/list_envs.py --keyword <name>
uv run python scripts/train.py <Task-ID> --help
uv run python scripts/play.py <Task-ID> --help
```

- RL、GPU、渲染或传感器相关改动需要说明实际跑过哪些验证；如果没有完整训练，也要明确只做了 smoke test 或静态检查。
- 涉及 `packages/rsl_rl` 的改动，优先运行该包相关测试或最小可复现脚本。

## 生成物与版本控制

不要提交以下生成物：

- `.venv/`
- `__pycache__/`
- `.pytest_cache/`
- `logs/`
- 模型 checkpoint
- 训练或回放视频
- TorchHub / HuggingFace / MuJoCo 缓存
- 本地调试输出和临时数据集

如果生成物已存在于工作区，除非任务明确要求清理，不要把它们混入代码变更。

## 风格偏好

- 代码以清晰、直接、最小实现为优先。
- 不为单次使用场景提前抽象。
- 不添加与当前任务无关的兼容层、兜底逻辑或配置项。
- 研究代码要注意性能和显存占用；改动并行环境、观测、渲染或特征缓存时，说明可能的内存影响。
- 保持现有配置风格，优先复用 MJLab、RSL-RL 和项目内已有 helper。
