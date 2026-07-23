# Omni-GS-Playground

Omni-GS-Playground 是一个基于 MJLab 的并行强化学习实验项目，当前重点是 MuJoCo / MuJoCo Warp 中的双足机器人速度跟踪任务，并为深度观测、视觉策略、鱼眼相机和 3D Gaussian Splatting 渲染实验预留接口。

项目使用 RSL-RL 训练策略，支持 Unitree G1 与 Kuavo S45/S54 系列机器人在平地和粗糙地形上的速度控制实验。GS 相机相关代码仍处于实验阶段，当前稳定入口以 MJLab 环境、训练脚本、回放脚本和地形可视化工具为主。

## 环境要求

- Python `>=3.11,<3.12`
- `uv`
- Linux + CUDA GPU 推荐用于训练；CPU 可用于部分轻量检查和命令帮助。
- Headless / 服务器环境建议使用 EGL。训练脚本会设置 `MUJOCO_GL=egl`，多 GPU 时会按 local rank 设置 `MUJOCO_EGL_DEVICE_ID`。

依赖版本由 `uv.lock` 固定。`rsl-rl-lib` 必须解析到 workspace 内的 `packages/rsl_rl`，不要替换为 PyPI 包；更多约束见 [doc/uv_environment.md](doc/uv_environment.md)。

## Quick Start

从仓库根目录同步环境：

```bash
uv sync --locked
```

查看已注册环境：

```bash
uv run python scripts/list_envs.py --keyword Kuavo
```

当前项目注册的任务包括：

- `Unitree-G1-Flat`
- `Unitree-G1-Rough`
- `Kuavo-S45-Flat`
- `Kuavo-S45-Rough`
- `Kuavo-S54-Flat`
- `Kuavo-S54-Rough`
- `Kuavo-S54-Head-CNN-Rough`

训练单 GPU 策略：

```bash
uv run python scripts/train.py Kuavo-S54-Flat --gpu-ids 0
```

训练多 GPU 策略并录制训练视频：

```bash
uv run python scripts/train.py Kuavo-S54-Rough --gpu-ids all --video True
```

使用零动作策略快速打开回放 viewer：

```bash
uv run python scripts/play.py Kuavo-S54-Flat --agent zero --num-envs 4 --viewer viser
```

加载本地 checkpoint 回放训练好的策略：

```bash
uv run python scripts/play.py Kuavo-S54-Flat --checkpoint-file logs/rsl_rl/<experiment>/<run>/model_*.pt
```

启动交互式地形可视化：

```bash
uv run python scripts/visualize_terrain.py
```

Tyro 参数帮助需要先给出任务 ID，例如：

```bash
uv run python scripts/train.py Kuavo-S54-Flat --help
uv run python scripts/play.py Kuavo-S54-Flat --help
```

## 输出与日志

训练输出默认写入：

```text
logs/rsl_rl/<experiment>/<timestamp>[_<run_name>]/
```

其中：

- 环境配置备份：`params/env.yaml`
- 算法配置备份：`params/agent.yaml`
- 训练视频：`videos/train/`
- 回放视频：`videos/play/`
- 多 GPU `torchrunx` 日志：默认位于当前 run 目录下的 `torchrunx/`

`logs/`、checkpoint、视频和缓存文件属于生成物，通常不应提交到版本库。

## 项目结构

```text
scripts/                       训练、回放、环境列表和地形可视化入口
src/omni_gs_playground/        项目任务、机器人资产和 RL 配置
src/sensor/                    实验性 GS camera sensor
src/managers/                  GS manager 预留目录，当前尚未实现
packages/rsl_rl/               本项目使用的本地改造版 rsl-rl-lib workspace 包
packages/GaussianRenderer/     Gaussian renderer 相关上游/移植代码
packages/3dgeer/               3DGEER 相关上游/移植代码
doc/                           项目说明和设计记录
```

## 开发提示

- 所有 Python 命令优先使用 `uv run ...`。
- 修改依赖后需要重新生成并检查 `uv.lock`。
- 新增任务应通过 `mjlab.tasks.registry.register_mjlab_task` 注册，并确认 `scripts/list_envs.py` 可发现。
- `scripts/play.py` 当前公开的本地模型入口是 `--checkpoint-file`；不要依赖未暴露在 help 中的 WandB 参数路径。
- `src/sensor/gs_camera_sensor.py` 和 GS/fisheye 渲染链路仍偏实验性，使用前请检查具体配置和渲染上下文限制。

## Acknowledge

本项目参考和复用了以下项目/组件的工作：

1. [GS-Playground](https://github.com/discoverse-dev/gs_playground/)
2. [3DGEER](https://github.com/boschresearch/3dgeer)
3. [GaussianRenderer](https://github.com/discoverse-dev/GaussianRenderer/)
4. MJLab
5. [RSL-RL](https://github.com/leggedrobotics/rsl_rl)


agent.resume=True