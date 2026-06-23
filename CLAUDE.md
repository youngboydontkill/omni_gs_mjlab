# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> 顶层协作规则、代码边界与生成物约定见 `AGENTS.md`；版本与 DeFM 安装约束见 `doc/uv_environment.md`；DeFM 特征缓存优化的设计与兼容性影响见 `doc/defm_feature_cache_optimization.md`。本文件补充常用命令与整体架构，不重复上述内容。文档默认使用中文。

## 环境

- Python `>=3.11,<3.12`，通过 `uv` 管理依赖，`uv.lock` 是权威快照。
- **必须固定**：`mjlab==1.4.0`、`rsl-rl-lib==5.2.0`。`rsl-rl-lib` 必须解析到 workspace 成员 `packages/rsl_rl`（根 `pyproject.toml` 的 `[tool.uv.sources]` / `[tool.uv.workspace]` 不可删除）。不要用 PyPI 包替换，也不要升级到 5.2.0 以上。
- **不要安装 `defm` 源码包**（会引入冲突的 NumPy 约束）。`DefmModel` 通过 `torch.hub.load("leggedrobotics/defm:main", "defm_vit_s14", pretrained=True)` 加载，首次需要网络并写入 TorchHub 缓存。
- 所有 Python 命令一律 `uv run ...`，不要直接用 `python`/`python3`。

## 常用命令

```bash
uv sync --locked                       # 从仓库根目录同步环境
uv lock --check                        # 确认依赖未漂移（文档/小改动最低验证）

# 任务发现与参数帮助（train/play 的 --help 必须先给出任务 ID）
uv run python scripts/list_envs.py --keyword Kuavo
uv run python scripts/train.py Kuavo-S54-Flat --help
uv run python scripts/play.py  Kuavo-S54-Flat --help

# 训练
uv run python scripts/train.py Kuavo-S54-Flat  --gpu-ids 0      # 单 GPU
uv run python scripts/train.py Kuavo-S54-Rough --gpu-ids all --video True   # 多 GPU（torchrunx）

# 回放（play.py 稳定的本地模型入口是 --checkpoint-file）
uv run python scripts/play.py Kuavo-S54-Flat --agent zero --num-envs 4 --viewer viser
uv run python scripts/play.py Kuavo-S54-Flat --checkpoint-file logs/rsl_rl/<experiment>/<run>/model_*.pt

uv run python scripts/visualize_terrain.py     # 交互式地形可视化
```

依赖声明有意变更时：`uv lock && uv sync --locked`。

### 测试

`pytest` 不在锁定环境中（见 `doc/defm_feature_cache_optimization.md` 的说明）。`packages/rsl_rl` 自带测试套件，临时引入 pytest 运行：

```bash
uv run --with pytest pytest packages/rsl_rl/tests/models/test_defm_model.py
uv run --with pytest pytest packages/rsl_rl/tests/algorithms/test_ppo.py::<test_name>   # 单个测试
```

`packages/rsl_rl` 的 DeFM 测试使用轻量 mock encoder，不加载 TorchHub、不使用 GPU，可在 CPU 上快速验证特征缓存与对齐逻辑。

## 已注册任务

通过 `mjlab.tasks.registry.register_mjlab_task` 在各 config 包的 `__init__.py` 中注册：`Unitree-G1-Flat/Rough`、`Kuavo-S45-Flat/Rough`、`Kuavo-S54-Flat/Rough`、`Kuavo-S54-Head-CNN-Rough`。所有任务的 runner 都是项目自有的 `VelocityOnPolicyRunner`。新增任务后必须确认 `scripts/list_envs.py` 能发现。

## 架构

项目是基于 **MJLab** 的 manager-based 并行 RL：在 MuJoCo / MuJoCo Warp 中训练双足机器人速度跟踪策略，用本地改造版 **rsl-rl-lib** 做 PPO。GS（3D Gaussian Splatting / 鱼眼相机）渲染链路目前是实验性预留接口。

### 任务发现链路

`scripts/*.py` 的 `main()` 都会先 `import mjlab.tasks` 和 `import omni_gs_playground.tasks`。后者通过 `mjlab.utils.lab_api.tasks.importer.import_packages` 遍历 `tasks/velocity/config/*/`（黑名单 `utils`、`.mdp`），从而执行各包 `__init__.py` 中的 `register_mjlab_task`。所以**新增任务 = 在某个 config 目录的 `__init__.py` 里调用 `register_mjlab_task`**，注册项里同时绑定 `env_cfg` / `play_env_cfg` / `rl_cfg` / `runner_cls`。tyro 的任务选择来自 `list_tasks()`，因此任务 ID 必须先注册才能出现在 `--help` 里。

### 分层环境配置（`tasks/velocity/`）

配置按「基类 → 机器人基类 → 具体任务」层层特化，理解这层级联是改动的关键：

1. `velocity_env_cfg.py::make_velocity_env_cfg()` — 纯本体感知（proprioception）基类：定义 observations（actor/critic 两组）、rewards、actions（`JointPositionAction`）、events、curriculum、`terrain_scan` 射线传感器与 rough 地形生成器。**不含深度相机**。
2. `velocity_env_cfg.py::make_kuavo_velocity_env_cfg()` — Kuavo 共享基类：在 1 的基础上挂载 42×42 `depth` 相机，移除 `phase` / `height_scan` 观测，加入 `depth` 观测项，改为仅前进的 twist 命令，移除 `foot_gait` / `angular_momentum` reward 与 `command_vel` 课程。
3. `config/<robot>/env_cfgs.py` — 具体任务工厂。Kuavo 的 `_kuavo_rough_env_cfg()` 设置机器人 entity、足部接触传感器、受控关节正则、pose reward 的 std、仿真参数（`nconmax`/`ccd_iterations` 等显存敏感项），再用 `play=True` 分支把课程/扰动关掉。`kuavo_s54_rough_env_cfg`、`kuavo_s54_head_cnn_rough_env_cfg`、各 `*_flat_env_cfg` 都从这里派生（flat 任务把地形换成 plane 并降低 contact 上限）。

`_separate_depth_observations()` 会把 `depth` 从 actor/critic 组里抽出来，放进独立的 `actor_depth` / `critic_depth` 2D 组（actor 组 history=5），供视觉 encoder 策略使用；这必须与下面的 `obs_groups` 配对。

### 观测与深度表示（`tasks/velocity/mdp/observations.py`）

`mdp.depth_image_obs()` 是唯一的深度观测函数，由 `flatten`/`normalize` 参数决定输出形态：

- `normalize=False, flatten=False` → 米制深度 `[N,1,H,W]`，给 **DeFM** 用。
- `normalize=True, flatten=False` → 归一化到 `[-1,1]`，给 **CNN** 用。
- `flatten=True` → 展平给 MLP。会做 `nan_to_num` + `[near, far]` clip。

`mdp/` 通过 `from mjlab.envs.mdp import *` 复用 MJLab 内置 term，再叠加本项目的 `curriculums / observations / rewards / terminations / velocity_command`。

### RL 配置与模型类型（`config/<robot>/rl_cfg.py`）

同一个 PPO 算法配置（5 epochs × 4 mini-batches、adaptive LR、`desired_kl=0.01`），按任务切换 actor/critic 的 `class_name`：

| 任务 | 模型 | 关键点 |
| --- | --- | --- |
| Kuavo-S45/S54-Flat | `MLPModel` (默认) | 纯本体感知 |
| Kuavo-S54-Head-CNN-Rough | `CNNModel` | `cnn_cfg` 自定义卷积；`share_cnn_encoders=False` |
| Kuavo-S54-Rough | `DefmModel` | `defm_cfg`：冻结 `defm_vit_s14`、`target_size=42`、`token_feature_dim=32`；`share_cnn_encoders=True` |

模型通过 `obs_groups` 声明每个 policy 头消费哪些观测组（如 `{"actor": ("actor","actor_depth"), "critic": ("critic","critic_depth")}`），与上面的分组配置一一对应。`RslRlDefmModelCfg` 是本项目新增的带 `defm_cfg` 的 dataclass。

### DeFM 特征缓存（`packages/rsl_rl/`）

这是本项目相对上游 rsl-rl 的核心改造，**会破坏旧 DeFM checkpoint**（详见 `doc/defm_feature_cache_optimization.md`）：

- **降维**：每个 384 维 patch token 按通道分 `token_feature_dim`(=32，必须是 384 的正因数) 组取均值，`9×384` → `9×32` → 展平 288 维，替代原先的 3456 维。
- **缓存**：`RolloutStorage.Transition/Batch` 新增 `extra: TensorDict | None`。当 Actor/Critic 都支持 feature cache 且所有 DeFM encoder 冻结时，PPO 在 rollout 时各算一次 encoder、`detach` 后存入 `extra`，learning 阶段直接走策略 MLP，把每轮 update 的 encoder 调用从约 40 次降到 0 次。
- **不启用缓存的情形**：`trainable=True` 的 encoder（保留梯度）、symmetry augmentation（镜像观测无法对齐缓存）。
- `packages/rsl_rl/rsl_rl/models/defm_model.py` 同时实现 encoder 共享（`share_cnn_encoders` 时 Actor/Critic 共享同一组 `_DefmEncoder` 对象）。

### 自定义 Runner（`tasks/velocity/rl/runner.py`）

`VelocityOnPolicyRunner(MjlabOnPolicyRunner)` 覆写 `save()`：每次存 checkpoint 时额外导出 `policy.onnx` 并附上 base metadata（wandb 模式下还会上传）。JIT/ONNX 推理仍从原始深度走完整 DeFM 前向 + 相同的固定分组平均。

### GS / 渲染（实验性）

`src/sensor/gs_camera_sensor.py` 是支持 pinhole/fisheye 的实验性相机 sensor，改动前需核对 MJLab sensor context 与 MuJoCo/MuJoCo Warp 渲染限制（同场景所有相机须共用 `use_textures`/`use_shadows`/`enabled_geom_groups`）。`src/managers/omni_gs_manger.py` 当前为空——**不要把 GS manager 描述为已完成能力**。

## 代码边界

- 优先改 `src/omni_gs_playground/`、`src/sensor/`、`src/managers/`、`scripts/`、`doc/`。
- `packages/rsl_rl` 是本地改造版，可按任务谨慎修改并补测试；改完优先跑对应 `packages/rsl_rl/tests/`。
- 其它 `packages/*`（仓库文档中提到的 `3dgeer` / `GaussianRenderer` 等上游移植代码）视作 vendored，不做无关重构或接口改动。
- 显存敏感项：改动并行环境数、观测维度、深度图分辨率、DeFM `token_feature_dim` 或 rollout `extra` 缓存时，需说明对 GPU 显存与 rollout 缓存的影响。

## 输出

训练产物默认写入 `logs/rsl_rl/<experiment_name>/<YYYY-MM-DD_HH-MM-SS>[_<run_name>]/`，含 `params/{env,agent}.yaml`、`videos/{train,play}/`、多 GPU 的 `torchrunx/`。`logs/`、checkpoint、视频与各类缓存（TorchHub/HF/MuJoCo）均不提交。
