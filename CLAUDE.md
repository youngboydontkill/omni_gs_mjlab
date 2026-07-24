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
uv run python scripts/train.py Kuavo-S54-Flat  --gpu-ids 0           # 单 GPU
uv run python scripts/train.py Kuavo-S54-Rough --gpu-ids all --video True   # 多 GPU

# 回放（play.py 稳定的本地模型入口是 --checkpoint-file）
uv run python scripts/play.py Kuavo-S54-Flat --agent zero --num-envs 4 --viewer viser
uv run python scripts/play.py Kuavo-S54-Flat --checkpoint-file logs/rsl_rl/<experiment>/<run>/model_*.pt
uv run python scripts/play.py Kuavo-S45-Rough-Distill \
  --checkpoint-file logs/rsl_rl/<experiment>/<run>/model_*.pt \
  --num-envs 1 --viewer viser

uv run python scripts/visualize_terrain.py     # 交互式地形可视化
uv run python scripts/visualize_teacher.py     # 独立 teacher 可视化（EMP checkpoint 推理）
uv run python scripts/estimate_teacher_norm.py # 统计 teacher 观测归一化范围
uv run python scripts/reexport_onnx.py         # 从 checkpoint 重新导出 ONNX
```

依赖声明有意变更时：`uv lock && uv sync --locked`。

### 测试

`pytest` 不在锁定环境中（见 `doc/defm_feature_cache_optimization.md` 的说明）。`packages/rsl_rl` 自带测试套件，临时引入 pytest 运行：

```bash
# 运行全部模型测试（MLP/CNN/DeFM/MoE/RNN ≈ 50+ 条）
uv run --with pytest pytest packages/rsl_rl/tests/models/

# 蒸馏算法单测（loss 下降、teacher 冻结、梯度累积、尾部窗口）
uv run --with pytest pytest packages/rsl_rl/tests/algorithms/test_distillation.py

# 单个测试
uv run --with pytest pytest packages/rsl_rl/tests/models/test_moe_model.py::<test_name>
uv run --with pytest pytest packages/rsl_rl/tests/algorithms/test_ppo.py::<test_name>
```

`packages/rsl_rl` 的 DeFM 测试使用轻量 mock encoder，不加载 TorchHub、不使用 GPU，可在 CPU 上快速验证特征缓存与对齐逻辑。蒸馏单测同样在 CPU 上运行。

## 已注册任务

通过 `mjlab.tasks.registry.register_mjlab_task` 在各 config 包的 `__init__.py` 中注册。以下为当前所有任务（见 `src/.../config/kuavo/__init__.py` 与 `config/g1/__init__.py`）：

| 任务 | runner | 模型 / 算法 | 说明 |
| --- | --- | --- | --- |
| Kuavo-S45-Rough | `VelocityOnPolicyRunner` | CNN | S45 EMP rewards + depth CNN |
| Kuavo-S45-DeFM-Rough | `VelocityOnPolicyRunner` | DeFM | S45 冻结 ViT 深度编码器 |
| Kuavo-S45-Flat | `VelocityOnPolicyRunner` | CNN | 平地版 S45-Rough |
| Kuavo-S45-Flat-Blind | `VelocityOnPolicyRunner` | MLP | 纯本体感知，无深度相机 |
| Kuavo-S45-Stairs | `VelocityOnPolicyRunner` | CNN | 台阶专用 S45 teacher（仅 stair 地形） |
| Kuavo-S45-Slope | `VelocityOnPolicyRunner` | CNN | 斜坡专用 S45 teacher（仅 slope 地形） |
| Kuavo-S45-AMP-Rough | `AMPVelocityOnPolicyRunner` | CNN+AMP | 对抗运动先验风格奖励 |
| Kuavo-S45-Rough-Distill | `DistillationRunner` | CNN→EMP Teacher | 行为克隆蒸馏 student 策略 |
| Kuavo-S54-Rough | `VelocityOnPolicyRunner` | DeFM(42×42) | 27 关节 S54 |
| Kuavo-S54-Head-CNN-Rough | `VelocityOnPolicyRunner` | CNN | 带可控头部的 S54 |
| Kuavo-S54-Head-MoE-Rough | `VelocityOnPolicyRunner` | MoE(4 experts) | MoE-Loco 风格门控专家 |
| Kuavo-S54-Flat | `VelocityOnPolicyRunner` | MLP | 平地 S54 |
| Unitree-G1-Flat | `VelocityOnPolicyRunner` | MLP | G1 机器人平地 |
| Unitree-G1-Rough | `VelocityOnPolicyRunner` | MLP | G1 机器人粗糙地形 |

新增任务后必须确认 `scripts/list_envs.py` 能发现。

## 架构

项目是基于 **MJLab** 的 manager-based 并行 RL：在 MuJoCo / MuJoCo Warp 中训练双足机器人速度跟踪策略，用本地改造版 **rsl-rl-lib** 做 PPO。GS（3D Gaussian Splatting / 鱼眼相机）渲染链路目前是实验性预留接口。

### 任务发现链路

`scripts/*.py` 的 `main()` 都会先 `import mjlab.tasks` 和 `import omni_gs_playground.tasks`。后者通过 `mjlab.utils.lab_api.tasks.importer.import_packages` 遍历 `tasks/velocity/config/*/`（黑名单 `utils`、`.mdp`），从而执行各包 `__init__.py` 中的 `register_mjlab_task`。所以**新增任务 = 在某个 config 目录的 `__init__.py` 里调用 `register_mjlab_task`**，注册项里同时绑定 `env_cfg` / `play_env_cfg` / `rl_cfg` / `runner_cls`。tyro 的任务选择来自 `list_tasks()`，因此任务 ID 必须先注册才能出现在 `--help` 里。

### 分层环境配置（`tasks/velocity/`）

配置按「基类 → 机器人基类 → 具体任务」层层特化，理解这层级联是改动的关键：

1. `velocity_env_cfg.py::make_velocity_env_cfg()` — 纯本体感知（proprioception）基类：定义 observations（actor/critic 两组）、rewards、actions（`JointPositionAction`）、events、curriculum、`terrain_scan` 射线传感器与 rough 地形生成器。**不含深度相机**。
2. `velocity_env_cfg.py::make_kuavo_velocity_env_cfg()` — Kuavo 共享基类：在 1 的基础上挂载 42×42 `depth` 相机，移除 `phase` / `height_scan` 观测，加入 `depth` 观测项，改为仅前进的 twist 命令，移除 `foot_gait` / `angular_momentum` reward 与 `command_vel` 课程。
3. `config/<robot>/env_cfgs.py` — 具体任务工厂。Kuavo 的 `_kuavo_rough_env_cfg()` 设置机器人 entity、足部接触传感器、受控关节正则、pose reward 的 std、仿真参数（`nconmax`/`ccd_iterations` 等显存敏感项），再用 `play=True` 分支把课程/扰动关掉。

`_separate_depth_observations()` 会把 `depth` 从 actor/critic 组里抽出来，放进独立的 `actor_depth` / `critic_depth` 2D 组（actor 组 history=5, no history flatten），供视觉 encoder 策略使用；这必须与下面的 `obs_groups` 配对。

### S45 EMP 奖励体系（`config/kuavo/env_cfgs.py`）

`_apply_s45_emp_rewards()` 对 S45 系列任务（Rough、DeFM-Rough、Flat、Flat-Blind、Distill）应用来自 Leju-IsaacLab EMP 的奖励体系，在共享配置基础上：

- **新增传感器**：`undesired_body_contact`（非足部身体接触惩罚）、左右足 `feet_[lr]_forward_scanner`（`toe_touch` 罚碰立面）、左右足 `feet_[lr]_edge_scanner`（`edge_contact` 罚踩台阶边缘，Hiking in the Wild §III-C 近似）。
- **替换/调整奖励权重**：提升速度跟踪权重（vx=8.0, wz=3.0）、增加 `joint_power_l2`、`dof_torques_ankle_l2`、`illegal_dof_barrier`、`feet_too_near`、`fly`、`undesired_contacts`、`toe_touch` 等项。
- S54 不走此路径，使用默认奖励集。

### 深度腐蚀（`tasks/velocity/mdp/observations.py`）

`depth_image_obs()` 的 `corrupt=True` 分支实现 Hiking in the Wild §III-B2 的 F_sim 退化链（仅对 actor 开启，critic 和 play 为干净深度）：range-dependent Gaussian noise → disparity white regions → Gaussian blur → OOD dropout。所有 CPU 小张量操作，对显存影响极小。

### 观测与深度表示（`tasks/velocity/mdp/observations.py`）

`mdp.depth_image_obs()` 是唯一的深度观测函数，由 `flatten`/`normalize`/`corrupt` 参数决定输出形态：

- `normalize=False, flatten=False` → 米制深度 `[N,1,H,W]`，给 **DeFM** 用。
- `normalize=True, flatten=False` → 归一化到 `[-1,1]`，给 **CNN** 用。
- `flatten=True` → 展平给 MLP。会做 `nan_to_num` + `[near, far]` clip。

`mdp/` 通过 `from mjlab.envs.mdp import *` 复用 MJLab 内置 term，再叠加本项目的 `curriculums / observations / rewards / terminations / velocity_command`。

### RL 配置与模型类型（`config/<robot>/rl_cfg.py`）

按任务切换 actor/critic 的 `class_name`。PPO 算法配置（5 epochs × 4 mini-batches、adaptive LR、`desired_kl=0.01`）在 `_kuavo_base_ppo_runner_cfg()` 中共享：

| 任务 | 模型 | 关键点 |
| --- | --- | --- |
| Kuavo-S45/S54-Flat | `MLPModel` (默认) | 纯本体感知 |
| Kuavo-S45/S54/Head-CNN-Rough | `CNNModel` | `cnn_cfg` 自定义卷积；`share_cnn_encoders=False` |
| Kuavo-S54-Head-MoE-Rough | `MoEModel(CNNModel)` | 门控软加权专家头插在 CNN latent 与策略 MLP 之间 |
| Kuavo-S45/S54-Rough (DeFM) | `DefmModel` | `defm_cfg`：冻结 `defm_vit_s14`、`target_size=42`、`token_feature_dim=32`；`share_cnn_encoders=True` |
| Kuavo-S45-Rough-Distill (Student) | `CNNModel` | 同 S45-CNN-Rough，obs_groups=("actor","actor_depth") |
| Kuavo-S45-Rough-Distill (Teacher) | `EMPTeacherModel` | 冻结 EMP checkpoint，见下节 |

模型通过 `obs_groups` 声明每个 policy 头消费哪些观测组（如 `{"actor": ("actor","actor_depth"), "critic": ("critic","critic_depth")}`），与上面的分组配置一一对应。`RslRlDefmModelCfg` / `RslRlMoEModelCfg` 是本项目新增的带额外字段的 dataclass。

### S45 动作 std 问题

S45 任务（CNN/DeFM）使用 `HeteroscedasticGaussianDistribution`（状态相关 std）+ `init_std=0.3~0.5`，解决状态无关标量 std 被 entropy 单边顶高的问题（详见 `doc/action_std_in_ppo.md`）。S54 及 Flat 任务默认 `GaussianDistribution`。

### 自定义 Runner（`tasks/velocity/rl/runner.py`）

- `VelocityOnPolicyRunner(MjlabOnPolicyRunner)`：覆写 `save()` 每次存 checkpoint 时额外导出 `policy.onnx` 并附上 base metadata。自动处理训练 (term-major) 和部署 (frame-major) 观测布局之间的 permute。
- `AMPVelocityOnPolicyRunner(VelocityOnPolicyRunner)`：从 config 顶层提取 `amp_cfg`，构建 `AMPStateComputer`，用 `AMPVecEnvWrapper` 重包装 env，attach reference motion sampler。无 `amp_cfg` 时降级为标准 runner。

### 蒸馏与 EMP Teacher 模型

`Kuavo-S45-Rough-Distill` 使用 `DistillationRunner` → `Distillation` 算法，实现在线行为克隆：

- **Student**：`CNNModel`（depth camera + proprioception），`obs_groups=("actor","actor_depth")`。
- **Teacher**：`EMPTeacherModel` 加载 `doc/model_48350.pt`（Isaac Lab 训练的 EMP checkpoint），`obs_groups=("teacher_cmd","teacher_proprio","teacher_height")`。

`distillation.py` 架构：
- `act()`：student 推理（默认确定性 mean action），teacher 打标签，存 `transition.privileged_actions`。
- `update()`：**梯度累积**（按 `gradient_length` 窗口累加并求均值后再反传），MSE/Huber loss。最后不足窗口不丢弃。
- `construct_algorithm()` 静态方法：从 dict 格式的 runner cfg 同时构建 student/teacher、storage 和算法。

`EMPTeacherModel` 内部：
- 从 checkpoint 加载 `ActorCriticCNN`（仅 actor 分支：423 维 normalizer + PolicyHeightMapCNN + 487→512→256→128→26 MLP）。
- **关节顺序转换**：`teacher_proprio`（420 维 term-major）中的 joint_pos/joint_vel/actions 块从 MJCF 分组顺序重排到 Lab USD 交错顺序（`lab2mjcf`/`mjcf2lab` 置换表），动作输出再转回 MJCF 顺序。
- Normalizer 完整保留 checkpoint 统计（`_mean`/`_std`/`_var`/`_count`），不做重估或覆盖。
- Teacher height scan：`terrain_scan` 传感器设置 `include_geom_groups=(0,)` 确保只命中 terrain、不扫到自身（S45 足/手臂 mesh 均在 group 1）。

`doc/emp-teacher-model.md` 记录完整的 486 维输入排列、关节置换表、CNN 架构和推理代码。`doc/distillation_research.md` 记录后续优化方向（DAgger-lite、BC+PPO、latent 蒸馏）。

### AMP（Adversarial Motion Priors）

`Kuavo-S45-AMP-Rough` 使用 AMP 风格奖励使步态更自然。骨架代码在 `packages/rsl_rl/rsl_rl/extensions/amp.py`，环境侧组件在 `tasks/velocity/amp/`：
- `AMPDiscriminator`：MLP trunk → 单 logit，MSE loss（式 7）+ quadratic style reward（式 8）。
- `MotionDataset`：从 CSV 运动数据文件加载，逐帧计算 AMP state（body-frame v/ω/g/q/q̇），滑动窗口扁平为 `[N, 244]`。
- `AMPSampler`：从 dataset 均匀采样用作判别器参考。
- `AMPVecEnvWrapper`：在 `step()` 中为每个环境维护 AMP 观测 history buffer。
- **与 DeFM feature cache 互斥**（同 symmetry augmentation）。

默认使用 `include_keywords` 过滤运动数据（只保留行走/转向，排除跳跃等），见 `doc/amp_scaffold.md`。

### DeFM 特征缓存（`packages/rsl_rl/`）

这是本项目相对上游 rsl-rl 的核心改造，**会破坏旧 DeFM checkpoint**（详见 `doc/defm_feature_cache_optimization.md`）：

- **降维**：每个 384 维 patch token 按通道分 `token_feature_dim`(=32，必须是 384 的正因数) 组取均值，`9×384` → `9×32` → 展平 288 维，替代原先的 3456 维。
- **缓存**：`RolloutStorage.Transition/Batch` 新增 `extra: TensorDict | None`。当 Actor/Critic 都支持 feature cache 且所有 DeFM encoder 冻结时，PPO 在 rollout 时各算一次 encoder、`detach` 后存入 `extra`，learning 阶段直接走策略 MLP，把每轮 update 的 encoder 调用从约 40 次降到 0 次。
- **不启用缓存的情形**：`trainable=True` 的 encoder（保留梯度）、symmetry augmentation（镜像观测无法对齐缓存）。
- `packages/rsl_rl/rsl_rl/models/defm_model.py` 同时实现 encoder 共享（`share_cnn_encoders` 时 Actor/Critic 共享同一组 `_DefmEncoder` 对象）。

### MoE 模型（`packages/rsl_rl/rsl_rl/models/moe_model.py`）

`MoEModel(CNNModel)` 在 CNN encoder latent 与策略 MLP 之间插入门控软加权专家头。JIT/ONNX 可 trace（无动态分支）。参数随 `num_experts` 线性增长，默认 4 专家、专家隐层较窄（`(256,)`）、gate 隐层 (64,)、moe_out=256。

### GS / 渲染（实验性）

`src/sensor/gs_camera_sensor.py` 是支持 pinhole/fisheye 的实验性相机 sensor，改动前需核对 MJLab sensor context 与 MuJoCo/MuJoCo Warp 渲染限制（同场景所有相机须共用 `use_textures`/`use_shadows`/`enabled_geom_groups`）。`src/managers/omni_gs_manger.py` 当前为空——**不要把 GS manager 描述为已完成能力**。

## 机器人常量

各机器人关节顺序/执行器参数/动作 scale 在 `src/.../assets/robots/<robot>/`：

| 机器人 | 文件 | 关节数 | 说明 |
| --- | --- | --- | --- |
| Kuavo S45 | `kuavo_s45_constants.py` | 26 | 6 腿 + 7 臂 × 2，无腰部 |
| Kuavo S54 | `kuavo_s54_constants.py` | 27 | 6 腿 + 1 腰 + 7 臂 × 2 |
| Unitree G1 | `g1_23dof_constants.py` | 23 | 23 关节 G1 |

S45 与 S54 动作 scale/执行器参数独立；S54 带可控头部（需 `head` 机器人 cfg）。

## 代码边界

- 优先改 `src/omni_gs_playground/`、`src/sensor/`、`src/managers/`、`scripts/`、`doc/`。
- `packages/rsl_rl` 是本地改造版，可按任务谨慎修改并补测试；改完优先跑对应 `packages/rsl_rl/tests/`。
- 其它 `packages/*`（仓库文档中提到的 `3dgeer` / `GaussianRenderer` 等上游移植代码）视作 vendored，不做无关重构或接口改动。
- 显存敏感项：改动并行环境数、观测维度、深度图分辨率、DeFM `token_feature_dim` 或 rollout `extra` 缓存时，需说明对 GPU 显存与 rollout 缓存的影响。

## 输出

训练产物默认写入 `logs/rsl_rl/<experiment_name>/<YYYY-MM-DD_HH-MM-SS>[_<run_name>]/`，含 `params/{env,agent}.yaml`、`videos/{train,play}/`、多 GPU 的 `torchrunx/`。`logs/`、checkpoint、视频与各类缓存（TorchHub/HF/MuJoCo）均不提交。

## 设计记录（`doc/`）

| 文件 | 内容 |
| --- | --- |
| `devlog.md` | 按时间倒序的会话改动日志（动机、根因、改动、验证） |
| `emp-teacher-model.md` | EMP teacher checkpoint 结构、486 维输入排列、关节置换表、推理代码 |
| `distillation_research.md` | 蒸馏方案调研：DAgger-lite、BC+PPO、latent 蒸馏路线图 |
| `amp_scaffold.md` | AMP 判别器骨架、观测格式、启用步骤 |
| `action_std_in_ppo.md` | PPO 动作 std 发散根因分析与 Heteroscedastic 修复 |
| `defm_feature_cache_optimization.md` | DeFM 特征缓存改造与旧 checkpoint 兼容性 |
| `uv_environment.md` | 版本约束与 DeFM 安装说明 |
| `nan_debugging.md` | 观测 NaN 溯源与修复记录 |
| `terrain_curriculum_stuck.md` | 地形课程 stuck 根因分析与 P0 复盘 |
| `joint_ordering.md` | 各机器人关节顺序说明 |
| `reward_migrate.md` / `toe_touch_migrate.md` | 奖励迁移与足端碰撞修复记录 |
