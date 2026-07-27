# 开发日志 (Dev Log)

> 按时间倒序记录每次会话的改动：动机、根因、改了什么、验证、预期效果与回看指标。
> 文档默认中文；命令保持可复制；权重/参数改动一律给出「原值 -> 新值」。

---

## 2026-07-26 新增 S54-Rough-Blind 盲策略任务

**动机**：参考 S54-Rough-CNN 的结构创建纯本体感知版本（无深度相机），用于对比有/无外感时的粗糙地形训练效果。

**改动**（3 文件）

| 文件 | 改动 |
|---|---|
| `config/kuavo/env_cfgs.py` | 新增 `kuavo_s54_rough_blind_env_cfg()`：基于 `_kuavo_s54_base_rough_env_cfg` + EMP 奖励，剥离 depth 传感器/观测/随机化事件，5 帧本体感知堆叠 |
| `config/kuavo/rl_cfg.py` | 新增 `kuavo_s54_rough_blind_ppo_runner_cfg()`：默认 `MLPModel`，无 CNN encoder，实验名 `kuavo_s54_blind_velocity` |
| `config/kuavo/__init__.py` | 注册 `Kuavo-S54-Rough-Blind` 任务 |

**验证**：
- `list_envs.py --keyword S54-Rough-Blind` → 可发现
- `play.py Kuavo-S54-Rough-Blind --agent zero --num-envs 1` → 正常启动

### feet_height 奖励

**动机**：盲策略无深度相机，无法"看到"障碍物，需要奖励项鼓励抬脚高度，帮助在粗糙地形上自动清除障碍。

**改动**

| 文件 | 改动 |
|---|---|
| `mdp/rewards.py` | 新增 `feet_height()`：测量足端 site 世界 Z 高度，离地时超过 `target_height` 的部分线性奖励，用命令幅度门控 |
| `config/kuavo/env_cfgs.py` | `kuavo_s54_rough_blind_env_cfg()` 中添加 `cfg.rewards["feet_height"]`，`weight=1.0`, `target_height=0.2` |

---

## 2026-07-26 S54 物理参数对齐 depth_loco_param.info + 腰部相机 pitch 修正

### 物理参数对齐

**动机**：`kuavo_s54_constants.py` 的 stiffness/damping/effort_limit 与真机部署使用的 `depth_loco_param.info` 不一致，导致仿真 PD 增益与力矩限制偏离实物。

**来源**：`/home/hitcsc/kuavo-ros-opensource/src/humanoid-control/humanoid_controllers/config/kuavo_v54/rl/depth_loco_param.info`

**改动**（`kuavo_s54_constants.py`）

| 关节 | stiffness(旧→新) | damping(旧→新) | effort_limit(旧→新) |
|------|------------------|----------------|--------------------|
| leg_[lr]1 (hip roll) | 60→48 | 6→5 | 101.6→100.0 |
| leg_[lr]2 (hip yaw) | 60→48 | 6→5 | 56.8→50.5 |
| leg_[lr]3 (hip pitch) | 80→68 | 6→5 | 105.6→100.0 |
| leg_[lr]4 (knee) | 95→68 | 6→6(不变) | 224→150.0 |
| leg_[lr]5 (ankle pitch) | 55→18 | 7.5→7.5(不变) | 91.6→70.0 |
| leg_[lr]6 (ankle roll) | 55→18 | 7.5→7.5(不变) | 68.4→50.0 |
| waist_yaw | 40→30 | 4→3 | 81.6→33.0 |
| zarm_[lr]1 | 20→30 | 3→3(不变) | 52.8→30.0 |
| zarm_[lr]2 | 20→30 | 3→3(不变) | 60→30.0 |
| zarm_[lr]3 | 20→15 | 3→3(不变) | 45.6→20.0 |
| zarm_[lr]4 | 20→30 | 3→3(不变) | 60→30.0 |
| zarm_[lr]5 | 15→15(不变) | 3→3(不变) | 11→14.1 |
| zarm_[lr]6 | 15→15(不变) | 3→3(不变) | 11→14.1 |
| zarm_[lr]7 | 15→15(不变) | 3→3(不变) | 11→14.1 |

**HOME_KEYFRAME 更新**：基础高度 0.925→0.965（匹配 `defaultBaseHeightControl`），腿关节角度对齐 `defaultJointState`（-0.4→-0.24, 0.69→0.5, -0.33→-0.26），手臂从全 0 改为参考 `defaultJointState`（l1=0.126, l2=0.1, l4=-0.27, 对称镜像）。

`KUAVO_S54_ACTION_SCALE` 自动重算（`0.25 × effort_limit / stiffness`），与参考 `actionScaleTest` 一致。

**验证**：`uv run python scripts/play.py Kuavo-S54-Rough-CNN --agent zero --num-envs 1` 启动正常。

---

## 2026-07-26 修正 S54-Rough-CNN 腰部相机 pitch 对齐真实外参

**动机**：S54-Rough-CNN 任务的深度相机 pitch 是 ~34°（继承自旧 S54 默认值），但真实机器人 `waist_camera_link` body 的 `quat=(0.868, 0, 0.497, 0)` 编码了 **59.6°** 俯角（R_y(59.6°)），二者相差约 25.6°，导致仿真中相机看到的场景与真实机器人不一致。

**根因**：`make_kuavo_velocity_env_cfg()` 设定 `CameraSensorCfg.quat = (0.624338, 0.331967, -0.331967, -0.624338)` 编码的是 S54 旧设计 pitch ~34°（0.593 rad）。`_kuavo_s54_base_rough_env_cfg()` 只覆盖了 `pos` 和 `parent_body`，未覆盖 `quat`，导致所有 S54 任务都使用这个偏浅的俯角。

此外 `_kuavo_rough_env_cfg()` 在 `if play:` 块之后添加了 `depth_camera_pitch` 域随机化事件，导致 **play 回放时相机 pitch 也会每 episode 随机化 ±8.6°**，观察到的 extrinsics 不固定。

**改动**

| 文件 | 改动 |
|---|---|
| `config/kuavo/env_cfgs.py:_kuavo_rough_env_cfg()` | play 分支新增 `cfg.events.pop("depth_camera_pitch", None)`，回放时相机外参固定 |
| `config/kuavo/env_cfgs.py:kuavo_s54_rough_cnn_env_cfg()` | 创建后直接设置 `depth_camera.quat = (0.682369, 0.185395, -0.185395, -0.682369)` |

**Quat 推导**

MJLab CameraSensorCfg quat 与 pitch θ 的关系（基于 `q_base = 0.5·(1, 1, -1, -1)` 复合 MuJoCo 光轴变换）：

```
q(θ) = 0.5·(cos(θ/2)+sin(θ/2), cos(θ/2)-sin(θ/2),
             -cos(θ/2)+sin(θ/2), -cos(θ/2)-sin(θ/2))
```

| pitch θ | 验算: w | 验算: x |
|---------|--------|---------|
| 34° | 0.5·(cos17°+sin17°) = 0.5·(0.9563+0.2924) = 0.62434 ✓ | 0.5·(0.9563-0.2924) = 0.33195 ✓ |
| 40° | 0.5·(cos20°+sin20°) = 0.64084 ✓ | 0.5·(cos20°-sin20°) = 0.29888 ✓ |
| 59.6° | 0.5·(cos29.8°+sin29.8°) = 0.682369 | 0.5·(cos29.8°-sin29.8°) = 0.185395 |
| 60° | 0.5·(cos30°+sin30°) = 0.6830 ✓ | 0.5·(cos30°-sin30°) = 0.1830 ✓ |

**验证**: `uv run python scripts/play.py Kuavo-S54-Rough-CNN --agent zero --num-envs 1` 启动正常，无报错。

**效果与预期**：
- 训练时深度相机 EM 域随机化继续生效（pitch 59.6° ± 8.6°），提升对安装角误差的鲁棒性
- 回放时 pitch 固定为 59.6°，与真实机器人 `waist_camera_link` 外参一致

---

## 2026-07-25 S54 机械模型迁移 + 新增 S54-Rough-CNN 任务

### S54 模型迁移

**动机**：当前项目使用的 S54 机械模型（XML/URDF/常量）与 `kuavo-ros-opensource` 仓库中的正式模型不同步，存在踝关节力矩偏低（57→68.4 N·m）、腰部相机缺失、头部相机结构过简、足部缺少 toe/heel 碰撞体等问题。

**来源**：`/home/hitcsc/kuavo-ros-opensource/src/kuavo_assets/models/biped_s54`

**改动**

| 文件 | 改动 |
|---|---|
| `assets/robots/kuavo/biped_s54/xml/biped_s54.xml` | 整体替换。关键差异：踝关节 actuator `ctrlrange` (30→68.4)、腰部新增 `waist_camera_link` body（含 camera）、头部相机重构为 `head_camera_base_link` + `head_camera_depth_link`、新增 `head_radar_link` 和足部 FT sensor、`waist_yaw` body inertial 修正（mass: 20.8448→20.7962） |
| `assets/robots/kuavo/biped_s54/urdf/biped_s54.urdf` | 整体替换。关键差异：踝关节 effort (30→68.4)、新增足尖/足跟碰撞球体（`ll_foot_toe`, `ll_foot_heel` 等 12 个 link）、uncomment `dummy_link` |
| `assets/robots/kuavo/biped_s54/meshes/` | 新增 5 个 STL（`head_camera_base`, `head_camera_depth`, `head_radar`, `waist_camera_base`, `waist_camera_depth`） |
| `assets/robots/kuavo/biped_s54/kuavo_s54_constants.py` | `KUAVO_S54_LEG_6_ACTUATOR.effort_limit`: 57.0→68.4（自动联动 `KUAVO_S54_ACTION_SCALE` 踝关节 scale: 0.259→0.311） |
| `tasks/velocity/config/kuavo/env_cfgs.py` | `_kuavo_s54_base_rough_env_cfg` 的 `depth_camera_pos`: (0.0987,0,-0.028449)→(0.09538,0,-0.01491)，与新 XML 的 `waist_camera_link` body 对齐 |

**修改后的 S54 腰部相机外参**：

| 参数 | XML（MuJoCo 原始相机） | MJLab CameraSensorCfg（训练/回放用） |
|---|---|---|
| 父级 body | `waist_yaw` | `robot/waist_yaw` |
| 位置 | body: (0.09538, 0, -0.01491) 相对 waist_yaw; 内含 camera: (0.01229, 0.02375, 0.01452) 相对 body | (0.09538, 0.0, -0.01491) — 对应 waist_camera_link body 位置 |
| 姿态 | body quat (0.868,0,0.497,0) = R_y(~59.6°); camera xyaxes="0 -1 0 0 0 1" | quat (0.624338, 0.331967, -0.331967, -0.624338) — 默认 MJCF 光轴变换 |
| FOV | 80° | 68.0°（Orbbec Gemini 335L） |

### 新增 Kuavo-S54-Rough-CNN 任务

**动机**：S45-Rough 的 EMP 奖励体系已证明对粗糙地形鲁棒性有效，需迁移到 S54 机器人上配合腰部相机 CNN 视觉策略。

**改动**

| 文件 | 改动 |
|---|---|
| `tasks/velocity/config/kuavo/env_cfgs.py` | ① `_apply_s45_emp_rewards()` 新增 `controlled_joints` 参数（默认 `S45_CONTROLLED_JOINTS` 向后兼容） ② 新增 `kuavo_s54_rough_cnn_env_cfg()` |
| `tasks/velocity/config/kuavo/rl_cfg.py` | 新增 `kuavo_s54_rough_cnn_ppo_runner_cfg()` — CNN encoder, non-shared, init_std=0.5, entropy_coef=5e-3 |
| `tasks/velocity/config/kuavo/__init__.py` | 注册 `Kuavo-S54-Rough-CNN`，`runner_cls=VelocityOnPolicyRunner` |

**验证**：
- `list_envs.py --keyword Rough` → 已列出 `Kuavo-S54-Rough-CNN`（#10）
- `play.py --agent zero` → S54-Rough-CNN、S54-Rough、S54-Head-CNN-Rough、S54-Flat 均正常初始化（terrain、CUDA、sensor context 无误）

---

## 2026-07-25 S45 Distill Fine-tune 漂移修复

**现象**：`2026-07-24_17-03-13` distill student 的平均 BC loss 已降至约 `0.019`，但 play 未稳定复现 teacher 落足；后续 fine-tune 总 reward 提高，实际却出现手臂高抬和足部踩边。

**日志证据与根因**

1. Fine-tune 的 `Loss/behavior` 从 `0.027` 升至 `0.351`，teacher regularizer 从 `0.2` 退火至约 `0.031`，策略已明显离开 teacher。
2. 配置要求 `init_std=0.15`，但首条日志和 `model_0.pt` 都约为 `0.5`。PPO 加载 distill checkpoint 时暂存的 distribution `state_dict` 与模型参数共享张量引用，actor load 原地覆写后无法恢复配置方差。
3. Distill 使用腿 `1.5`、手臂 `0.5` 的 action 权重，原 fine-tune 却对 26 维等权，未保护落足相关腿部动作。
4. `edge_contact` 长期仅约 `-3e-4`，远小于速度跟踪项；原 `5cm x 5cm` 足底网格也只覆盖脚掌中心。
5. Fine-tune 的 `track_default_arm_pos` 已接近零饱和，`joint_deviation_arms` 恶化至约 `-0.36`，原 `-0.1` L1 成本不足以阻止上肢补偿。

**改动**

| 项目 | 原值 | 新值 |
|---|---:|---:|
| PPO distill load std | 实际导入 `0.5` | 深拷贝并保留配置 `0.15` |
| learning rate | `3e-4` | `1e-4` |
| entropy coefficient | `1e-3` | `5e-4` |
| BC coefficient | `0.2 -> 0.02 / 10k` | hold `0.3` 3k，再于 12k 退火到 `0.05` |
| Fine-tune BC action 权重 | 26 维等权 | 腿 `1.5`，手臂 `0.5` |
| arm L1 deviation | `-0.1` | `-0.3` |
| sole edge grid | `3x3`, `0.05x0.05m` | `9x5`, `0.16x0.08m` |
| edge reward weight | `-0.5` | `-2.0` |

- `edge_contact_penalty` 的高度差改为无量纲 `[0,1]` severity，并增加静态接触成本，避免脚停在边缘时因速度接近零而没有惩罚。
- `Kuavo-S45-Rough-Distill-Finetune` 改为固定地形分布的 5k 稳定阶段。
- 新增 `Kuavo-S45-Rough-Distill-Finetune-Curriculum`，用于从第一阶段 PPO checkpoint 继续训练 10k updates。
- PPO checkpoint 新增 `algorithm_num_updates`，续训时恢复 BC 退火进度；旧 PPO checkpoint 回退使用 `iter + 1`。
- `scripts/train.py` 加载 checkpoint 时显式使用当前训练 device，允许在 CPU smoke 中读取 GPU 保存的模型。
- 增加单测覆盖 action 加权、BC hold/decay、distill load 保留 std，以及 PPO resume 恢复 schedule。

**资源影响**：足底 ray 从每脚 9 条增至 45 条，1024 环境约增加 7.4 万条 ray query。需要通过 `Perf/collection_time` 与显存实测决定是否将环境数降到 512。

**训练与验收**：完整两阶段命令及 checkpoint 选择条件见 `doc/s45_distillation_training_guide.md`。旧 fine-tune run 已从错误的 `std=0.5` 出发，不建议继续训练；应从 distill `model_15000.pt` 重新启动。

**验证**：任务发现和两个 fine-tune train/play help 通过；`uv lock --check`、`compileall`、配置断言和 4 个 PPO 回归断言通过。使用 distill `model_15000.pt` 完成 CPU `1 env x 1 update` smoke，实测 `behavior=0.0057`、`behavior_coef=0.3000`、`mean_std=0.15`，固定阶段 curriculum 为 inactive。完整 `pytest` runner 因当前环境未安装 pytest 且网络受限而无法启动。

## 2026-07-24 Depth-to-Terrain 辅助蒸馏落地

**动机**：进一步核对观测几何后确认，完整 terrain scan 虽然覆盖身后，但 EMP teacher 实际使用的 `teacher_height` 已裁成前方 `7x9`，不含身后区域。原方案仍有两个问题：teacher height 与 student perspective depth 不是同构观测；两侧 global pooling latent 的直接对齐丢失空间位置，不能有效监督落足点。

**改动**

1. `packages/rsl_rl/rsl_rl/models/cnn_model.py`
   - 暴露 global pooling 前的 depth CNN 空间特征。
   - action 与空间辅助特征共用一次 CNN forward，避免辅助训练重复编码 depth。
2. `packages/rsl_rl/rsl_rl/algorithms/distillation.py`
   - 新增轻量 terrain decoder，将空间特征重建为 `7x9` 局部高度图。
   - 新增 masked Smooth L1 height loss 与相邻网格 height-gradient loss。
   - decoder 纳入 optimizer、梯度裁剪、多 GPU 同步及 checkpoint 保存/恢复。
   - global latent matching 与 terrain reconstruction 设为互斥，避免两种表征目标同时拉扯 encoder。
3. `src/omni_gs_playground/tasks/velocity/mdp/observations.py` 与 `env_cfgs.py`
   - 新增 `teacher_height_valid`，屏蔽 terrain ray miss。
   - 该标记只表示 target 有效，不冒充严格的 depth camera visibility。
4. `src/omni_gs_playground/tasks/velocity/config/kuavo/rl_cfg.py`

   | 参数 | 原值 | 新值 |
   |---|---:|---:|
   | `latent_loss_coef` | `0.1` | `0.0` |
   | `terrain_reconstruction_loss_coef` | - | `0.1` |
   | `terrain_gradient_loss_coef` | - | `0.05` |
   | `terrain_target_shape` | - | `(7, 9)` |

5. 新增定向测试，覆盖空间特征接口、mask 行为及辅助 loss 对 decoder/depth CNN 的梯度回传。
6. 新增使用指南 `doc/s45_distillation_training_guide.md`。

**设计边界**

- 本次是 SSR 思路的局部、单帧版本，不重建身后或历史地形。
- `teacher_height_valid` 不是基于相机内外参和 depth consistency 的 visibility mask。扩展监督范围前，应先实现时序 depth + 位姿补偿 + BEV memory。
- decoder 只用于训练，不进入 student export，不增加部署推理开销。
- distill reward 仍只记日志；落足闭环最终由 `Kuavo-S45-Rough-Distill-Finetune` 的 PPO reward 优化。

**验证**

- `uv lock --check` 通过。
- 修改文件 `compileall` 通过。
- 小型 TensorDict/CNN distill smoke 成功执行 forward、backward 和 optimizer step，并输出 `behavior`、`terrain_reconstruction`、`terrain_gradient`。
- 任务发现以及 distill 的 train/play help 通过；新增 loss 参数已出现在 CLI。
- 真实环境 1 iteration smoke 在设备选择阶段因当前容器无法初始化 NVML/GPU 而停止，尚未进入 MuJoCo 场景或新 loss 路径。
- 当前环境未安装 `pytest` / `ruff` executable，定向测试文件已补但未通过对应 runner 执行。

## 2026-07-24 Distill/BC 框架与 reward 设计优化

**动机**：对比 `logs/rsl_rl/kuavo_s45_distill_velocity` 与 `logs/rsl_rl/kuavo_s45_velocity/2026-07-22_16-42-01` 后，确认当前 distill 训练存在三个核心现象：`episode_length/mean_reward` 上升快但抖动大、各项 reward 数值看似更好但方差剧烈、play 中 student 没学到 teacher 的落足点控制。进一步分析后发现，问题不在“reward 权重本身”，而在 **纯 action BC 与现有 student reward 目标错位**：reward 只记日志，不进入 distill loss，因此无法直接约束 rough 上的接地/落足恢复行为。

**结论**
1. distill 阶段本质上是纯 on-policy action BC，当前 reward 不参与反向传播。
2. `mean_reward` 快涨且抖动大，主要来自 teacher 先验、无 value 平滑、课程/终止项离散冲击。
3. 只蒸馏 action 不足以学会 teacher 的落足点控制，缺少 terrain/latent 对齐与 teacher 接管衰减。
4. BC 与现有 student reward 不冲突，但如果不把 reward 重新接入优化，就会长期停留在“动作拟合好、闭环控制弱”的状态。

**改动**

1. `packages/rsl_rl/rsl_rl/algorithms/distillation.py`
   - teacher 接管改为退火式 DAgger-lite：

     | 字段 | 原值 | 新值 |
     |---|---|---|
     | `teacher_intervention_start` | `1.0` | `1.0` |
     | `teacher_intervention_end` | - | `0.05` |
     | `teacher_intervention_decay_updates` | - | `12000` |

   - action loss 改为 Huber，降低 teacher/student action 尖峰对 BC 的放大效应。
   - 增加 joint 加权，优先约束腿部动作，避免手臂/非关键关节稀释落足控制信号。
   - 曾增加 teacher height latent 与 student visual latent 对齐；后续已由上节的空间 terrain reconstruction 取代。

2. `packages/rsl_rl/rsl_rl/algorithms/ppo.py`
   - 增加从 distill checkpoint 进入 PPO + BC fine-tune 的桥接路径。
   - 保留衰减式 teacher action regularizer，让 reward 在 PPO 阶段真正进入优化。

3. `packages/rsl_rl/rsl_rl/storage/rollout_storage.py`
   - 扩展 rollout 侧缓存字段，支持 distill / fine-tune 的额外监督信号。

4. `packages/rsl_rl/rsl_rl/models/cnn_model.py`
   - 补齐 student 侧视觉 latent 暴露与辅助蒸馏接口。

5. `packages/rsl_rl/rsl_rl/models/emp_teacher_model.py`
   - teacher height latent 作为显式蒸馏目标输出。

6. `packages/rsl_rl/rsl_rl/modules/emp_modules.py`
   - 补齐 latent / auxiliary loss 所需的模块接口。

7. `src/omni_gs_playground/tasks/velocity/config/kuavo/rl_cfg.py`
   - 增加 distill / fine-tune 相关配置入口。

8. `src/omni_gs_playground/tasks/velocity/config/kuavo/env_cfgs.py`
   - Distill 阶段默认关闭 terrain curriculum 自适应推进，减少目标漂移。
   - 新增 `edge_contact` 作为 foothold 安全信号，为后续 PPO fine-tune 提供更直接的落足约束。

9. `src/omni_gs_playground/tasks/velocity/config/kuavo/__init__.py`
   - 新增 `Kuavo-S45-Rough-Distill-Finetune` 任务注册。

10. `scripts/train.py`
    - 打通 distill checkpoint 启动与任务参数流转。

**优化方向**
- distill 阶段：优先学“teacher 的落足策略 + 视觉/地形表征”，而不是只拟合 action 均值。
- fine-tune 阶段：把 reward 重新变成优化目标，用 PPO 纠正闭环稳定性。
- reward 设计：让 `edge_contact` 等地形相关信号服务于 PPO，而不是继续停留在日志项。

**验证**
- 已完成日志对比与根因分析，结论写入 `doc/distill_vs_rough_log_analysis.md`。
- 本轮未做完整训练回归，后续需补 `train.py --help`、任务注册发现和最小 smoke test。

## 2026-07-24 注册 Distill play 可视化入口

**动机**：`Kuavo-S45-Rough-Distill` 已有 student checkpoint（`logs/rsl_rl/kuavo_s45_distill_velocity/.../model_15000.pt`），但 `scripts/play.py` 仍按 PPO dataclass 写死 `asdict(agent_cfg)` / `load_cfg={"actor": True}`，无法加载蒸馏 student。

**根因**
1. 蒸馏任务 `rl_cfg` 是 plain dict（`student` / `teacher` / `Distillation`），不是 `RslRlOnPolicyRunnerCfg`。
2. distill checkpoint 键为 `student_state_dict` / `teacher_state_dict`，不是 PPO 的 `actor_state_dict`。
3. play env 保留 `lin_vel_y=(0,0)`，Viser 命令滑块会因零区间触发 AssertionError。

**改动**（`scripts/play.py`）
- 复用 train 侧 `_agent_get` / `_agent_to_dict`，兼容 dict 与 dataclass runner cfg。
- 蒸馏任务加载 `student=True`，不强制加载 teacher/optimizer。
- Viser 路径复用 zero-range command slider 补丁，固定 `lin_vel_y=0`。
- 训练好的策略缺省要求 `--checkpoint-file`。

**用法**
```bash
uv run python scripts/list_envs.py --keyword Rough-Distill
uv run python scripts/play.py Kuavo-S45-Rough-Distill --help
uv run python scripts/play.py Kuavo-S45-Rough-Distill \
  --checkpoint-file logs/rsl_rl/kuavo_s45_distill_velocity/2026-07-23_19-13-10/model_15000.pt \
  --num-envs 1 --viewer viser
```

**验证**
- `list_envs --keyword Rough-Distill` 可见 `Kuavo-S45-Rough-Distill`。
- `play.py Kuavo-S45-Rough-Distill --help` 正常。
- CPU headless smoke：构建 DistillationRunner，加载 `model_15000.pt` student，推理 3 步。

**说明**
- 任务本身此前已注册；本次补的是 play 加载/可视化路径。
- student 动作为 MJCF 顺序，play 直接用 student 策略，不需要 lab2mjcf 转换。

## 2026-07-19 激活 AMP 训练（Kuavo-S45-AMP-Rough）

**动机**：已有 Kuavo S45 重定向运动数据（`GMR/motion_data/kuavo_s45_locomotion_pkl/csv`，135 个 CSV 文件，30fps），以 Kuavo-S45-Rough（CNN）为基础启用 AMP 风格奖励训练。批次 4 的 scaffold（判别器 + PPO 钩子）从未被任何任务激活，本次打通全链路。

**改动**

1. **新增 `tasks/velocity/amp/` 模块**（4 文件）：
   - `motion_loader.py`：`MotionDataset` 从 CSV 加载运动数据，逐帧转换 AMP state `s_t = [v(3), ω(3), g(3), q(26), q̇(26)]`（61 维）→ 滑动窗口扁平序列 `[N, 244]`（seq_len=4）。数据验证：10,249 条序列，`range: [-5.3, 7.0]`。
     - 有限差分求 body-frame v/ω/q̇；CSV 的 28 dof 丢弃 head 两列（列 33-34）取前 26。
   - `amp_obs.py`：`AMPStateComputer` 从 MuJoCo `EntityData` 实时读取 `root_link_lin_vel_b / root_link_ang_vel_b / projected_gravity_b / joint_pos / joint_vel`。
   - `env_wrapper.py`：`AMPVecEnvWrapper(RslRlVecEnvWrapper)` 维护 per-env history buffer `[B, seq_len, 61]`，`step()` 注入 `extras["amp_obs"]`（`[B, 244]`）；reset 时 done env 整段填充当前帧。
   - `__init__.py`：公开导出四类。

2. **修改 `rl_cfg.py`**：
   - 新增 `RslRlAmpOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg)`：顶层 `amp_cfg: dict | None = None`，绕过 `RslRlPpoAlgorithmCfg` 无此字段的 `asdict` 瓶颈。
   - `kuavo_s45_amp_ppo_runner_cfg()`：基于 CNN S45 配置，`share_cnn_encoders=False`（避免与 AMP 互斥）。

     | amp_cfg 字段 | 值 |
     |---|---|
     | input_dim | 244（61×4） |
     | hidden_dims | (256, 256) |
     | reward_coef | 0.5 |
     | seq_len | 4 |
     | motion_data_dir | `/home/hitcsc/YX/GMR/motion_data/kuavo_s45_locomotion_pkl/csv` |

3. **修改 `runner.py`**：新增 `AMPVelocityOnPolicyRunner`：
   - 从 `train_cfg` 顶层 pop `amp_cfg` → 注入 `train_cfg["algorithm"]["amp_cfg"]`（PPO 接收）。
   - 从 obs manager 读已解析 joint_ids 构建 `AMPStateComputer`。
   - 用 `AMPVecEnvWrapper` 重包装 env。
   - `super().__init__()` 后 attach `AMPSampler(MotionDataset(...))`。
   - 无 `amp_cfg` 时降级为正常 `VelocityOnPolicyRunner`。

4. **修改 `config/kuavo/__init__.py` + `rl/__init__.py`**：注册 `Kuavo-S45-AMP-Rough` 任务，runner 用新类。

**config 流向**：
```
RslRlAmpOnPolicyRunnerCfg.amp_cfg
  → asdict() → agent_cfg["amp_cfg"] (已验证 ✓)
  → AMPVelocityOnPolicyRunner pop→注入 algorithm dict
  → PPO.construct_algorithm → PPO.__init__(amp_cfg=...)
```

**生效范围**：仅 `Kuavo-S45-AMP-Rough` 新任务启用 AMP；现有 8 个任务行为不变。

**不修改的文件**（复用已有能力）：
- `scripts/train.py` — 现有 `asdict` + `load_runner_cls` 直接支持
- `packages/rsl_rl/` — AMP 判别器/PPO 钩子已完成
- `.venv/` vendored mjlab — 不修改任何 vendored 代码

**显存影响**：AMPDiscriminator (256×256 MLP, input_dim=244) ≈ 0.2M 参数；`amp_obs` 每个 transition 缓存 `[B, 244]`，约 244×4B ×4096 ≈ 4 MB 额外。`MotionDataset` 全量 ~19 MB CPU 内存。

**验证**：
- ✅ `list_envs --keyword AMP` → `Kuavo-S45-AMP-Rough` 可见
- ✅ `train.py Kuavo-S45-AMP-Rough --help` → 正常构建
- ✅ `asdict(load_rl_cfg("Kuavo-S45-AMP-Rough"))` → `amp_cfg` 在 dict 顶层
- ✅ `MotionDataset` 加载 10,249 条序列，sample(4) → `[4, 244]`
- ✅ `AMPDiscriminator(device="cpu", input_dim=244)` 构造 + style_reward + discriminator_loss 冒烟

---

**动机**：论文 §III-E 使用 Adversarial Motion Priors 让策略步态更自然。本仓库缺乏 Kuavo 重定向运动数据集，做完整 AMP 训练的前提不成立。本批仅搭**骨架**（判别器 + PPO 集成点 + 文档），由 `amp_cfg=None` 短路，不改变任何现有任务行为。

**改动**

1. `packages/rsl_rl/rsl_rl/extensions/amp.py`（新）：
   - `AMPDiscriminator`：MLP trunk 输出单个 logit；MSE loss（式7）、quadratic style reward（式8）、梯度惩罚。
   - 独立 Adam optimizer（weight decay）。
2. `extensions/__init__.py`：导出 `AMPDiscriminator`。
3. `algorithms/ppo.py`：四个集成钩子，全由 `self.amp is None` 短路：
   - `__init__`：接受 `amp_cfg` → 构造 `AMPDiscriminator`；`_amp_reference_sampler = None` + `set_amp_reference_sampler()` setter。
   - cache_features + AMP 互斥检查（raise）。
   - `process_env_step()`：从 `extras["amp_obs"]` 取判别器观测 → style reward → 加权累加至 `transition.rewards`；观测缓存至 `transition.extra["amp_obs"]`。
   - `update()`：当 `amp_obs in batch.extra` 且 reference sampler 已设置时，训练判别器 + 独立 optimizer 步。
   - `train_mode()` / `eval_mode()` 覆盖 AMP。
4. `doc/amp_scaffold.md`（新）：AMP 观测格式（`s_t = v,ω,g,q,q̇` 拼接）、启用步骤（获得重定向数据 → sampler → env 返回 extras → cfg → attach）、与 DeFM cache 互斥说明。

**生效范围**：无——`amp_cfg` 不在任何 runner config 中出现，PPO 行为逐位不变。

**验证**：`uv run --with pytest pytest packages/rsl_rl/tests/algorithms/test_ppo.py` 9 passed（`amp=None` 场景全覆盖）。

---

## 2026-07-15（下午）迁移「Hiking in the Wild」· 批次 3：足部边缘接触惩罚（光线近似）

**动机**：论文 §III-C 用地形网格 dihedral 边缘检测 + 足部 volume points 的 Warp 点-网格穿透惩罚，避免踩台阶边缘打滑。mjlab 地形是 box/hfield 异构 MuJoCo geom，**无统一 trimesh、无运行时顶点/面、无 wp.Mesh**，无法忠实移植。改用已有 `RayCastSensor` 的 BVH 光线做**近似**。

**改动**

1. `mdp/rewards.py::edge_contact_penalty`（新）：每足下方一小簇向下光线，用两路几何信号近似「脚踩边缘」：
   - 邻接高度突变：簇内命中 `z` 的 max-min（跨台阶边缘/gap 时大）。
   - 法向偏离：`1 - n_z > normal_threshold` 的光线占比（脚在斜侧面而非平顶）。
   - `severity = height_cue + normal_cue·height_threshold`，按**接地**门控（只罚真正承重的脚）、按足世界线速 `(‖v‖+ε)` 缩放，近似论文 `r_vol=-Σ‖d_i‖·(‖v_i‖+ε)`。miss/NaN 全守卫。
2. `config/kuavo/env_cfgs.py::_apply_s45_emp_rewards`：每足加 3×3 向下 `feet_[lr]_edge_scanner`（5cm 跨度、2.5cm 分辨率），并挂 reward term `edge_contact`（`weight=-0.5`，保守）。
   - 与 `toe_touch` 互补：toe_touch 管「别踢立面」，edge_contact 管「落脚居中、别踩边缘」。

**生效范围**：走 `_apply_s45_emp_rewards` 的 S45 系列（Rough / DeFM-Rough / Flat；flat 无边缘但传感器仍在、severity≈0 零成本）。S54 系列不经此路径，不受影响。

**显存影响**：每足 +9 条向下光线（3×3），raycaster 张量极小可忽略；无 rollout 缓存维度变化。

**标注**：这是**光线法向近似**，非论文 Warp 点-网格穿透惩罚。

**验证**：
- `list_envs --keyword Kuavo`：8 任务正常注册。
- `edge_contact_penalty` 合成数据单测：平地→0、边缘+接地+运动→正、悬空→0、侧面法向→正、全 miss→0 且有限。

---

## 2026-07-15（下午）迁移「Hiking in the Wild」· 批次 2：MoE 策略模型

**动机**：论文 §I 用 Mixture-of-Experts（MoE-Loco 风格）处理高维视觉 + 多地形技能。本框架此前只有 MLP/CNN/DeFM 三类模型，补齐 MoE。

**改动**

1. `packages/rsl_rl/rsl_rl/models/moe_model.py`（新）：`MoEModel(CNNModel)`。
   - `_MoEHead`：`gate=softmax(GateMLP(x))`、`out=Σ_k gate_k·ExpertMLP_k(x)`（软加权，纯张量、无动态分支 -> JIT/ONNX 可 trace）。
   - MoE 头插在 **CNN encoder latent 与策略 MLP 之间**：`get_latent` 先走父类拿 `[1D‖cnn]`，再过 MoE 得固定宽度 latent；`self.mlp` 仍是普通 MLP，故分布 `init_mlp_weights` 与导出路径与父类完全一致。
   - encoder 共享（`self.cnns`）继承父类不变，`share_cnn_encoders` 可用。
   - `_TorchMoEModel`/`_OnnxMoEModel`：仿 CNN 导出包装，前向多一步 `self.moe(latent)`。
2. `models/__init__.py`：导出 `MoEModel`。
3. `config/kuavo/rl_cfg.py`：新增 `RslRlMoEModelCfg(RslRlModelCfg)`（带 `moe_cfg`）+ `kuavo_s54_head_moe_ppo_runner_cfg()`（复用 Head-CNN 的 env 与 CNN 配置，actor/critic 换 `MoEModel`，`share_cnn_encoders=False`）。默认 `num_experts=4`、`expert_hidden_dims=(256,)`、`gate_hidden_dims=(64,)`、`moe_output_dim=256`。
4. 注册新任务 `Kuavo-S54-Head-MoE-Rough`（`config/kuavo/__init__.py`），env 复用 `kuavo_s54_head_cnn_rough_env_cfg`。
5. 测试 `packages/rsl_rl/tests/models/test_moe_model.py`（新，9 例）：结构/门控归一/encoder 共享/JIT/ONNX 保真。

**显存影响**：`num_experts` 个专家 MLP 参数量线性增长，前向 FLOPs ≈ ×num_experts；encoder 不变，rollout 缓存维度不变。默认 4 专家、专家隐层较窄以控参数量（实测 actor ≈ 0.73M 参数）。

**验证**：
- `uv run --with pytest pytest packages/rsl_rl/tests/models/`：53 passed（含新增 9 例 MoE）。
- `list_envs --keyword Kuavo`：8 任务（新增 Head-MoE-Rough）。
- `train.py Kuavo-S54-Head-MoE-Rough --help` 正常构建 cfg。
- 真实 obs 形状（actor_depth `[B,1,42,42]`）冒烟：actor 输出 `[B,27]`、critic `[B,1]`、log_prob 有限。

---

## 2026-07-15（下午）迁移「Hiking in the Wild」训练 tricks · 批次 1

论文：*Hiking in the Wild: A Scalable Perceptive Parkour Framework for Humanoids*（arXiv 2601.07718）。本批只做 §III-B2 的**真实深度合成（F_sim 退化链）**，其余（MoE / 边缘惩罚 / AMP 骨架）分批跟进。

**动机**：现有 `depth_image_obs` 仅 `nan_to_num` + clip + 均匀噪声（obs 层 `Unoise ±0.1`），缺乏真实 RGB-D 传感器的 range 依赖噪声、双目失配白区、运动模糊与瞬时失效，深度策略 sim2real 鲁棒性不足。

**改动**

1. `mdp/observations.py::depth_image_obs`：新增 `corrupt`（默认 `False`）及一组合成参数。`corrupt=False` 时逐位等价旧实现（已单测验证）。`corrupt=True` 时在 **clip 之后、normalize 之前**按论文顺序作用于米制域：
   - `_range_gaussian_noise`：`z'=z+N(0,σ²)`，仅 `noise_range=[d_min,d_max]` 内加噪（σ 米制）。
   - `_disparity_white_regions`：按 `white_prob` 贴 `white_max_blocks` 个矩形块置 `far`（双目失配）。
   - `_gaussian_blur`：depthwise 可分离高斯卷积（运动模糊），按 `blur_prob` 逐 env 生效。
   - `_ood_dropout`：按 `ood_prob` 整帧替换为 `[near,far]` 随机深度（瞬时失效）。
   - 加噪后再次 `clamp(near,far)`，normalize 不越界。
2. `config/kuavo/env_cfgs.py::_separate_depth_observations`：在唯一深度分组入口挂保守默认，`corrupt` 绑定 actor 组 `enable_corruption`（play 分支已置 False -> 回放拿干净深度；critic 恒 `corrupt=False`）。

   | 参数 | 值 |
   |---|---|
   | noise_std / noise_range | 0.02 m / (0.15, 3.0) |
   | white_prob / max_blocks / block_size | 0.10 / 2 / (8,8) |
   | blur_prob / kernel / sigma | 0.20 / 3 / 0.8 |
   | ood_prob | 0.005 |

**生效范围**：所有走 `_separate_depth_observations` 的深度任务（S45-DeFM、S54-Rough(DeFM)、S45-Rough(CNN)、S54-Head-CNN-Rough）actor 分支；critic 与 play 均为干净深度。blind/flat 无 depth 不受影响。

**显存影响**：blur conv2d 与白区 mask 均为逐 env 42×42 小张量，rollout 缓存维度不变，显存基本无增量。

**验证**：`list_envs --keyword Kuavo` 7 任务正常注册；纯函数单测通过（corrupt=False 逐位等价旧实现、corrupt=True 输出有限且落在 `[near,far]`、normalize 落在 `[-1,1]`、全零概率退化为旧实现）。

---

## 2026-07-15

### A. 深度相机外参域随机化（仅 pitch）

**动机**：深度策略 sim2real 时相机外参固定，缺乏对真实安装 pitch 误差的鲁棒性。

**改动**（`src/omni_gs_playground/tasks/velocity/config/kuavo/env_cfgs.py`）：
- 新增 `from mjlab.envs.mdp import dr as mjlab_dr`。
- `_kuavo_rough_env_cfg()` 的 play 分支后、return 前加 `depth_camera_pitch` 事件：

  | 字段 | 值 |
  |---|---|
  | func | `mjlab_dr.cam_quat` |
  | mode | `reset`（每 episode 重采样） |
  | roll_range / yaw_range | `(0.0, 0.0)`（不动） |
  | pitch_range | `(-0.15, 0.15)` rad（±~8.6°） |
  | asset_cfg | `SceneEntityCfg("robot", camera_names=("depth",))` |

- `kuavo_s45_flat_blind_env_cfg` 中 `cfg.events.pop("depth_camera_pitch", None)`（blind 无 depth 相机，优雅降级）。

**机制**：`dr.camera.cam_quat` 把采样 RPY 扰动与**默认四元数**组合（非当前值），重复调用不累积。
**生效范围**：所有带 depth 相机的 Kuavo 任务（S45/S54-Rough/Flat、Head-CNN-Rough），训练与 play 模式都生效。
**验证**：`list_envs --keyword Kuavo` 7 任务正常注册；语法 OK。

---

### B. Kuavo-S45-Rough 踝关节 roll 扭动修复

**背景日志**：`logs/rsl_rl/kuavo_s45_velocity/2026-07-06_16-30-20`（CNN 版 S45-Rough，40000 iter）。

**根因**（数据支撑，详见各小节）：

1. **物理层 - 软踝 roll**：CST 配置 `leg_[lr]6_joint` kp=8 / kd=3 / effort=36 N·m、范围仅 ±0.262 rad（±15°），并联 4-bar 机构使侧向扰动直接变现为可见 roll 摆动；低阻尼无法快速收敛。sim/real 都软 -> 两边都抖。
2. **策略层 - 动作噪声发散**：`Policy/mean_std` 从 init 0.5 单调涨到 **1.377** 不收敛（异常发散，见 `doc/action_std_in_ppo.md` §4/§5.2）；`clip_actions=null` 不裁剪。对刚性关节（kp=100）噪声被位置环滤掉，对 kp=8 的踝 roll 直接表现为目标高频反转 -> 扭动。`mean_action_acc` 从 0.87 涨到 2.58（动作 jerk 全程上升）即数值证据。
3. **奖励层 - 踝 roll 零成本**：`dof_torques_ankle_l2` 权重 -1e-5（reward≈-0.001 形同虚设）、无踝专属 action_rate、`pose` leg_6 std=0.1（±5.7° 内不扣分）、`illegal_dof_barrier` 仅边界生效。后半程 `error_vel_xy` 反弹（0.46->0.55）、`fly` 10× 增长 -> 策略在用踝/脚抖薅 `feet_air_time`，进入不稳区。
4. **为何 sim/real 都出现**：根因在策略侧（噪声 + 弱正则）× 软执行器，非单纯 sim2real gap。

**改动**（4 个杠杆，3 个文件）：

#### 杠杆 A - clip_actions（`config/kuavo/rl_cfg.py::kuavo_s45_ppo_runner_cfg`）
`null` -> `6.0`。卫生截断，对所有关节 offset（scale=0.25 -> ±1.5 rad）安全。

#### 杠杆 B - 踝专属正则（`env_cfgs.py::_apply_s45_emp_rewards`，S45 专属，S54 不受影响）

| 项 | 原值 | 新值 | 作用 |
|---|---|---|---|
| `action_rate_l2` | -0.005 | **-0.01** | 压动作一阶抖动 |
| `action_smoothness_l2` | -0.01 | **-0.02** | 压高频 jerk |
| `dof_torques_ankle_l2` | -1e-5 | **-1e-3** | 让踝 roll 力矩不再零成本 |
| `ankle_roll_vel_l2`（新增） | - | **-0.02** | `joint_vel_l2` scoped 到 `leg_[lr]6_joint`，直接惩罚 roll 角速度 |
| `pose` leg_6 std（walking/running） | 0.1 | **0.05** | 收紧踝 roll 姿态容忍（±5.7°->±2.86°） |

> `action_rate_l2` 无 asset_cfg、作用于全维 raw action，无法 scoped 到踝，故用 `joint_vel_l2` 替代作为扭动的直接代理量。B 全在 `_apply_s45_emp_rewards` 内，仅影响 S45 系列（Rough / DeFM-Rough / Flat-Blind）。

#### 杠杆 C - 状态相关 std（`rl_cfg.py`，根因修复，升级自"降 init_std"）
`GaussianDistribution` -> `HeteroscedasticGaussianDistribution`，`init_std` 0.5 -> **0.3**。
- 上次 init_std 已是 0.5，std 仍涨到 1.377 -> 证明单纯降 init_std 无效。
- 根因：状态无关标量 std 在 advantage normalization 下 policy-loss 净梯度≈0，被 entropy 单边顶高（仓库已诊断于 `doc/action_std_in_ppo.md`，DeFM 任务已用此法根治）。
- 切状态相关 std（actor 头输出 mean‖std）后 policy-loss 重新对 std 产生梯度。CNNModel 经 MLPModel 路径兼容；导出走 deterministic mean，ONNX/缓存不受影响。

#### 杠杆 D - 踝 damping（`kuavo_s45_constants.py::_ACTUATOR_PARAMS`）
`leg_[lr]5_joint` / `leg_[lr]6_joint` damping **3 -> 5**（kp 保持 8.0）。加速踝并联机构被动振荡收敛，不改位置环刚度，sim2real 风险最小。S45 全系列一致；S54 用独立 constants 不受影响。

**影响范围**：A+C 在 `kuavo_s45_ppo_runner_cfg` -> S45-Rough 及复用它的 S45-Flat（同软踝，安全改进）；B 仅 S45；D 仅 S45。

**验证**：
- 三文件语法 OK；`kuavo_s45_rough_env_cfg()` 与 `kuavo_s45_ppo_runner_cfg()` 均成功构建。
- reward 权重 / distribution class_name / clip_actions / 踝 kd=5 全部确认落地。
- `list_envs --keyword Kuavo` 7 任务正常注册。
- 未做完整训练（需重训），仅静态/配置构建 smoke test。

**重训回看指标**：
- `Policy/mean_std`：应收敛到 0.15–0.35，不再单调涨。
- `Episode_Metrics/mean_action_acc`：应从 2.58 显著下降。
- `Episode_Reward/ankle_roll_vel_l2` / `dof_torques_ankle_l2`：非零、有梯度。
- `Metrics/twist/error_vel_xy`：后半程不应再反弹。

---

## 2026-07-23 EMP Teacher 可视化链路修复

**现象**：Isaac Lab EMP teacher checkpoint 控制 MJLab 的 Kuavo-S45-Rough-Distill 时，关节目标明显错位，无法稳定控制；修复后 Viser 又在命令滑块初始化阶段触发 AssertionError。

**根因**

1. **关节置换表错误**：MJCF 顺序为 leg_l1..leg_l6、leg_r1..leg_r6、zarm_l1..zarm_l7、zarm_r1..zarm_r7；checkpoint 使用腿/手臂按关节编号交错的 USD 顺序。原实现使用另一套左右交错表，导致 joint_pos、joint_vel、历史 action 和输出 action 都映射到错误关节。
2. **Normalizer 被覆盖**：build_teacher() 把 checkpoint 的 actor_obs_normalizer._std 全部填成 1.0，破坏训练输入尺度。
3. **Height scan 预处理不完整**：teacher height 观测缺少 Isaac Lab 的最终 [-1, 1] clip。
4. **Viser 零范围滑块限制**：训练分布要求 lin_vel_y=(0, 0)，但 Viser 每个速度滑块要求最大值至少为 0.1，导致初始化断言失败。

**改动**

1. emp_modules.py 使用正确映射：
   mjcf2lab = [0, 6, 12, 19, 1, 7, 13, 20, 2, 8, 14, 21, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25]
   lab2mjcf = [0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 24, 3, 7, 11, 15, 19, 23, 25]
   term-major 的三个 130 维历史块分别重排 joint_pos、joint_vel、action，不转换为 frame-major。
2. build_teacher() 完整保留 checkpoint 的 _mean/_std/_var/count，移除 _std=1.0 和自动覆盖重估 normalizer 的逻辑。
3. teacher_height 增加 clip=(-1.0, 1.0)，保持 NaN/Inf 替换、裁剪区域和 offset=0.5 与 Isaac Lab 一致。
4. visualize_teacher.py 在 Viser 创建控件时临时放宽固定零速度轴，创建后恢复真实范围；每个 control step 强制固定轴归零，不改变训练配置或静态推理分布。
5. emp-teacher-model.md 修正 486 维输入说明为 command + term-major [15, 15, 130, 130, 130] + height scan；新增 test_emp_modules.py 覆盖置换表互逆性和历史关节块重排。

**验证**

- compileall 通过；置换表互逆性通过；checkpoint normalizer 与 .pt 中 actor_obs_normalizer._std 逐元素一致。
- Kuavo-S45-Rough-Distill CPU headless static rollout 200 步正常结束，无 NaN、shape 或关节索引错误。
- teacher_height 形状 [1, 63]，范围在 [-1, 1] 内。
- 真实 ViserServer 可创建 command GUI，创建后 lin_vel_y 仍为 (0, 0)。
- uv lock --check 和 git diff --check 通过。

---

## 2026-07-23 S45 Distill 训练链路与奖励复核

**结论**：`Kuavo-S45-Rough-Distill` 使用 `Distillation` 的纯行为克隆目标；环境
reward 仅用于 rollout/episode 日志，不进入 student 的反向传播。因此不能通过修改
distill task 的 reward 权重直接提高 student 的 imitation loss。奖励权重真正影响的是
teacher 的 PPO 训练任务 `Kuavo-S45-Rough`，以及蒸馏时用于观察 student 状态分布。

**修复**

1. `EMPTeacherModel` 成功从 checkpoint 构造后标记 `is_loaded`，
   `Distillation.construct_algorithm()` 将其同步到 `teacher_loaded`。此前 runner 会把已
   加载的 teacher 误判为未加载，并在 `learn()` 前拒绝运行。
2. distillation rollout 默认使用 student 的确定性 mean action。纯 MSE student 在初期
   采样 `std=0.5` 会迅速偏离 frozen teacher 可恢复的状态分布，得到无效的 teacher
   target；若要实验带噪 DAgger，可显式设 `student_rollout_stochastic=True`。
3. 梯度累积按实际累积 batch 数求均值，并优化最后不足 `gradient_length` 的窗口；原实现
   对 15 个 MSE 求和后直接反传，等效放大梯度约 15 倍，且会丢弃尾部 batch。
4. S45 EMP reward 的 `track_default_arm_pos` 从 `+3.0` 修正为 `+1.0`。14 个手臂
   DoF 下 `+3.0` 的稳定姿态回报足以压过移动探索，且与已有 P1 复盘的目标值不一致。
   速度跟踪（linear `+8.0`、angular `+3.0`）、姿态/接触/足端安全成本保持不变。
   当前实际踝力矩成本是 `dof_torques_ankle_l2=-1e-4`；较早日志中的 `-1e-3` 是未
   落地的调参目标，不应作为当前训练配置引用。

**验证重点**：`Loss/behavior` 应稳定下降；rollout 的 `Metrics/twist/error_vel_xy` 和
终止率不应因初期随机 action 激增。由于 reward 不参与 BC loss，
`Episode_Reward/*` 只用于确认 student 没有漂离安全状态，不能作为 student 收敛主指标。

**实际验证**

- `Kuavo-S45-Rough-Distill` 已通过任务注册、CLI help、runner 构造和 CPU 1 iteration
  smoke test（1 env × 24 steps）：teacher checkpoint 被识别为已加载，行为克隆 update
  正常完成，首轮 `Mean behavior loss=1.3505`，日志和 checkpoint 保存后正常退出。
- distillation 单测覆盖 loss 下降、teacher 参数冻结、整窗口及尾部窗口的梯度累积；
  `compileall`、`uv lock --check` 与 `git diff --check` 均通过。

---

## 2026-07-23 S45 Teacher Height Scan 仅地形命中

Isaac Lab 的 EMP `height_scanner` 显式只 raycast `/World/ground`。MJLab 原
`terrain_scan` 使用默认 `(0, 1, 2)` geom group，且 `exclude_parent_body=True`
仅排除挂载点 `base_link`，不会排除同一机器人的腿、脚和手臂；S45 robot mesh 全在
group 1，存在 height scan 命中自身的风险。

`Kuavo-S45-Rough-Distill` 的 teacher `terrain_scan` 现设置
`include_geom_groups=(0,)`。该过滤参数直接传入此传感器的 ray query；即使
SensorContext 为深度相机等其他传感器构建的 BVH 使用更宽的 group 并集，height scan
仍只接受 group 0 的交点。运行时读取 `terrain_scan._ray_geomid` 验证，所有有效命中
均为 `terrain_*` geom，所属 body 为 `terrain`，没有机器人命中。保留原有 base 挂载、
yaw 对齐、17×11 射线网格与 7×9 前方裁剪；深度相机和足端 raycast 不受影响。

---

## 2026-07-23 S45 蒸馏方案调研

当前 Distillation 已是“student 自己 rollout、teacher 在线打标签”的在线 BC，
但还不是完整 DAgger：没有 teacher 接管衰减、replay 聚合、恢复状态优先采样，也没有
teacher latent/value 或动作分布监督。`Now You See That` 一类 raw-pixel humanoid
方法的关键收益来自闭环状态分布和 privileged teacher 表征，而不只是把 MSE 换成另一个
动作损失。

建议路线为：先做 DAgger-lite（teacher env-mask 接管 + beta 衰减 + replay），再做
BC warm-start 后 PPO 微调，并将 action BC 作为逐渐减小的正则；确认感知表征成为瓶颈后，
再蒸馏 teacher height-map CNN/hidden latent。当前不要直接把 reward 加到 BC MSE：reward
没有 advantage/value 估计时量纲不匹配，不能提供可靠梯度；也不要长期使用高斯 student
rollout，以免状态分布脱离 teacher。

详细方案、配置建议和评估矩阵见 `doc/distillation_research.md`。
