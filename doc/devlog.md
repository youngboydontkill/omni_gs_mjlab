# 开发日志 (Dev Log)

> 按时间倒序记录每次会话的改动：动机、根因、改了什么、验证、预期效果与回看指标。
> 文档默认中文；命令保持可复制；权重/参数改动一律给出「原值 -> 新值」。

---

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
