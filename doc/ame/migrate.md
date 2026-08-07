# AME 迁移总结：Kuavo-S54-AME

> 记录 AME-Locomotion（Isaac Lab + Unitree G1）的 AME 地形感知策略迁移到本项目
> MJLab/MuJoCo 的 Kuavo-S54（27 关节）的全量改动，以及与 `Kuavo-S54-Rough`（DeFM）
> 任务的奖励对比。奖励部分为本文重点。

## 一、改动总览

AME 不是直接拷贝（Isaac Lab 与 MJLab/MuJoCo 不兼容），而是按 `MIGRATION_GUIDE.md`
的精神移植：网络结构不变、奖励设计不变，只重映射机器人名称/传感器。

| 文件 | 改动 |
| --- | --- |
| `packages/rsl_rl/rsl_rl/models/ame_model.py` | **新增** `AMEModel(MLPModel)`，忠实移植 `ActorCriticEncoder`：CNN + Multi-Head-Attention（proprio 为 query，CNN 地形特征为 k/v），`map_scan_dim=(33,21,3)`、`mha_dim=64`、`num_heads=16`、`cnn_downsample=True`、`attach_global=False`；`self.cnns` 支持 actor/critic 共享地形编码器；`as_jit()/as_onnx()` 导出。 |
| `packages/rsl_rl/rsl_rl/models/__init__.py` | 注册 `AMEModel` 到 `__all__`。 |
| `packages/rsl_rl/tests/models/test_ame_model.py` | **新增** 单测（结构/共享/JIT/ONNX，8 条）。 |
| `src/omni_gs_playground/tasks/velocity/mdp/observations.py` | **新增** `elevation_map(env, sensor_name="terrain_scan", noise, z_min=-1.2, z_max=0.0)`：`terrain_scan` 射线 hit 相对 base 转机器人 yaw 系 → `[B, N*3]`；miss 射线显式 `z=z_min`；`noise=True` 时 z 加 `randn*0.03`。 |
| `src/omni_gs_playground/tasks/velocity/mdp/rewards.py` | **新增** 3 个奖励函数：`air_time_variance_penalty`（双足腾空/触地时间方差惩罚）、`joint_coordination_rel`（跨体关节协调）、`applied_torque_limits`（超出执行器力矩上限计数）。 |
| `src/omni_gs_playground/tasks/velocity/terrains/ame_terrains.py` | **新增** `AME_ROUGH_TERRAINS_CFG`（8 类训练地形）+ 自定义 `HfConcentricGapTerrainCfg`（同心间隙高度场，AME 无 mjlab 等价物的唯一子地形）。 |
| `src/omni_gs_playground/tasks/velocity/terrains/__init__.py` | **新增** 导出 `AME_ROUGH_TERRAINS_CFG` / `HfConcentricGapTerrainCfg`。 |
| `src/omni_gs_playground/tasks/velocity/config/kuavo/env_cfgs.py` | **新增** `_apply_s54_ame_rewards()`（整体替换奖励表）与 `kuavo_s54_ame_env_cfg()`（去掉 depth、加 `terrain_scan`、重建 actor/critic 观测、改命令范围、换训练地形）。 |
| `src/omni_gs_playground/tasks/velocity/config/kuavo/rl_cfg.py` | **新增** `RslRlAmeModelCfg` 与 `kuavo_s54_ame_ppo_runner_cfg()`（actor/critic 均 `AMEModel`，`share_cnn_encoders=True`，`entropy_coef=0.008`，`obs_normalization=False`）。 |
| `src/omni_gs_playground/tasks/velocity/config/kuavo/__init__.py` | 注册任务 `Kuavo-S54-AME`（train + play env + rl_cfg + `VelocityOnPolicyRunner`）。 |
| `CLAUDE.md` | 「已注册任务」表新增 AME 行。 |

### 环境侧关键差异（相对 S54-Rough）

- **观测**：无 depth 相机。`terrain_scan`（`RayCastSensorCfg`，`GridPatternCfg(size=(1.6,1.0), resolution=0.05)` → 33×21 射线，`ray_alignment="yaw"`，`include_geom_groups=(0,)`）替代 depth；`height_scan`（elevation_map）必须是 actor/critic 观测组**最后**一项。
- **actor obs**：`[base_ang_vel, projected_gravity, command, joint_pos, joint_vel, actions, height_scan(noise=not play)]`（90 维 proprio + 2079 地图）。
- **critic obs**：额外 `base_lin_vel`，共 93 维 proprio，`height_scan(noise=False)`。
- **history**：`history_length=1`（AME 无观测历史）。
- **命令**：`heading_command=True`、`heading=(-π,π)`、`lin_vel_x=(0,1.5)`、`ang_vel_z=(-1,1)`、`rel_standing_envs=0`、`rel_heading_envs=1`。
- **地形**：训练用 `AME_ROUGH_TERRAINS_CFG`（`curriculum=True`）；play 用 5×5 非课程（`border_width=10`）。
- **终止**：沿用 S54 的 `time_out` + `bad_orientation`（未加 AME 的 `illegal_contact` 终止，与项目 S54 惯例一致，非足接触已由 `undesired_contacts` 奖励惩罚覆盖）。
- **nconmax**：AME 工厂设为 `81`（基础 rough 为 128）。注释仍残留 "solver requires >= 217" 字样，与当前值不一致，为 hfield 分辨率降到 0.1 之前的旧注释，需以实际训练为准。

### 地形说明（`AME_ROUGH_TERRAINS_CFG`）

8 类子地形，比例与 AME `ROUGH_TERRAINS_CFG` 一致（0.1×6 + 0.2×2）：

| 子地形 | 类型 | 说明 |
| --- | --- | --- |
| pyramid_stairs / pyramid_stairs_inv | BoxPyramidStairs / BoxInvertedPyramidStairs | `step_height_range=(0.05,0.2)`、`step_width=0.3`、`platform_width=3.0` |
| boxes | BoxRandomGrid | `grid_width=0.45`、`grid_height_range=(0.05,0.2)` |
| random_rough | HfRandomUniform | `noise_range=(0.02,0.10)`、`horizontal_scale=0.1` |
| hf_pyramid_slope / _inv | HfPyramidSloped | `slope_range=(0.0,0.4)`、`inverted` 分支 |
| hf_steppingstones | BoxSteppingStones | 石台 `±0.05`、`floor_depth=2.0` |
| hf_gaps | **自定义** `HfConcentricGapTerrainCfg` | 同心间隙/地面环 + 中央平台，`gap_width=(0.1,0.5)`、`ground_height_max=0.025`、`gap_depth=-2.0` |

**分辨率决策**：hfield 地形 `horizontal_scale` 用 0.1 m（AME 源码为 0.05）。原因是 MuJoCo
Warp 的 hfield 碰撞每个 geom pair 的接触点数上限 `MJ_MAXCONPAIR=50`：0.05 m 网格下机器人
摔平后躯干覆盖 >50 个单元 → 运行时碰撞溢出（`number of collisions >= 50`）。0.1 m 下摔平
躯干约 15 单元，安全。`elevation_map` 观测网格（0.05 m）独立、不受影响。

## 二、与 S54-Rough 的奖励对比（重点）

`_apply_s54_ame_rewards()` **整体替换** `cfg.rewards`，奖励表来自 AME `RewardsCfg` / `G1RoughEnvCfg.__post_init__`（非 FINETUNE 权重集），只重映射身体/关节名。新增非足接触传感器 `_AME_UNDESIRED_CONTACT_SENSOR`（`leg_[lr][1-5]_link` / `base_link` / `waist_yaw` / `zarm_[lr][1-7]_link`）。

### S54-Rough（DeFM）基线奖励表

S54-Rough 的 `kuavo_s54_rough_env_cfg()` 只做深度相机分辨率 + 分组，**不改奖励**，即用
`make_kuavo_velocity_env_cfg()` 基础表 + `_kuavo_rough_env_cfg` 的 `self_collisions`
（其中 `foot_gait`、`angular_momentum` 已被 Kuavo 基类移除）：

| term | func | weight | 说明 |
| --- | --- | --- | --- |
| track_linear_velocity | `track_linear_velocity` | 1.0 | std sqrt(0.25) |
| track_angular_velocity | `track_angular_velocity` | 1.0 | std sqrt(0.5) |
| body_orientation_l2 | `body_orientation_l2` | -1.0 | |
| pose | `variable_posture` | 1.0 | 逐关节 std（standing/walking/running） |
| body_ang_vel | `body_angular_velocity_penalty` | -0.05 | |
| is_terminated | `is_terminated` | -200.0 | |
| joint_acc_l2 | `joint_acc_l2` | -2.5e-7 | |
| joint_pos_limits | `joint_pos_limits` | -10.0 | |
| action_rate_l2 | `action_rate_l2` | -0.05 | |
| foot_clearance | `feet_clearance` | -1.0 | |
| foot_slip | `feet_slip` | -0.25 | |
| soft_landing | `soft_landing` | -1e-3 | |
| stand_still | `stand_still` | -4.0 | |
| self_collisions | `self_collision_cost` | -1.0 | `force_threshold=10.0` |

### AME 奖励表（21 项）

| term | func | weight | 说明 |
| --- | --- | --- | --- |
| is_terminated | `is_terminated` | -200.0 | |
| track_linear_velocity | `track_linear_velocity` | 2.0 | std sqrt(0.25) |
| track_angular_velocity | `track_angular_velocity` | 3.0 | std sqrt(0.25) |
| ang_vel_xy_l2 | `body_angular_velocity_penalty` | -0.05 | ROOT_BODY |
| body_orientation_l2 | `body_orientation_l2` | -2.0 | ROOT_BODY |
| undesired_contacts | `undesired_contacts` | -1.0 | 非足身体接触，threshold 1.0 |
| dof_torques_l2 | `joint_torques_l2` | -1.5e-7 | 受控关节 |
| dof_acc_l2 | `joint_acc_l2` | -1.25e-7 | 受控关节 |
| dof_vel_l2 | `joint_vel_l2` | -0.001 | 受控关节 |
| dof_pos_limits | `joint_pos_limits` | -1.0 | 受控关节 |
| dof_torques_limits | `applied_torque_limits` | -0.01 | `KUAVO_S54_CONTROLLED_JOINT_EFFORT_LIMITS` |
| action_rate_l2 | `action_rate_l2` | -0.01 | |
| feet_air_time | `feet_air_time_positive_biped` | 0.25 | threshold 0.6 |
| feet_air_time_variance | `air_time_variance_penalty` | -0.7 | 新增函数 |
| feet_slide | `feet_slide` | -0.1 | |
| feet_stumble | `feet_stumble` | -2.0 | |
| feet_too_near | `feet_too_near_humanoid` | -1.0 | threshold 0.2 |
| joint_coordination | `joint_coordination_rel` | -0.2 | 髋 pitch↔肩 pitch 交叉协调，新增函数 |
| joint_deviation_hip | `joint_deviation_l1` | -0.1 | `leg_[lr][12]_joint` |
| joint_deviation_arms | `joint_deviation_l1` | -0.3 | `zarm_[lr][1-7]_joint` |
| joint_deviation_waists | `joint_deviation_l1` | -1.0 | `waist_yaw_joint` |

### 差异分类

**1. 保留（函数与权重均不变）**

| term | S54-Rough | AME |
| --- | --- | --- |
| is_terminated | -200.0 | -200.0 |
| ang_vel_xy_l2（=body_ang_vel） | `body_angular_velocity_penalty` -0.05 | `body_angular_velocity_penalty` -0.05 |

**2. 保留但权重/参数调整**

| term | S54-Rough | AME | 变化 |
| --- | --- | --- | --- |
| track_linear_velocity | 1.0 | 2.0 | 权重 ×2 |
| track_angular_velocity | 1.0（std sqrt(0.5)） | 3.0（std sqrt(0.25)） | 权重 ×3 + std 收紧 |
| body_orientation_l2 | -1.0 | -2.0 | 权重 ×2 |
| joint_acc_l2 → dof_acc_l2 | -2.5e-7 | -1.25e-7 | 权重减半 |
| joint_pos_limits → dof_pos_limits | -10.0 | -1.0 | 权重 ÷10 |
| action_rate_l2 | -0.05 | -0.01 | 权重 ÷5 |
| foot_slip → feet_slide | -0.25 | -0.1 | 权重减小 |

**3. 移除（AME 不再使用）**

| term | S54-Rough 权重 | 说明 |
| --- | --- | --- |
| pose（`variable_posture`） | 1.0 | 由 3 个 `joint_deviation_*`（hip/arms/waists）替代 |
| foot_clearance | -1.0 | |
| soft_landing | -1e-3 | |
| stand_still | -4.0 | |
| self_collisions | -1.0 | |

**4. 新增（AME 独有）**

| term | 权重 | 说明 |
| --- | --- | --- |
| undesired_contacts | -1.0 | 非足身体接触惩罚（含 S54 腰部） |
| dof_torques_l2 | -1.5e-7 | 力矩 L2 |
| dof_vel_l2 | -0.001 | 关节速度 L2 |
| dof_torques_limits | -0.01 | 力矩超限计数 |
| feet_air_time | 0.25 | 正收益，鼓励腾空 |
| feet_air_time_variance | -0.7 | 双足腾空/触地不对称惩罚 |
| feet_stumble | -2.0 | 绊倒惩罚 |
| feet_too_near | -1.0 | 双脚过近惩罚 |
| joint_coordination | -0.2 | 髋 pitch ↔ 对侧肩 pitch 交叉协调 |
| joint_deviation_hip / arms / waists | -0.1 / -0.3 / -1.0 | 偏离默认姿态惩罚 |

### 设计意图解读

- **速度跟踪权重翻倍/三倍**：AME 更强调前向 + 转向速度，且角速度 std 收紧（sqrt(0.5)→sqrt(0.25)）使奖励对偏差更敏感。
- **`pose` → `joint_deviation_*`**：AME 用「偏离默认姿态的 L1 惩罚」替代「soft 姿态保持」，并按身体区（髋/臂/腰）分权，腰部惩罚最重（-1.0）。
- **新增足部与力矩质量项**：`feet_stumble`（-2.0）、`feet_air_time`（+0.25）等从 AME 奖励集移植，强化地形行走时的足部节律；`dof_torques_l2/limits`、`dof_vel_l2` 约束执行器能耗与超限。
- **`undesired_contacts` 而非终止**：AME 源码用 `illegal_contact` 终止，本项目沿用 S54 惯例只做 `bad_orientation` + `time_out`，非足接触改为奖励惩罚（-1.0），避免姿态外失败过早截断训练。
- **`feet_slide` 权重下调**：-0.25 → -0.1，配合 AME 更宽松的足部滑移容忍。

## 三、验证

- `scripts/list_envs.py --keyword AME` → 发现 `Kuavo-S54-AME` ✓
- `scripts/train.py Kuavo-S54-AME --help` 配置可构建 ✓
- `pytest packages/rsl_rl/tests/models/test_ame_model.py` → 8 passed ✓
- 地形编译 5246 geoms / 40 hfields；env 冒烟 actor obs `[1,2169]` 无 NaN，elevation_map z ∈ [-1.2,-0.85] ✓
- hfield 0.1 m 分辨率 + nconmax=81 下 1500 步混沌随机动作无碰撞溢出 ✓
