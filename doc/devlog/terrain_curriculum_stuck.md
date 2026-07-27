# Kuavo-S45-Rough 地形课程（terrain level）无法提升：原因分析与解决方案

> 分析对象：`logs/rsl_rl/kuavo_s45_velocity/2026-06-24_15-44-31/`（CNN 策略，S45-Rough，forward-only 速度命令）。
> 结论一句话：**terrain level 卡在 3.4/10 不是课程逻辑的 bug，而是 reward 设计把策略推向"站着别动"的局部最优 → 机器人几乎不前进 → 单 episode 位移过不了 4 m 的晋升阈值 → 课程在 level 3.4 反复振荡。** 课程冻结是症状，reward 设计是病根。

---

## 0. 现象（训练日志证据）

源自 `events.out.tfevents.1782287087.hitcsc.259509.0`：

| 指标 | 训练初期（iter ~11000） | 末期（iter ~17900） | 趋势 |
| --- | --- | --- | --- |
| `Curriculum/terrain_levels` | 3.51 | **3.49** | **冻结**（min 2.45 / max 3.62） |
| `Train/mean_episode_length` | 11.4 | ~779（max 812） | 上升后 plateau（上限 ~1000） |
| `Metrics/twist/error_vel_xy` | 0.018 | **0.57** | **变差**（速度跟踪失败） |
| `Metrics/twist/error_vel_yaw` | 0.014 | **0.87** | **变差** |
| `Train/mean_reward` | +0.4 | +30.5 | ↑ 但靠"活更久累积"，非更会走 |
| `Episode_Termination/fell_over` | — | 0.63 | ~29% episode 因摔倒截断 |
| `Policy/mean_std` | 1.32 | 1.30 | **恒定**（从不衰减） |

最反常的两个信号：
1. reward 在涨，但速度跟踪误差也在涨 → **reward hacking**（策略在钻奖励空子，而非学会走路）。
2. `terrain_levels` 全程在 3.4–3.62 之间窄幅振荡，**从未稳定上升**。

---

## 1. 课程机制：terrain level 是怎么升降的（代码）

课程函数 `terrain_levels_vel`（`src/omni_gs_playground/tasks/velocity/mdp/curriculums.py:30`）：

```python
distance = ||root_link_pos_w[env_ids, :2] - env_origins[env_ids, :2]||   # 本 episode 走出的位移
move_up   = distance > terrain_generator.size[0] / 2                     # > 8/2 = 4 m → 升级
move_down = distance < ||command_xy|| * max_episode_length_s * 0.5       # < ||cmd_xy||*20*0.5 = ||cmd_xy||*10 → 降级
move_down *= ~move_up
terrain.update_env_origins(env_ids, move_up, move_down)
return mean(terrain.terrain_levels)
```

代入实际配置（`params/env.yaml`）：
- `terrain_generator.size = (8.0, 8.0)` → **晋升阈值 = 4 m**。
- `episode_length_s = 20.0` → 降级阈值 = `||cmd_xy|| × 10`。
- 速度命令范围（`env.yaml` / `velocity_env_cfg.py:581`）：`lin_vel_x ∈ [0,1]`、`lin_vel_y ∈ [-1,1]`、`ang_vel_z ∈ [-1,1]`，`||cmd_xy||` 典型 ≈ 0.5–0.7 → 降级阈值 ≈ **5–7 m**。

关键：由于降级阈值（5–7 m）> 晋升阈值（4 m），两者之间的"既不升也不降"区间对大多数命令为空。课程实际退化为一条硬规则：

> **单 episode 走出 > 4 m → 升一级；否则 → 降一级。**

地形是 `num_rows=10 × num_cols=10`（`env.yaml`），行 = 难度等级（0 最易≈flat，9 最难），列 = 地形类型。level 3.4/10 意味着机器人只能稳定应对低难度地形（约 6–7 cm 台阶量级，见 §4），一上 level 4+ 就摔倒降回。

---

## 2. 根因链：reward → 速度跟踪失败 → 位移不足 → 课程冻结

把"位移 < 4 m"拆开看：`distance ≈ 实际前进速度 × 存活时长`。

- `mean_episode_length ≈ 779` 步 ≈ 15–16 s（50 Hz）。
- 若机器人按命令（~0.5 m/s）正常走，16 s 应位移 ≈ 8 m → 轻松 > 4 m → 持续升级。
- 但 `error_vel_xy = 0.57`，说明**机器人几乎没在前进**（实际净速度可能 ~0.2 m/s）→ 16 s × 0.2 ≈ 3.2 m **< 4 m** → 被降级。
- 偶尔在简单地形上存活满 20 s 且挪动 > 4 m → 升级；升到 level 4+ 又摔倒 → 降回 → **在 3.4 反复振荡**。

**所以课程冻结 = 位移卡在 4 m 阈值附近 = 速度跟踪失败的直接几何后果。** 而 `error_vel` 随训练**变大**，证明策略不是"学得慢"，而是**主动放弃跟踪**——这就是 reward 设计的局部最优在起作用。

---

## 3. reward 变化分析（日志 + 权重）

### 3.1 各分量末期实测贡献（`Episode_Reward/*`，按 |贡献| 排序）

| 分量 | 末期均值 | 权重（`env_cfgs.py`） | 是否需要走路 |
| --- | --- | --- | --- |
| `track_linear_velocity` | **+2.12** | 5.0（std=0.5） | ✅ 必须 |
| `track_default_arm_pos` | **+1.41** | 3.0（`env_cfgs.py:397`） | ❌ 手臂放默认即可 |
| `track_angular_velocity` | +1.13 | 3.0 | ❌ 命令常为 0，站着就拿满 |
| `pose` | +0.39 | — | ❌ |
| `feet_air_time` | +0.07 | 2.0 | 半 |
| `action_smoothness_l2` | **−1.94** | −0.01（`env_cfgs.py:364`） | — |
| `action_rate_l2` | −0.41 | −0.005 | — |
| `illegal_dof_barrier` | −0.24 | −0.1 | — |
| `joint_pos_limits` | −0.21 | −10.0 | — |
| `dof_vel_l2` / `dof_torques_l2` / `joint_acc_l2` | −0.17 / −0.17 / −0.16 | — | — |
| `fly` | −0.13 | **−10.0**（`env_cfgs.py:461`） | — |
| `is_terminated` | −0.09 | **−200.0** | — |

Gross 正奖励 ≈ +5.1，Gross 负奖励 ≈ −3.6，净 ≈ +1.5/step。

### 3.2 三个结构性病灶

**病灶 A：静态局部最优。** Gross 正奖励里约 **2.9（≈57%）不需要走路**就能拿：`track_default_arm_pos`（+1.41，手臂 14/26 DoF 放默认位）+ `track_angular`（+1.13，forward-only 命令下 yaw 常为 0，站着不转就拿满）+ `pose`。而 `track_linear`（+2.12）是唯一真正需要前进的项。**站着不动的收益/风险比远高于走路。**

**病灶 B：风险严重不对称。** 动起来的惩罚极陡：`fly −10`、`feet_too_near −5`、`is_terminated −200`、`joint_pos_limits −10`。一旦尝试走路触发这些，损失远超 `track_linear` 的收益 → 策略避险 → 不走 → `error_vel` 维持 0.57。

**病灶 C：动作正则吃掉跟踪信号。** `action_smoothness_l2`（−1.94）是全场绝对值最大的单项，几乎整笔抵消 `track_linear`（+2.12）；加上 `action_rate_l2`（−0.41），动作正则合计 −2.35 > track_linear。**"学会走路"的净奖励接近 0**，梯度被"减小 jerk"主导。

> 三者叠加：策略发现"站着把手臂摆好、避免一切风险动作"是收益最高的稳定解 → 不前进 → 位移 < 4 m → 课程冻结。`mean_reward` 上升只是 episode 变长带来的累积，**不代表任务在变好**。

---

## 4. 地形设置分析（代码）

`terrain_generator`（`env.yaml`）：`curriculum=true`，10×10，每格 8×8 m，`border_width=20`。子地形与占比：

| 子地形 | 占比 | 关键难度参数 |
| --- | --- | --- |
| flat | 0.30 | — |
| pyramid_stairs | 0.20 | step_height [0.02, 0.14]，step_width 0.32 |
| pyramid_stairs_inv | 0.15 | 同上（下行台阶） |
| tilted_grid | 0.10 | tilt 24°，height 0.35 |
| box_random_grid | 0.10 | grid_height [0.02, 0.45] |
| narrow_beams | 0.05 | beam_width [0.18, 0.45] |
| stepping_stones | 0.05 | stone_size [0.35, 0.75] |

- 行方向（难度）线性缩放：level 3.4/10 时，台阶量级 ≈ `0.02 + (3.4/9)×(0.14−0.02) ≈ 0.065 m`（约 6.5 cm）。
- 这对 **std=1.3、每步注入 ≈0.07–0.13 rad 噪声**的策略而言已经接近极限——在 level 0–3（flat + 极浅台阶）能存活，level 4+（台阶 >8 cm、tilted_grid、box grid）就站不稳 → 摔倒降级。
- 即便 reward 调好，`pyramid_stairs_inv`（0.15）、`tilted_grid`（0.10）这类连续起伏地形对纯本体感知 + 42×42 深度（CNN）的策略本就偏难，课程在 3–4 级卡住有一定合理性；但**当前主要矛盾是策略不前进（位移问题），而非地形太难（生存问题）**。

---

## 5. 放大器：std 为什么不衰减

`Policy/mean_std` 全程恒定 1.3 是上述问题持续存在的物理放大器：

- 分布为状态无关可学习 scalar std（`distribution.py:166`），下界 clamp 仅 1e-6，**没有任何机制托住它**。
- checkpoint 实测 `std_param`（26 维）：iter 11000 mean=1.315 → iter 48000 mean=1.228，**37000 iter 仅漂移 0.087**，且每维 0.68–1.88、维间 std 恒为 0.43——策略在任何一维都没收敛。
- 机制：PPO 损失 `−entropy_coef·entropy`（`ppo.py:321`）把 std 持续往上推；而 advantage 归一化（`ppo.py:212`，`E[A]=0`）使状态无关 std **几乎拿不到策略损失的一阶下拽梯度** → std 被熵项钉在 1.3。
- 反算自洽：`26×(0.5·log2πe + log1.3) ≈ 43.7`，与日志 `Loss/entropy≈42` 吻合。
- 后果：每步关节目标噪声 ≈ `action_scale(≈0.05–0.1) × 1.3 ≈ 0.07–0.13 rad` → 无法锁定步态 → 间歇摔倒 → 持续压低 episode_length 与位移。

---

## 6. 解决方案（按优先级）

### P0｜让 std 衰减（直接减少摔倒，解锁位移）
- `entropy_coef` 0.01 → **0（或 0.001）**；`init_std` 1.0 → **0.5**。
- 或改 `std_type="log"` 并收紧 `std_range`（如 `(0.05, 2.0)`），或换状态相关 std（`HeteroscedasticGaussianDistribution`）。
- **此条不动，reward 再调也难收敛**——噪声永远在打断步态。

### P1｜破除静态局部最优（让策略愿意走）
- `track_default_arm_pos` 权重 **3.0 → 0.5–1.0**，并下调 `joint_deviation_arms`；让走路相关奖励回到主导。
- 给 `fly / feet_too_near / feet_stumble` 加 `command_threshold` 门控（命令速度 >0.1 才生效），或整体软化（`fly −10→−2`、`feet_too_near −5→−1`）。
- 可选：把 `track_linear_velocity` 权重上调（5.0→8.0）或收窄 std（0.5→0.25），强化前进信号。

### P2｜松绑动作正则
- P0 见效后若 `action_smoothness` 仍主导，再下调：`action_acc_l2 −0.01→−0.002`、`action_rate_l2 −0.005→−0.001`，确保 `track_linear` 净信号为正。

### P3｜给课程解冻的缓冲
- 课程逻辑本身合理（依赖"会走"），P0–P2 见效、位移稳定 >4 m 后 level 自然解冻。
- 若想加速验证，可临时把晋升阈值 `size[0]/2`（4 m）调小，或在训练初期降低 `step_height_range` 下限，让机器人先在更易地形上累积 timeout 成功、建立前进行为，再逐步加难。

---

## 7. 建议的验证实验（最低成本）

1. **std 实验**：仅把 `kuavo_s45_ppo_runner_cfg` 的 `entropy_coef` 设 0，其余不动，跑 ~2000 iter。观察 `Policy/mean_std` 是否跌破 1.0、`Curriculum/terrain_levels` 是否突破 3.62。一锤定音 P0。
2. **reward 实验**：在 P0 基础上落 P1（降 `track_default_arm_pos` + 软化 `fly/feet_too_near`），观察 `error_vel_xy` 是否开始下降、`terrain_levels` 是否稳步爬升。
3. 判据：`terrain_levels` 在 5k iter 内稳定 >5.0，且 `error_vel_xy < 0.3`，即认为课程解冻、问题根除。

---

## 附：关键文件与行号
- 课程函数：`src/omni_gs_playground/tasks/velocity/mdp/curriculums.py:30`（`terrain_levels_vel`）
- S45 reward 移植：`src/omni_gs_playground/tasks/velocity/config/kuavo/env_cfgs.py:283`（`_apply_s45_emp_rewards`）、`:494`（`_separate_depth_observations`）
- 速度命令与 depth 相机：`src/omni_gs_playground/tasks/velocity/velocity_env_cfg.py:536`（`make_kuavo_velocity_env_cfg`）
- RL/CNN 配置：`src/omni_gs_playground/tasks/velocity/config/kuavo/rl_cfg.py:93`（`kuavo_s45_ppo_runner_cfg`）
- 分布定义：`packages/rsl_rl/rsl_rl/modules/distribution.py:132`（`GaussianDistribution`）
- PPO 损失：`packages/rsl_rl/rsl_rl/algorithms/ppo.py:321`
- 训练日志：`logs/rsl_rl/kuavo_s45_velocity/2026-06-24_15-44-31/events.out.tfevents.*`、`params/{env,agent}.yaml`
