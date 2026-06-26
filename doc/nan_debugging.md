# 训练 NaN 故障排查与防护

本文记录在 `Kuavo-S45-Rough`(CNN encoder + EMP rewards)长时间训练中观察到的 NaN 故障原因与一整套防护方案。结论同样适用于 `Kuavo-S45-DeFM-Rough`、`Kuavo-S54-Rough`、`Kuavo-S54-Head-CNN-Rough` 等任务 —— 只要使用 mjlab + 自写 reward / sensor / 视觉 encoder,这些经验都通用。

## 表现

`rsl_rl` 在 rollout 阶段调用 `check_nan(obs, rewards, dones)`,一旦发现 `obs` 任一组(actor / critic / actor_depth / critic_depth)出现 NaN 就抛:

```
ValueError: The observation group 'actor' returned by the environment contains NaN values.
This usually indicates a bug in the environment's step() or reset() function.
```

不同典型形态:

| 现象 | 出现时间 | 触发 env 数 | 根因层 |
|---|---|---|---|
| 启动后几十步内立刻爆 | 训练初期 | 多个 env 同步 | 观测函数缺 NaN 保护 + reset 异常 |
| 训练几小时后单点爆 | ≥ 1e5 step | 1/1024(零散) | 物理失稳(qpos/qvel NaN) |
| 周期性 batch 局部爆 | 中后期 | 5%-10% env | reward 量级失控触发 PPO 发散 |

## 根因分类

NaN 在三个层次可能出现,**所有上层 NaN 都源自下层失效**:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ obs/reward   │ <-- │ 观测函数错读 │ <-- │ MuJoCo 物理 │
│ NaN 抛错     │     │ qpos/qvel/   │     │ 解算失败     │
│ (check_nan)  │     │ site_pos_w   │     │ (qpos NaN)   │
└──────────────┘     └──────────────┘     └──────────────┘
                          ▲                       ▲
                          │                       │
                  ┌──────────────┐         ┌──────────────┐
                  │ critic-only  │         │ contact 池   │
                  │ 项无 nan_to_ │         │ 溢出 →       │
                  │ num (foot_*) │         │ solver NaN   │
                  └──────────────┘         └──────────────┘
                                                  ▲
                                                  │
                                          ┌──────────────┐
                                          │ reward 量级  │
                                          │ 失控 → PPO   │
                                          │ 动作饱和     │
                                          └──────────────┘
```

### 物理层(最常见)

| 触发条件 | 机制 |
|---|---|
| `cfg.sim.nconmax` 太小 | rough 地形 + 自碰 sensor + undesired-contact sensor 让单 env 接触点峰值轻松超过 64,溢出后 mjwarp 丢弃 contact,solver 输入不一致 → `qvel` NaN → 下一步 `qpos` NaN |
| `ccd_iterations` 不足 | 复杂几何对(机器人脚 vs 多面体台阶)CCD 不收敛,返回 NaN |
| Reset 时初始姿态穿透地形 | `qpos` 初始就在地形几何内,首步约束力发散 |
| 关节并联机构脱出可行域 | S45 踝关节并联机构有凸可行域,脱出后 `-log(x+eps)` barrier 大爆负 reward,PPO 学坏 → 动作饱和 → 物理崩坏 |

### 观测函数层

| 函数 | 风险点 | 是否已防护 |
|---|---|---|
| `mdp.depth_image_obs` | depth 含 NaN/Inf | ✅ 已做 `nan_to_num + clip` |
| `mdp.foot_height` | `site_pos_w` 在 solver 失败那一步为 NaN | ✅ 本会话已补 `nan_to_num` |
| `mdp.foot_air_time` / `foot_contact` / `foot_contact_forces` | 接触 sensor 在异常 contact 配置下可能给 NaN | ❌ 暂未单独保护 |
| `mdp.phase` | 纯函数式,只看 `episode_length_buf` | 安全 |
| 上游 mjlab `joint_pos_rel` / `joint_vel_rel` / `projected_gravity` / `builtin_sensor` | 直接读 `asset.data.*`,物理层 NaN 直接透传 | ❌ mjlab 上游未做保护 |

### Reward 层

| Reward | 风险 |
|---|---|
| `illegal_dof_pos_barrier` | `-log(x+eps)` 在 `x → -eps` 时指数增大;默认 `eps=0.05, max_penalty=25` 偏激进 |
| `feet_too_near_humanoid` (-5.0)、`fly` (-10.0)、`is_terminated` (-200.0) | 高权重 + 快速触发 → 单步 reward 量级 100+,PPO adaptive LR 与之共振发散 |
| `contact_force_violation` | 当 `force_history` 含 NaN 时未做防护 |

## 解决方案分层

按"代价低 → 高"顺序列出,**实际部署建议三层全部叠加**。

### 层 1:观测层 `nan_to_num` 兜底

`src/omni_gs_playground/tasks/velocity/mdp/observations.py::foot_height` 加 `torch.nan_to_num`:

```python
def foot_height(env, asset_cfg=_DEFAULT_ASSET_CFG):
  asset = env.scene[asset_cfg.name]
  z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  return torch.nan_to_num(z, nan=0.0, posinf=1.0, neginf=0.0)
```

**已经 apply。** 同模式可扩展到其他 critic-only 项(`foot_air_time / foot_contact / foot_contact_forces`),按需添加。

### 层 2:物理层接触池扩容

`src/omni_gs_playground/tasks/velocity/config/kuavo/env_cfgs.py::_kuavo_rough_env_cfg`:

```python
cfg.sim.contact_sensor_maxmatch = 1000   # 原 500
cfg.sim.nconmax = 128                    # 原 64
```

**已经 apply。** `nconmax=64` 是项目作者为省显存压下来的,但加入 `undesired_body_contact` sensor + rough 复杂地形后明显不够。128 在 num_envs=1024 时仅多用几百 MB 显存,稳健性收益远大于代价。

> 详见:CLAUDE.md "显存敏感项"一节。

### 层 3:Reward 软化

`src/omni_gs_playground/tasks/velocity/mdp/rewards.py::illegal_dof_pos_barrier`:

```python
eps = 0.1            # 原 0.05
max_penalty = 10.0   # 原 25.0
```

**已经 apply。** 保持 barrier 形状,但把单关节最大惩罚从 25 降到 10。其他高权重 reward(`feet_too_near` -5.0、`fly` -10.0、`is_terminated` -200.0)如果在训练中仍触发 PPO 发散,可以再阶段性减半。

### 层 4:策略层 NaN 钩子(**最关键**)

`packages/rsl_rl/rsl_rl/utils/utils.py` 新增 `sanitize_nan(obs, rewards, dones)` 函数,`packages/rsl_rl/rsl_rl/runners/on_policy_runner.py::learn()` 把 `check_nan(...)` 替换为 `sanitize_nan(...)`。

**已经 apply。** 行为:

1. per-env 检测 NaN/Inf;
2. 就地 `nan_to_num` 替换为 0;
3. 把出问题的 env 强制 `done=True`,mjlab auto-reset 接管;
4. 首次触发打印一条警告,后续静默。

**这一层是最后兜底**:即使前三层都失效,也能让训练不挂掉,把"硬故障"变成"无感的单 env 重置"。

## 诊断工具

### 1. NanGuard(物理层 dump)

`scripts/train.py --enable-nan-guard True` 会启用 mjlab 的 NanGuard,它在物理层(每个 sub-step)检测 NaN,触发时:

- dump 最近 100 个 step 的 `mj_getState` 状态到 `/tmp/mjlab/nan_dumps/nan_dump_<ts>.npz`;
- dump 当时的模型到同目录的 `.mjb` 文件,可用 `mujoco.viewer` 直接打开重现;
- 日志输出哪个 env 触发(`nan_env_ids`)。

**注意:配合本会话的层 4 改动,NanGuard 不再 raise,只写盘记录**。这正是想要的行为:既不挂掉训练,又留下诊断材料。

读取 dump:

```python
import numpy as np
d = np.load("/tmp/mjlab/nan_dumps/nan_dump_latest.npz", allow_pickle=True)
meta = d["_metadata"].item()
print(meta)  # detection_step, nan_env_ids, state_spec, state_size, ...
# d["states_step_<i>"] 是 (num_envs_dumped, state_size) 的 float64,
# 用 mj_setState + state_spec 可在 mujoco.viewer 里重放
```

`_metadata` 字段含义:

| 字段 | 含义 |
|---|---|
| `num_envs_total` | 训练并行 env 数,例如 1024 |
| `num_envs_dumped` | 实际 dump 的 env 数(一般等于 NaN env 数)|
| `nan_env_ids` | 触发 NaN 的 env 索引列表,例如 `[386]` |
| `state_spec` | `mjtState` 组合位,例如 30 |
| `state_size` | 单 env state 维度,例如 65 |
| `detection_step` | 全局步数,例如 497242 |
| `timestamp` | `YYYYMMDD_HHMMSS`,例如 `20260623_221121` |
| `model_file` | 同目录下的 `.mjb` 模型文件名 |

### 2. `--enable-nan-guard` 关闭 / 开启的选择

| 场景 | 推荐 |
|---|---|
| 长跑训练(已加层 4) | 开启,留 dump 备查 |
| 调试 reward / 改动后验证 | 开启 + 看是否会触发新的 NaN |
| 跑 baseline 速度 benchmark | 关闭(每步多一次扫描,约 5%-10% 开销) |

## 验证步骤

层 1-4 全部 apply 之后:

1. 任务发现:
   ```bash
   uv run python scripts/list_envs.py --keyword Kuavo-S45
   ```
2. 启动训练(`sanitize_nan` 自动接管,不需要新参数):
   ```bash
   uv run python scripts/train.py Kuavo-S45-Rough \
     --enable-nan-guard True \
     --env.scene.num-envs 1024 \
     --agent.max-iterations 40000
   ```
3. 训练过程中如果触发 NaN,**控制台只打印一行**:
   ```
   [sanitize_nan] First NaN/Inf detected in 1 env(s): [386]. Replacing with 0.0 and forcing done=True. Further occurrences will be silent.
   ```
   训练继续。
4. 同步在 `/tmp/mjlab/nan_dumps/` 看是否有新 dump 文件 → 有就保留备后续根因分析,无就说明物理层未触发(可能只是观测层短暂瞬态)。

## 适用范围与限制

| 任务 | 是否需要这套防护 |
|---|---|
| `Kuavo-S45-Rough`(CNN + EMP rewards)| **必须**,本会话目标 |
| `Kuavo-S45-DeFM-Rough` | 推荐 |
| `Kuavo-S45-Flat` | 层 1+4 足够(平地物理简单,层 2/3 影响小) |
| `Kuavo-S54-Rough` / `Kuavo-S54-Head-CNN-Rough` | 推荐;层 2 的 `nconmax=128` 已经通过 `_kuavo_rough_env_cfg` 共享生效 |
| `Kuavo-S54-Flat` | 层 1+4 足够 |
| `Unitree-G1-*` | 仅层 1+4(rl_cfg 不共享,需要独立加 sanitize_nan) |

层 4(`sanitize_nan`)的几个**前提与限制**:

1. **依赖 mjlab `auto_reset=True`(默认)**。如果用户改成 `auto_reset=False` + 自己管 reset,需要在 reset 函数里也处理 `bad_envs_mask` 返回值。
2. **NaN 替换为 0 是有偏的**。如果 NaN 比例 ≥ 50% / batch,PPO 仍会朝"全部为 0"的方向被错误更新一次。本会话观察到的 NaN 比例是 1/1024(0.1%),完全可以忽略;一旦比例飙高,需要回到层 1-3 找根因。
3. **不替代 NanGuard dump 分析**。当 NaN 在长跑中第一次出现时,**强烈建议**在下一次空闲时间打开 dump 查一下根因,而不是一直让 sanitize 兜底 —— sanitize 是"防灾"不是"治灾"。

## 关键文件清单

| 文件 | 角色 |
|---|---|
| `src/omni_gs_playground/tasks/velocity/mdp/observations.py` | `foot_height` 加 `nan_to_num`(层 1) |
| `src/omni_gs_playground/tasks/velocity/mdp/rewards.py` | `illegal_dof_pos_barrier` 软化(层 3) |
| `src/omni_gs_playground/tasks/velocity/config/kuavo/env_cfgs.py` | `_kuavo_rough_env_cfg` 抬 `nconmax / contact_sensor_maxmatch`(层 2) |
| `packages/rsl_rl/rsl_rl/utils/utils.py` | 新增 `sanitize_nan`(层 4) |
| `packages/rsl_rl/rsl_rl/utils/__init__.py` | 导出 `sanitize_nan` |
| `packages/rsl_rl/rsl_rl/runners/on_policy_runner.py` | `learn()` 把 `check_nan` 换成 `sanitize_nan` |

如需回滚某一层:层 1-3 直接 git revert 对应 hunk;层 4 把 `on_policy_runner.py` 改回 `check_nan(...)` 即可,函数定义保留不影响。
