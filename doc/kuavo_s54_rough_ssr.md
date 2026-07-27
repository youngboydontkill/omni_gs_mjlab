# Kuavo-S54-Rough-SSR 改动总结

本文档总结 `Kuavo-S54-Rough-SSR` 任务从论文
《SSR: Scaling Surefooted and Symmetric Humanoid Traversal to the Open World》
迁移到本仓库后的主要实现、与论文的一致性边界，以及已经做过的验证。

当前实现结论：

- 已迁移 SSR 的 rough-terrain locomotion 奖励设计。
- 不包含动作风格奖励。
- Actor 已按论文思路改成“视觉特征 + 本体特征先融合，再进入 GRU”。
- depth 输入分辨率改为 `42 x 42`。
- 已将原先的“当前摆动脚竖直投影支撑奖励”替换为论文思路对应的
  imagined foothold guidance。
- 仍保留单个 asymmetric critic，没有实现论文里的三 critic 拆分。

## 1. 任务与配置入口

任务名：

- `Kuavo-S54-Rough-SSR`

主要配置入口：

- `src/omni_gs_playground/tasks/velocity/config/kuavo/env_cfgs.py`
- `src/omni_gs_playground/tasks/velocity/config/kuavo/rl_cfg.py`
- `packages/rsl_rl/rsl_rl/models/ssr_model.py`
- `packages/rsl_rl/rsl_rl/algorithms/ppo.py`
- `packages/rsl_rl/rsl_rl/extensions/foothold.py`

## 2. 机器人参数与足底几何对齐

这次迁移里，S54 相关的物理参数、几何尺寸和扫描分辨率统一取自：

- `src/omni_gs_playground/assets/robots/kuavo/biped_s54/kuavo_s54_constants.py`

足底扫描对齐做了两件事：

1. 使用 S54 的实际 sole scan site，而不是沿用 S45 的位置假设。
2. 使用 S54 常量中定义的足底尺寸与采样分辨率。

当前足底相关参数：

- sole size: `0.25 x 0.10 m`
- sole sampling resolution: `0.025 m`

这里要区分两层分辨率：

- 足底支撑率评估仍按 `2.5 cm` 的足底采样点计算。
- imagined foothold 的 planning map 为了控制 1024 并行环境下的显存，占用的是
  `1.0 x 0.6 m`、`0.05 m` 分辨率的地形栅格，再对足底采样点做双线性插值。

也就是说，`5 cm` 是底层地形图分辨率，不是足底接触采样分辨率。

## 3. SSR Actor 网络迁移

实现文件：

- `packages/rsl_rl/rsl_rl/models/ssr_model.py`

当前 actor 结构：

1. 5 帧本体历史输入。
2. 当前时刻 `42 x 42` depth 输入。
3. 单帧本体先经过 MLP 编码到 `128` 维。
4. depth 经过三层 CNN 编码到 `128` 维。
5. 将 depth latent 广播到每个历史步，与每一帧本体 latent 拼接成 `256` 维序列。
6. 融合后的时序特征输入 `GRU(256 -> 256)`。
7. 取最后时刻 hidden state，经 fusion MLP 得到共享表示。
8. 从共享表示分出：
   - `z_foot` 16 维
   - `z_body` 16 维
   - `z_motion` 16 维
   - velocity estimator 输出 3 维基座速度
9. 将“当前帧本体 + 估计速度 + 48 维 latent”送入 5 专家 MoE actor 输出动作。

这一步修正了先前的偏差：现在是“视觉和本体先融合，再进 GRU”，而不是
“只让本体进 GRU，再与视觉特征在后面拼接”。

### 3.1 latent heads 与 velocity estimator 的位置

当前实现中：

- `latent heads` 位于 GRU 之后、fusion MLP 之后。
- `velocity estimator` 直接由整段本体历史输入的 MLP 输出。

这与论文的核心关系保持一致：

- 时序融合结果负责产生高层 latent。
- velocity estimator 是辅助预测头，不是 actor 的单独时序分支。

### 3.2 保留的辅助重建头

当前仍保留：

- `z_foot -> foot_height_decoder`
- `z_body -> body_height_decoder`
- `z_motion -> next_proprio_decoder`
- `velocity_estimator -> base_velocity`

对应 hybrid prediction losses：

- body height map: `2.0`
- foot height map: `1.0`
- next proprioception: `5.0`
- KL: `1.0`
- base velocity: `2.0`

## 4. 五专家 MoE 的具体含义

实现仍是论文中的 mixture-of-experts actor 思路：

- 一个 gate 网络输出 5 个权重。
- 5 个 expert MLP 分别输出动作均值特征。
- 最终输出为 5 个 expert 输出的加权和。

对应代码位置：

- gate: `_SSRMoEActor.gate`
- experts: `_SSRMoEActor.experts`

每个 expert 的宽度保持为：

- `[1024, 512, 128]`

## 5. 奖励设计迁移

本次迁移保留了 SSR locomotion 奖励设计，不包含动作风格奖励。

另外，原先环境里的旧版落足奖励：

- `mdp.ssr_foothold_support`

已经从环境奖励项中删除，不再作为最终实现使用。

## 6. imagined foothold guidance 实现

这是这次改动的核心变化。实现文件：

- `packages/rsl_rl/rsl_rl/extensions/foothold.py`

论文中的思路是：训练一个独立的、特权信息驱动的未来落足点想象模型，用它对策略的落脚位置提供训练期引导。当前代码已经按这个思路落地成一个训练期扩展模块 `ImaginedFoothold`。

### 6.1 现在的实现包含什么

已经实现的 5 个关键点：

1. 独立的特权落足点预测器。
2. 使用 critic / privileged state 作为输入，而不是 actor 可部署观测。
3. 使用“未来首次接触”作为延迟监督信号。
4. 对任意候选落点从 planning height map 查询支撑率。
5. 用 imagined foothold guidance 替换旧的摆动脚竖直投影奖励，并加入课程门控。

### 6.2 预测器输入输出

输入：

- 特权状态 `state_group="critic"`
- 当前动作

输出：

- 左右脚各自的二维高斯分布参数
- 即每只脚的 `mu_x, mu_y` 和共享轴向不确定度 `std_x, std_y`

代码里组织为：

- `mu`: `[B, 2, 2]`
- `std`: `[B, 2, 2]` 的等价广播语义，内部由 4 个均值和 2 个标准差构成

### 6.3 延迟监督如何落地

策略在时刻 `t` 看到的是“将来会落在哪里”，真实标签要等到将来的 first contact 才知道。因此实现里：

1. 在每个源时刻缓存 `(s_t, a_t)`。
2. 同时缓存该时刻 root 的世界坐标与 yaw。
3. 当某只脚出现 first contact 时，读取该脚 sole center 的真实世界坐标。
4. 将这个真实 touchdown 点变换回源时刻的 root-yaw 局部坐标系。
5. 把它作为监督标签写入 replay buffer。

这正对应论文里的“未来落足点想象”监督，而不是拿当前脚位置做伪标签。

### 6.4 奖励如何计算

训练时 imagined foothold predictor 会对左右脚给出未来候选落点分布。当前实现采用确定性 sigma points 近似期望支撑率：

- 中心点 1 个
- 沿 `+x/-x/+y/-y` 方向各 1 个

总共 5 个 sigma points。

对每个候选点：

1. 在 root-yaw 对齐的 planning map 上查询局部地形高度。
2. 用 S54 的足底采样点铺开接触面。
3. 判断该足底区域是否被地形充分支撑。
4. 计算 support deficiency。

最后将 imagined foothold reward 注入 PPO 的环境步奖励中。

当前默认系数：

- reward weight `w_f = 0.25`
- reward variance `sigma_f = 0.0625`
- support height threshold `0.03 m`

### 6.5 课程门控

这个奖励不会从训练第一步就打开，而是满足条件后才生效。默认门控条件：

- replay 样本数至少 `4096`
- predictor 更新次数至少 `100`
- terrain level 至少 `5`

这样可以避免 predictor 还没学会时过早污染 PPO reward。

## 7. PPO 接入方式

改动文件：

- `packages/rsl_rl/rsl_rl/algorithms/ppo.py`

接入点如下：

1. `act()` 后调用 imagined foothold predictor，基于 `(obs, action)` 计算训练期奖励。
2. `process_env_step()` 时把 imagined foothold reward 加到环境奖励上。
3. 同一阶段解析 first-contact 监督，写入 predictor replay buffer。
4. 每次 PPO update 后，再单独更新 predictor。
5. checkpoint 中同时保存和恢复 predictor 及其 optimizer。

为了降低 rollout 显存，planning map 和 foothold geometry 不写入 rollout storage，只在当前 step 使用。

## 8. 新增观测与环境传感器

改动文件：

- `src/omni_gs_playground/tasks/velocity/mdp/observations.py`
- `src/omni_gs_playground/tasks/velocity/config/kuavo/env_cfgs.py`

新增了两组专门给 imagined foothold 用的观测：

### 8.1 `ssr_foothold_terrain`

包含：

- root-yaw 对齐的 planning map 高度
- 对应 ray hit validity mask

这样落足点评估时可以把“超出地图范围 / ray 未命中”显式视为无支撑，而不是伪造高度。

### 8.2 `ssr_foothold_geometry`

固定 13 维，布局为：

1. root world XY: 2
2. root yaw cos/sin: 2
3. 左右 sole center world XY: 4
4. 当前接触标记: 2
5. first contact 标记: 2
6. terrain level: 1

这组几何量既用于落足点监督对齐，也用于奖励门控。

### 8.3 planning ray scan

新增了 root-yaw 对齐的 planning raycast：

- map size: `1.0 x 0.6 m`
- map resolution: `0.05 m`

这里是显存和分辨率之间的折中。更大的窗口或更细的栅格在 `1024 env` 下会明显增加 Warp CUDA graph 开销，之前更大的配置会触发显存不足。

## 9. 与论文仍存在的差异

当前实现已经把 SSR 的关键训练思路迁到了 S54，但仍有几处没有逐项复现：

1. 没有实现论文中的 3 个 critic，当前仍是单个 asymmetric critic。
2. 没有实现动作风格奖励。
3. imagined foothold predictor 作为训练期扩展存在，不参与部署期 actor 输入。
4. planning map 的空间范围和分辨率为了 1024 并行环境做了显存约束下的缩减。

这些差异里，第 1、2 条是功能边界，第 3 条是符合论文训练期特权模块定位的，第 4 条是工程折中。

## 10. 验证记录

已经做过的检查包括：

```bash
uv run python scripts/list_envs.py --keyword Kuavo-S54-Rough-SSR
uv run python scripts/train.py Kuavo-S54-Rough-SSR --help
uv run python scripts/play.py Kuavo-S54-Rough-SSR --help
uv run python -m compileall -q packages/rsl_rl/rsl_rl packages/rsl_rl/tests src/omni_gs_playground
uv lock --check
git diff --check
git -C packages/rsl_rl diff --check
```

由于环境里没有安装 `pytest`，相关测试通过直接调用验证：

```bash
uv run python -c "import runpy; m=runpy.run_path('packages/rsl_rl/tests/extensions/test_foothold.py'); m['test_delayed_touchdown_supervision_and_guidance_reward']()"
```

还做过一次 `1024 env` 的训练 smoke test：

```bash
WANDB_MODE=disabled uv run python scripts/train.py Kuavo-S54-Rough-SSR \
  --enable-nan-guard True \
  --env.scene.num-envs 1024 \
  --agent.max-iterations 1
```

该次 smoke test 的关键信息：

- 24,576 steps
- 约 5,502 steps/s
- 约 7,586 个落足监督样本
- predictor NLL 约 `-3.504`
- 平均落点误差约 `0.110 m`

同时，由于默认门控要求 predictor 至少更新 100 次，而这次 smoke test 只更新了 4 次，所以 imagined foothold reward 在这次短跑中保持关闭。这符合设计预期。

## 11. 代码改动清单

本次与 `Kuavo-S54-Rough-SSR` 直接相关的主要文件：

- `packages/rsl_rl/rsl_rl/extensions/foothold.py`
- `packages/rsl_rl/rsl_rl/extensions/__init__.py`
- `packages/rsl_rl/rsl_rl/algorithms/ppo.py`
- `packages/rsl_rl/tests/extensions/test_foothold.py`
- `packages/rsl_rl/rsl_rl/models/ssr_model.py`
- `src/omni_gs_playground/tasks/velocity/config/kuavo/env_cfgs.py`
- `src/omni_gs_playground/tasks/velocity/config/kuavo/rl_cfg.py`
- `src/omni_gs_playground/tasks/velocity/mdp/observations.py`
- `src/omni_gs_playground/tasks/velocity/mdp/rewards.py`

注意：

- `env_cfgs.py` 中若存在与 arm reward 等相关的其他差异，不属于这次 SSR 迁移文档总结范围。

## 12. 使用方式

```bash
uv run python scripts/train.py Kuavo-S54-Rough-SSR --help
uv run python scripts/train.py Kuavo-S54-Rough-SSR
uv run python scripts/play.py Kuavo-S54-Rough-SSR \
  --checkpoint-file logs/rsl_rl/kuavo_s54_rough_ssr/<run>/model_<iteration>.pt
```
