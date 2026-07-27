# S45 Distill 训练日志问题分析

对比对象：

- Distill: `logs/rsl_rl/kuavo_s45_distill_velocity/2026-07-23_19-13-10`（约 16.5k iter）
- Rough PPO teacher 基线: `logs/rsl_rl/kuavo_s45_velocity/2026-07-22_16-42-01`（约 40k iter）

用户观察到的 3 个现象：

1. Distill 的 `episode_length`、`mean_reward` 上升很快，但抖动大。
2. 分项 reward 数值也常“比直接训更好看”，但同样剧烈抖动。
3. Play 时 student 没有学会 teacher 的落足点控制。

结论先说：

- 现象 1、2 主要是**指标语义和训练目标不同**，不是 student 已经学得比 Rough PPO 更稳。
- 现象 3 是核心真实问题：当前算法是**纯 on-policy action BC**，环境 reward 只记日志、不参与优化，所以 student 可以拟合局部动作均值，却学不会 teacher 在 rough 上的落足与恢复策略。
- **BC 与现有 student reward 目前没有优化层面的冲突**，因为 reward 不进 loss；冲突发生在“指标解读”和“后续若把 reward 直接塞进 BC loss”时。

---

## 1. 两边训练目标本质不同

| 项目 | Distill | Rough PPO |
|---|---|---|
| 算法 | `Distillation` | `PPO` |
| 优化目标 | student 动作 MSE 对齐 frozen teacher | task reward + value/advantage |
| Rollout 动作 | student 确定性 mean（`student_rollout_stochastic=False`） | 高斯采样探索 |
| Reward 是否反传 | **否**，只写 TensorBoard | **是**，直接驱动策略 |
| 输入 | student: depth + proprio；teacher: cmd + 5 帧 proprio + height scan | actor/critic: depth + proprio/critic 特权 |
| 命令分布 | `vx∈[0,1]`, `vy=0`, `wz∈[-0.8,0.8]`, 无 standing | `vx∈[0,1]`, `vy∈[-1,1]`, `wz∈[-1,1]`, 5% standing |

关键代码事实：

```python
# packages/rsl_rl/rsl_rl/algorithms/distillation.py
behavior_loss = self.loss_fn(actions, batch.privileged_actions)
# rewards 只在 process_env_step 里被记录，不进入 update()
```

因此：

- Distill 曲线“reward 高”≠ “已经按 reward 学好了”。
- Rough 曲线“reward 慢慢涨”才是真正被优化出来的。
- 两边的 `Train/mean_reward` 只是同构环境返回值的日志，不是同一优化问题下的可比 score。

---

## 2. 日志证据：快速上升 + 高抖动是什么

### 2.1 关键标量对比

窗口均值（按 iteration）：

| 指标 | Distill 0-200 | Distill 1k-5k | Distill 10k-16k | Rough 0-200 | Rough 5k-10k | Rough 20k-40k |
|---|---:|---:|---:|---:|---:|---:|
| `Train/mean_reward` | 5.3 | 59.2 | 58.4 | -2.4 | 10.4 | 117.6 |
| reward 相邻差分均值 | 9.7 | 11.8 | 9.6 | 0.74 | 1.79 | 2.66 |
| `Train/mean_episode_length` | 485 | 891 | 882 | 79 | 262 | 921 |
| `Episode_Reward/track_linear_velocity` | 1.80 | 4.27 | 4.29 | 0.10 | 0.48 | 5.05 |
| `Curriculum/terrain_levels` | 0.64 | 3.66 | 3.72 | 0.13 | ~0 | 4.70 |
| Distill `Loss/behavior` | 1.22 | 0.057 | 0.040 | - | - | - |
| Distill 跌倒率 `fell_over` | 很高后降到 ~0.27 | ~0.28 | ~0.27 | 12.9 | 4.06 | 0.19 |

读法：

1. **Distill 在 200~1000 iter 就把 length / track reward 拉到高位**，因为 frozen teacher 本身已经会走；student 只要部分模仿 teacher 的低频动作，闭环就能活很久。
2. **Rough PPO 早期必须从摔倒中探索**，所以同样 reward 项要 10k+ iter 才爬起来。
3. Distill 的 mean_reward 后期卡在 ~58，而 Rough 后期到 ~118。说明 Distill 并没有真正超过 teacher/PPO 基线，只是“提前到了中等平台”。
4. Distill 的 reward 抖动（相邻差分 ~10）比 Rough（~2~3）大一个数量级。这不是“学得更好”，而是**策略对局部状态扰动更脆**。

### 2.2 为什么 reward/length 抖得很厉害

当前 Distill 有几层放大抖动的机制：

1. **纯 BC、无 value baseline / advantage 平滑**  
   PPO 的 critic 会吸收一部分回报方差；Distill 只有动作 MSE，环境回报完全是被动结果。

2. **确定性 student rollout**  
   `student_rollout_stochastic=False` 让状态轨迹集中，但一旦偏离 teacher 分布，错误会沿轨迹累积，episode 间差异更大。这是正确的 BC 设置，却会表现为“某些 seed/terrain 上突然塌”。

3. **课程与终止事件是离散冲击**  
   Distill 在 1k iter 后 terrain level 已到 ~3.7，而跌倒率长期约 25%~30%。  
   一个 episode 若 timeout，reward 高；若中途 fell_over，`is_terminated=-200` 会把累计 reward 拉崩。  
   Logger 用最近 100 个完成 episode 的均值，所以曲线天然锯齿。

4. **命令分布更“好赚”**  
   Distill 去掉了侧向速度和 standing。Rough 的 `vy` 与 standing 会压低早期 track reward。  
   所以 Distill 分项 reward 更漂亮，部分来自**任务更简单**，不是 student 更强。

5. **behavior loss 已经很小，但闭环质量没有同步变好**  
   后期 `Loss/behavior≈0.038` 仍缓慢下降，而 mean_reward / fell_over / terrain 几乎不动。  
   说明 student 主要在拟合“平均动作残差”，而不是继续学到 rough 上的关键落足决策。

---

## 3. 为什么 play 里看不到 teacher 的落足点控制

这是最关键的问题，且与日志完全一致。

### 3.1 Teacher 的落足能力从哪里来

Frozen EMP teacher 使用：

- 5 帧 proprio history
- privileged height scan（7×9 crop）
- 在 Isaac Lab EMP 上训练出的高度图 CNN

也就是说，teacher 的落足点策略高度依赖 **height-map 特权观测**。

Student 则只有：

- 当前 proprio
- depth camera + 浅层 CNN

两边信息结构不对称。只做最终 26 维 action MSE，并不保证 depth CNN 学到“哪里能踩/不能踩”。

### 3.2 当前算法只会模仿动作，不会优化落足结果

Distill update 只有：

```text
L = MSE(student_action, teacher_action)
```

没有：

- foothold / edge / toe 相关 reward 反传
- teacher height-map latent 蒸馏
- 多步一致性或恢复动作加权
- DAgger 式 teacher 接管与历史数据聚合

因此 student 更容易学到：

- 平均步态与速度跟踪
- 躯干/手臂姿态

而更难学到：

- 楼梯边缘避让
- 落足修正
- 触边后的快速恢复

这与 play 现象一致：能走一段，但没有 teacher 的落足控制。

### 3.3 模态缺口会放大“动作接近但落足不像”

即使 action MSE 到 0.04，也只说明在 student 自己访问到的状态上，平均动作接近 teacher。  
一旦 student 偏离 teacher 状态分布：

1. teacher label 变成 OOD 纠正信号；
2. student depth 对局部几何估计差；
3. 下一步状态更偏；
4. 落足错误累积成跌倒或“乱踩”。

日志里的长期 ~27% fell_over，正是这种闭环脆弱性的统计表现。

### 3.4 环境侧也没有把“落足安全”做成足够强的可学目标

`_apply_s45_emp_rewards` 里：

- `toe_touch` 权重 `-5.0`，存在；
- `edge_contact` 目前被注释掉，没有启用。

即便启用，**在当前 Distill 中 reward 也不进 loss**，所以对 student 参数更新仍无直接影响。  
它们只改变你在 TensorBoard 上看到的“好不好看”，不改变 BC 学什么。

---

## 4. BC 与现有 student reward 会不会冲突？

### 4.1 现状：优化上不冲突，语义上容易误导

现状是：

```text
优化目标 = BC(action)
日志目标 = EMP reward 集合
```

所以：

- **没有梯度冲突**，因为 reward 不参与反向传播。
- **有解读冲突**：人眼看着 reward 变好，会误以为 BC 已把 task 学好。
- **有评估冲突**：play 关心落足与恢复，BC loss 关心逐步动作 MSE。

一句话：现在不是“BC 和 reward 打架”，而是“**BC 在学 A，你在用 B 评价它**”。

### 4.2 如果以后把 reward 直接加进 BC，会不会冲突？

会，而且冲突点很具体。

1. **量纲冲突**  
   action MSE 约 `1e-2` 量级；单步/累计 reward 可达几十到上百。直接 `L = MSE - α * reward` 很容易被 reward 主导或数值炸掉。

2. **目标冲突**  
   teacher action 来自 Isaac Lab EMP 策略与 height-map 先验；student 环境是 MJLab + depth + 当前 EMP reward 移植版。  
   teacher 未必是当前 reward 的最优策略。强行同时拟合 teacher action 和最大化当前 reward，会出现：
   - BC 拉向 teacher 落足风格；
   - reward 拉向 MJLab 当前权重下的局部最优（例如更保守步态、不同抬脚高度）。

3. **安全项与模仿项冲突**  
   高权重惩罚如 `is_terminated=-200`、`toe_touch=-5`、`fly=-10`、`feet_too_near=-1` 等，若未经 advantage 归一化直接加到 BC，会让 student 过度规避，而不是学 teacher 的有效穿越。

4. **不冲突、反而互补的部分**  
   这些项通常与 teacher 意图一致，适合作为后续 PPO fine-tune 的 task reward，而不是硬塞进纯 BC：
   - track linear / angular velocity
   - feet air time
   - pose / arm prior
   - 适度 action smoothness

### 4.3 推荐的组合方式

不要：

```text
L = MSE(action) + raw_env_reward
```

而应：

1. **阶段 1：纯 BC / DAgger-lite**  
   只优化 action（或 action + teacher latent），reward 仅监控。

2. **阶段 2：BC warm-start 后 PPO fine-tune**  
   ```text
   L = L_PPO(task_reward) + λ_bc(t) * L_action(+ L_latent)
   ```
   其中 `λ_bc` 从 1 降到 0.05~0.1。

3. **监控冲突，而不是假设没有冲突**  
   同时看：
   - `Loss/behavior`
   - fell_over / timeout
   - track velocity error
   - toe_touch / feet_slide / undesired_contacts
   - play 时楼梯边沿落足可视化

若 BC loss 下降但 toe/edge/fall 变差，说明 imitation target 与当前环境动力学/感知已错位。

---

## 5. 对 3 个现象的直接回答

### 现象 1：episode_length / mean_reward 上升快但抖

原因：

- teacher 先验让 student 很快获得“能走”的开环近似；
- 但 student 没有 value 平滑，也没有从 reward 学恢复；
- 命令更简单、课程爬升更快，进一步抬高早期曲线；
- 一旦偏航/踩边，episode 结果两极分化，日志均值抖动大。

### 现象 2：分项 reward 看起来更好却更抖

原因：

- 多数正 reward（track、air time、pose）是“走起来就会自动变高”的；
- Distill 没在优化这些项，只是被动表现出来；
- Rough PPO 后期其实更高更稳（reward ~118 vs Distill ~58）；
- 所以“看起来更好”只在训练前中期、且在更窄命令分布下成立。

### 现象 3：play 不会 teacher 落足点控制

原因：

- 只蒸馏最终动作，不蒸馏 height-map 表征；
- student 感知是 depth，不是 teacher 的 privileged scan；
- 无 DAgger 接管，student 自推状态后很难回到 teacher 落足分布；
- reward/edge 信号不进训练，无法补足 action MSE 对稀疏落足错误不敏感的问题。

---

## 6. 潜在问题清单（按优先级）

### P0. 训练范式不足以学落足

- 现状：on-policy action BC only
- 风险：behavior loss 很好看，play 仍然不会踩点
- 建议：先做 DAgger-lite（teacher 接管衰减 + replay），再 BC→PPO

### P0. reward 被误当成优化目标

- 现状：reward 仅日志
- 风险：误判“蒸馏已经成功”
- 建议：Distill 主看 `Loss/behavior`、action error、fall rate、play 落足；reward 只作辅助

### P1. 感知与特权信息缺口

- teacher: height scan CNN
- student: depth CNN
- 建议：蒸馏 teacher height-map latent / actor feature，而不是只拟合 26 维 action

### P1. 状态分布漂移

- student 自己 rollout，早期误差把轨迹带出 teacher 支撑集
- 建议：`beta` teacher 接管、失败状态重采样、replay 聚合

### P2. 评估设定不一致

- Distill 命令更窄（无 vy、无 standing）
- Rough 更难
- 建议：对比时固定同一 command / terrain curriculum，否则 reward 对比无意义

### P2. 落足相关 reward 未进入学习，且 edge term 未启用

- `edge_contact` 仍注释
- 即使打开，当前 Distill 也不会用它更新策略
- 建议：放到 PPO fine-tune 阶段，而不是指望纯 BC 吃到

### P3. 可能存在 teacher 迁移残差

已做的对齐：

- 关节顺序 MJCF ↔ Lab
- teacher height crop 7×9
- action scale 0.25
- 命令范围收缩到 teacher 训练分布

仍可能有：

- MJLab 与 Isaac Lab 动力学/接触差异
- depth 与 height-map 的几何不对齐
- teacher normalizer 在 MJLab 分布上的轻微偏移

这些会让“动作可模仿，落足不可复现”更严重。

---

## 7. 建议的下一步（可执行）

1. **先修正评估口径**  
   用同一 command range 和同一 terrain 子集，比较：
   - action MSE
   - fall rate
   - 速度误差
   - 楼梯/边缘场景 play

2. **给 Distill 增加 DAgger-lite**  
   - 执行动作：`a_env = teacher if u<β else student`
   - 标签始终是 teacher action
   - β 从 1.0 降到 0.05

3. **BC 收敛后切 PPO fine-tune**  
   - 保留小权重 BC 正则
   - 用现有 EMP reward 学恢复与落足
   - 这时 reward 才真正进入优化

4. **加 teacher latent 蒸馏**  
   - 目标：让 depth CNN 预测 teacher height feature
   - 这比只学 action 更接近“学会落足点”

5. **不要把 raw reward 直接加进 MSE**  
   - 那会制造真冲突
   - 正确路径是 PPO/advantage 体系里用 reward，BC 只作先验

---

## 8. 一句话总结

当前 Distill 的高 reward / 高 length 更像“**借助 teacher 先验快速获得可行走闭环**”，不是“**已经学会 teacher 的 rough 落足控制**”。  
BC 与现有 reward 在现状下**不发生梯度冲突**，但**严重存在目标错位**：训练只认 action MSE，而你关心的是 reward 与落足结果。  
要解决 play 不会踩点，必须从“纯动作 BC”升级到“分布匹配（DAgger）+ 表征蒸馏 + reward fine-tune”，而不是继续从 reward 曲线判断蒸馏成功。

---

## 附：关键文件

- 算法：[`packages/rsl_rl/rsl_rl/algorithms/distillation.py`](packages/rsl_rl/rsl_rl/algorithms/distillation.py)
- Runner：[`packages/rsl_rl/rsl_rl/runners/distillation_runner.py`](packages/rsl_rl/rsl_rl/runners/distillation_runner.py)
- Distill 配置：[`src/omni_gs_playground/tasks/velocity/config/kuavo/rl_cfg.py`](src/omni_gs_playground/tasks/velocity/config/kuavo/rl_cfg.py)
- Distill 环境：[`src/omni_gs_playground/tasks/velocity/config/kuavo/env_cfgs.py`](src/omni_gs_playground/tasks/velocity/config/kuavo/env_cfgs.py)
- Teacher wrapper：[`packages/rsl_rl/rsl_rl/models/emp_teacher_model.py`](packages/rsl_rl/rsl_rl/models/emp_teacher_model.py)
- 既有研究笔记：[`doc/distillation_research.md`](doc/distillation_research.md)
