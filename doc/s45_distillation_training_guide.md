# S45 Depth-to-Terrain 蒸馏训练指南

## 1. 训练目标

当前 `Kuavo-S45-Rough-Distill` 使用三个互补目标：

```text
L = L_action_bc + 0.10 * L_height + 0.05 * L_height_gradient
```

- `L_action_bc`：student 与 frozen EMP teacher 的加权 Huber action loss。腿部关节权重为 `1.5`，手臂关节为 `0.5`。
- `L_height`：student depth CNN 的池化前空间特征重建 teacher 前方 `7x9` 局部高度图。
- `L_height_gradient`：约束相邻网格的高度差，强化台阶边缘、坡度和可落足区域边界。

全局 `latent_loss_coef` 默认为 `0.0`。teacher height scan 与 perspective depth 不是同构观测，直接对齐两个 global-pooled latent 会丢失落足所需的空间位置，并向 student 注入不可观测信息。

## 2. 辅助监督范围

terrain target 来自 `teacher_height`，对应 base yaw frame 中前方 `x=0.0...0.8 m`、横向 `y=-0.3...0.3 m` 的 `7x9` crop，不包含身后区域。`teacher_height_valid` 会屏蔽 terrain ray miss。

该 mask 是 terrain target 的有效性标记，不是严格的相机 visibility mask。当前实现属于 SSR 思路的轻量版本：让 depth 学习与落足相关的前向局部几何，但不要求单帧 depth 猜测身后或历史地形。若以后扩大到脚下或身后区域，应增加时序 depth、相机位姿补偿和 BEV memory，再对累积地图监督。

辅助 decoder 只在 distill 训练时存在，不进入导出的 student policy，因此不会增加部署推理开销。训练时会增加一个小型 `Conv2d -> ELU -> Conv2d` head 的计算和参数。

## 3. 开始新蒸馏训练

先确认任务和参数入口：

```bash
uv run python scripts/list_envs.py --keyword Rough-Distill
uv run python scripts/train.py Kuavo-S45-Rough-Distill --help
```

启动训练：

```bash
uv run python scripts/train.py Kuavo-S45-Rough-Distill
```

DAgger-lite 的 teacher intervention probability 在前 `12000` 次 update 内从 `1.0` 线性降到 `0.05`。它控制谁执行环境动作，不是 BC loss 权重。distill 阶段环境 reward 仍只用于评估和日志，不参与反向传播。

新增辅助 decoder 后，建议从新 run 开始训练。旧 distill checkpoint 没有 `terrain_decoder_state_dict`，不适合按完整 optimizer 状态直接续训；它仍可正常用于 student play，或作为 PPO fine-tune 的 actor 初始化。

## 4. 监控指标

优先观察：

| 指标 | 含义 | 异常信号 |
|---|---|---|
| `Loss/behavior` | teacher action 拟合 | 长期不降或突然尖峰 |
| `Loss/terrain_reconstruction` | 局部高度重建 | 不降说明 depth 特征未学到几何 |
| `Loss/terrain_gradient` | 台阶/坡度边缘重建 | 高度 loss 降而该项不降，通常仍会模糊边缘 |
| `Loss/teacher_intervention_rate` | 实际 teacher 接管率 | 与 beta 长期偏离说明采样或日志异常 |
| `Episode/fell_over` | 闭环稳定性 | beta 降低时快速升高表示 student rollout 脆弱 |
| `Episode/edge_contact` | 落足边缘风险 | distill 只评估；PPO fine-tune 才真正优化 |

不要仅凭 distill 的 `mean_reward` 或 episode length 判断是否学会落足。teacher 接管会直接改善这些曲线，而且 distill reward 不产生梯度。固定命令、固定 terrain seed 的 play 对比更有判断力。

## 5. 两阶段 PPO + BC Fine-tune

Fine-tune 不再直接开启 terrain curriculum。第一阶段使用与 distill 一致的固定地形分布，让新 critic 收敛，同时保护 BC 初始化：

```bash
uv run python scripts/train.py Kuavo-S45-Rough-Distill-Finetune \
  --checkpoint-file logs/rsl_rl/kuavo_s45_distill_velocity/<run>/model_<iteration>.pt \
  --gpu-ids '[0]' \
  --enable-nan-guard True \
  --env.scene.num-envs 1024 \
  --agent.max-iterations 5001
```

第一阶段确认稳定后，从 PPO checkpoint 进入 curriculum 阶段：

```bash
uv run python scripts/train.py Kuavo-S45-Rough-Distill-Finetune-Curriculum \
  --checkpoint-file logs/rsl_rl/kuavo_s45_distill_finetune_velocity/<run>/model_5000.pt \
  --gpu-ids '[0]' \
  --enable-nan-guard True \
  --env.scene.num-envs 1024 \
  --agent.max-iterations 10001
```

`--agent.max-iterations` 表示本次额外执行的 update 数。第二阶段会恢复 actor、critic、optimizer、iteration 和 BC 退火进度，不会从 `0.3` 重新开始。

Fine-tune 的目标是 PPO reward 加 teacher action regularizer：

```text
L = L_PPO + lambda_bc * weighted_huber(student_mean, teacher_action)
```

- 前 `3000` update 固定 `lambda_bc=0.3`，随后用 `12000` update 退火到 `0.05`。
- 12 个腿部动作权重为 `1.5`，14 个手臂动作权重为 `0.5`，与 distill 保持一致。
- 从 distill checkpoint 加载时保留 PPO 配置的 `std=0.15`，不导入 BC 中未训练的 `std=0.5`。
- 学习率为 `1e-4`，entropy coefficient 为 `5e-4`，降低新 critic 尚未稳定时的策略漂移。
- `edge_contact` 使用覆盖大部分脚掌的 `9x5` downward ray grid，并具有静态接触成本；手臂 L1 偏离成本为 `-0.3`。

足底 ray 数从每脚 `9` 增至 `45`。`1024` 个环境总计增加约 `7.4` 万条 ray query；如果显存或采集 FPS 明显恶化，先将环境数降到 `512`，不要缩回只覆盖脚掌中心的网格。

查看 student：

```bash
uv run python scripts/play.py Kuavo-S45-Rough-Distill \
  --checkpoint-file logs/rsl_rl/kuavo_s45_distill_velocity/<run>/model_<iteration>.pt \
  --num-envs 1 --viewer viser
```

查看第一阶段 policy：

```bash
uv run python scripts/play.py Kuavo-S45-Rough-Distill-Finetune \
  --checkpoint-file logs/rsl_rl/kuavo_s45_distill_finetune_velocity/<run>/model_<iteration>.pt \
  --num-envs 1 --viewer viser
```

查看 curriculum policy 时，将任务名换成 `Kuavo-S45-Rough-Distill-Finetune-Curriculum`，checkpoint 指向 `kuavo_s45_distill_finetune_curriculum_velocity` 对应 run。

## 6. Fine-tune 验收条件

不要再以总 reward 单独选 checkpoint。每 `500` update 至少检查：

| 指标 | 预期 | 停止条件 |
|---|---|---|
| `Policy/mean_std` | 首轮接近 `0.15`，之后缓慢变化 | 首轮仍接近 `0.5` |
| `Loss/behavior_coef` | 前 3000 update 保持 `0.3` | 提前退火 |
| `Loss/behavior` | 不持续偏离 BC 基线 | 连续多个 checkpoint 高于约 `0.08` |
| `Episode_Reward/joint_deviation_arms` | 不持续恶化 | play 出现明显高抬臂 |
| `Episode_Reward/edge_contact` | 与固定 seed play 的踩边次数同步下降 | 总 reward 上升但踩边增加 |
| `Episode_Termination/fell_over` | 不高于 BC 固定评估 | curriculum 开启后阶跃上升 |

第一阶段只有在固定 seed 的台阶上行、下行和 tilted-grid play 均不劣于 BC 时，才进入 curriculum。建议保存 BC、第一阶段 `model_3000/5000` 和 curriculum `model_5000/10000` 的相同命令、相同 terrain seed 视频做并排比较。

## 7. 调参顺序

1. 保持 `latent_loss_coef=0.0`，先确认三个 loss 都有限且下降。
2. 若 behavior 收敛明显变慢，先将 `terrain_reconstruction_loss_coef` 从 `0.10` 降至 `0.05`，不要先提高 BC 学习率。
3. 若高度重建平滑但台阶边缘仍差，将 `terrain_gradient_loss_coef` 从 `0.05` 提到 `0.10`。
4. 若 beta 降低后跌倒率陡升，延长 `teacher_intervention_decay_updates`，不要让 teacher 最终接管率高于 `0.05` 来掩盖 student 问题。
5. 辅助 loss 收敛但 play 落足仍弱时，进入 PPO fine-tune；继续增加重建权重不能替代 reward 驱动的闭环优化。
6. Fine-tune 的 `Loss/behavior` 先恶化时，延长 hold 或提高 BC 终值；只有 BC 稳定而踩边仍多时，才调 `edge_contact`。
7. 手臂仍抬高时，先确认 `joint_deviation_arms` 非零且在恶化，再将其从 `-0.3` 调到 `-0.5`；不要靠提高全部 action-rate cost 间接压手臂。
