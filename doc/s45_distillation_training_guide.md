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

## 5. PPO + BC Fine-tune

选择新的 distill checkpoint 后启动：

```bash
uv run python scripts/train.py Kuavo-S45-Rough-Distill-Finetune \
  --checkpoint-file logs/rsl_rl/kuavo_s45_distill_velocity/<run>/model_<iteration>.pt
```

fine-tune 阶段的目标是 PPO reward 加退火 teacher action regularizer。BC 系数从 `0.2` 降到 `0.02`，而 reward 经 PPO advantage 真正参与策略更新。`edge_contact`、跌倒、速度跟踪和姿态等 reward 因而能够修正纯 BC 的闭环误差。

查看 student：

```bash
uv run python scripts/play.py Kuavo-S45-Rough-Distill \
  --checkpoint-file logs/rsl_rl/kuavo_s45_distill_velocity/<run>/model_<iteration>.pt \
  --num-envs 1 --viewer viser
```

查看 fine-tune policy 时，把任务名换成 `Kuavo-S45-Rough-Distill-Finetune`，checkpoint 指向 `kuavo_s45_distill_finetune_velocity` 对应 run。

## 6. 调参顺序

1. 保持 `latent_loss_coef=0.0`，先确认三个 loss 都有限且下降。
2. 若 behavior 收敛明显变慢，先将 `terrain_reconstruction_loss_coef` 从 `0.10` 降至 `0.05`，不要先提高 BC 学习率。
3. 若高度重建平滑但台阶边缘仍差，将 `terrain_gradient_loss_coef` 从 `0.05` 提到 `0.10`。
4. 若 beta 降低后跌倒率陡升，延长 `teacher_intervention_decay_updates`，不要让 teacher 最终接管率高于 `0.05` 来掩盖 student 问题。
5. 辅助 loss 收敛但 play 落足仍弱时，进入 PPO fine-tune；继续增加重建权重不能替代 reward 驱动的闭环优化。
