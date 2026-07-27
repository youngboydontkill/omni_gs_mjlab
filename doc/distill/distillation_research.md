# S45 视觉学生蒸馏方案调研

## 结论

`Kuavo-S45-Rough-Distill` 当前不是离线数据集上的最简单 BC：每轮 rollout 都由 student 的部署策略推进环境，同时用 frozen teacher 对同一状态打标签，再对动作做 MSE。这已经是 on-policy behavior cloning，但还不是完整的 DAgger，也没有使用 teacher 的价值、隐状态或动作分布。

对 S45，最值得优先尝试的顺序是：

1. **DAgger-lite：teacher 接管混合 + 数据聚合**，解决 student 早期偏离 teacher 状态分布的问题。
2. **BC warm-start 后 PPO 微调**，用环境 reward 恢复闭环稳定性；保留 teacher action 作为正则，而不是让 reward 只做日志。
3. **teacher latent / terrain representation 蒸馏**，让深度 CNN 学习 teacher 的地形表征，通常比只拟合最终 action 更容易泛化。
4. **动作分布或风险加权损失**，最后再调；它不能替代前面三个闭环问题的修复。

## 方法对比

### Now You See That

这类工作采用 privileged-information teacher 与 raw-pixel student 的闭环训练：teacher 访问高度/地形等 privileged observation，student 只能访问视觉和本体感觉。关键不是固定 teacher 轨迹，而是覆盖 student 实际访问的状态分布，并通过 teacher 监督保持闭环可恢复性。

可迁移的技巧是：让 student 自己推进环境；目标不只关注逐帧动作误差，还要保持跨时间稳定；privileged teacher 的中间地形表征可作为比最终动作更稠密的监督。

### DAgger

当前的 student rollout + teacher label 是 DAgger 的最小在线部分，但缺少历史样本 replay、初期 teacher 接管、随训练衰减的 beta，以及终止前恢复状态优先采样。

### RMA / privileged latent

可以把 EMP teacher 的 height-map CNN 或 policy hidden feature 作为 privileged target，让 S45 的 depth CNN 预测投影后的 latent，再由 student policy 使用该 latent。监督信号比 26 维 action 更稠密。

### BC 后 RL fine-tuning

纯 BC 不会因 reward 权重改变而变好；当前 Distillation 中 reward 只进日志。实用目标是：

`L = L_task_PPO + lambda_bc * L_action + lambda_latent * L_latent`

其中 `lambda_bc` 随训练下降，使 student 能修正动力学或传感器差异，同时保留 teacher 的安全先验。

## 当前实现缺口

`packages/rsl_rl/rsl_rl/algorithms/distillation.py` 当前只有 frozen teacher action、确定性 student rollout、MSE/Huber action loss 和时间窗口梯度累积。它没有保存 teacher value、latent、log-std、接管标记或 replay buffer。因此 KL、advantage 加权、latent matching、DAgger 聚合和 BC+PPO 都不能只改一个 loss 名称实现。

新增损失必须在同一关节顺序下计算；当前 student 输出和 teacher wrapper 已统一为 MJCF 顺序。

## 推荐实施路线

### 阶段 A：DAgger-lite（最高优先级）

建议初始配置：`beta_start=1.0`、`beta_end=0.05`、`beta_decay_steps=30_000`、`replay_capacity=200_000`、`replay_ratio=0.5`。每步计算 student 和 teacher 动作，用 env mask 选择 teacher 或 student 执行，并保存 student 观测、teacher 观测和 teacher action。position action 不建议直接线性混合，因为中间关节目标可能不安全。

### 阶段 B：BC 初始化后 PPO 微调

在 action MSE、跌倒率和速度跟踪稳定后切换 PPO：

`loss = ppo_loss + lambda_bc * huber(student_mean, teacher_action)`

建议 `lambda_bc` 从 1.0 逐步降到 0.05~0.1。继续使用现有安全 reward；`track_default_arm_pos=+1.0` 作为姿态先验即可，不应恢复到 +3.0。

### 阶段 C：latent 蒸馏

从 ActorCriticCNN 暴露 height-map CNN pooled feature 或 actor MLP feature，student depth CNN 增加 projection head：

`L_latent = smooth_l1(student_latent, stopgrad(project(teacher_latent)))`

latent loss 从 0.1 起步，并按 action loss 数量级归一化；先冻结 projection 训练 5k~10k iterations，再联合训练。

### 阶段 D：动作损失细化

稳定后再尝试 Huber、按 torque limit/action scale 加权、teacher mean + log-std KL，以及 2~4 步短 horizon consistency。当前 EMP wrapper 只输出 action，不能假设存在可用 log-std。

## 不建议

离线 teacher trajectory BC 会放大 covariate shift；直接把 reward 加进 MSE 会产生量纲问题；长期开启高斯 student rollout 会把环境推离 teacher 分布；不同输入模态的 CNN 不应做参数 L2。

## 评估

每种方案至少比较 action MSE、速度误差、跌倒率、episode length、遮挡/深度噪声下的地形泛化，并运行 3 个 seed。当前在线 BC、DAgger-lite、BC+PPO、latent+action 和组合方案应使用固定 terrain/command 条件对比。

## 最终建议

先实现 teacher 接管衰减 + replay 的 DAgger-lite，再把 student 接到 PPO 做短程 reward fine-tuning；只有确认 action BC 稳定且视觉表征是瓶颈时，再加入 teacher latent loss。

## 参考工作

- Ross, Gordon, and Bagnell, **A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning**，DAgger 的数据聚合与 expert/learner 混合执行基础。
- Peng et al., **Learning Agile Robotic Locomotion Skills by Imitating Animals**，DeepMimic；说明动作/轨迹监督与物理闭环目标可以结合。
- Lee et al., **Learning Quadrupedal Locomotion over Challenging Terrain**，RMA；privileged teacher latent 与受限观测 adaptation 的代表性范式。
- Miki et al., **Learning Robust Perceptive Locomotion for Quadrupedal Robots in the Wild**，展示感知输入、teacher 先验和真实闭环鲁棒性的结合方式。
- **Now You See That: Learning End-to-End Humanoid Locomotion from Raw Pixels**，本文关注的 privileged teacher 到 raw-pixel student 路线；对 S45 最直接的启发是闭环 student 状态分布、teacher 的地形表征监督以及后续 reward-based refinement。

注：上述工作解决的问题不同，不能直接把其 loss 或网络结构照搬到 S45；本项目当前最缺的是数据分布与训练阶段设计，而不是再增加一个未经校准的损失项。
