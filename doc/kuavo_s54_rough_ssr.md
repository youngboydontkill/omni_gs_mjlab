# Kuavo-S54-Rough-SSR

`Kuavo-S54-Rough-SSR` 是独立于已有 S54 任务的 SSR 迁移任务，参考论文
“SSR: Scaling Surefooted and Symmetric Humanoid Traversal to the Open World”
（arXiv:2605.30770）。本任务不包含 AMP 或其他动作风格奖励。

## 迁移内容

- Actor 使用 36×36 深度图和 5 帧本体历史。本体单帧沿用论文语义，但因
  Kuavo-S54 控制 27 个关节，维度由论文的 72 扩展为 90，动作输出由 21 扩展为 27。
- 网络宽度沿用论文附录：本体 MLP `[512, 256, 128]`、深度 CNN 通道
  `[32, 64, 128]`（kernel `[8, 4, 3]`、stride `[4, 2, 2]`）、256 维 GRU、
  三个 16 维 latent、3 维速度估计器，以及 5 个 `[1024, 512, 128]` 专家。
- Critic 使用真实基座速度、足端状态，以及 body/foot raycast 高度图，不依赖部署深度图。
- 启用 Kuavo-S54 关节轴和左右链对应关系定义的镜像 PPO 数据增强。
- 启用论文 hybrid prediction loss：body height map `2.0`、foot height map `1.0`、
  next proprioception `5.0`、VAE KL `1.0`、base velocity `2.0`。
- locomotion 奖励的表达式、方差和权重按论文表 6 迁移。机器人相关的基座高度、
  关节力矩限制和归一化系数统一取自 `kuavo_s54_constants.py`。
- 足底 raycast 按 S54 MJCF 的实际碰撞足底对齐：足底主体约为 24.4 cm × 10 cm，
  相对踝部坐标系向前偏移约 5.1 cm；扫描使用独立的 sole-center site 和
  25 cm × 10 cm、2.5 cm 间隔的 11×5 网格计算 support deficiency，组权重为 `0.25`。

## 与论文实现的边界

论文的摆动足分支另训一个高斯未来接触模型，并在 latent 空间追加镜像轨迹；当前
MJLab RewardManager 无法直接读取策略内部预测，因此本任务对摆动足使用当前足底投影的
稠密 pre-contact support deficiency，并在 PPO mini-batch 中执行完整观测镜像。这保留了
安全落脚的提前反馈和左右对称学习，但不是论文 imagination/latent augmentation 的逐位复现。
此外，当前 RSL-RL rollout 只返回合计奖励，因此使用一个 asymmetric critic，而不是论文按
locomotion、foothold、style 分组的三个 critic；style 组已按任务要求删除。

足底两个 11×5 ray map 和 body 9×9 ray map 会增加 raycast、critic 观测和 rollout
存储开销；并行环境数量较大时应根据显存逐步放大 `--env.scene.num-envs`。

## 使用

```bash
uv run python scripts/train.py Kuavo-S54-Rough-SSR --help
uv run python scripts/train.py Kuavo-S54-Rough-SSR
uv run python scripts/play.py Kuavo-S54-Rough-SSR \
  --checkpoint-file logs/rsl_rl/kuavo_s54_rough_ssr/<run>/model_<iteration>.pt
```
