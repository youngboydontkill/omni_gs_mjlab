# AMP (Adversarial Motion Priors) 骨架 — 文档与接线指南

> 本文档说明代码仓库中已集成的 AMP 判别器骨架的架构、集成点、数据结构，以及如何在**获得 Kuavo 重定向运动数据后**启用完整训练。

---

## 状态

**骨架已就位，默认关闭。** 以下组件已实现在 `packages/rsl_rl/` 中，**但无参考数据集，因此不会自动构造**（`PPO.__init__` 仅在 `amp_cfg` 非 None 时构建判别器，而目前无任何 runner config 传入 `amp_cfg`）：

- ✅ `AMPDiscriminator` 模块（`rsl_rl/extensions/amp.py`）
  - 序列判别器 `D(S)`：MLP trunk → 单个 logit
  - 论文式(7) MSE loss：`E_ref[(D-1)²] + E_policy[(D+1)²]`
  - 论文式(8) quadratic style reward：`r = max(0, 1-0.25·(D-1)²)`
  - 梯度惩罚（可选，`grad_penalty_coeff`）
  - 自身 optimizer（Adam，自带 weight decay）
- ✅ PPO 集成钩子
  - `process_env_step()`：从 `extras["amp_obs"]` 取判别器观测 → style reward → 加权累加至 `transition.rewards`，观测缓存至 `transition.extra["amp_obs"]`
  - `update()`：当 `transition.extra` 含 `amp_obs` 且 `set_amp_reference_sampler` 已设置时，训练判别器；独立 optimizer 步
  - `set_amp_reference_sampler` setter 方法
  - 与 DeFM feature cache 互斥检查（同 symmetry）
  - loss 字典含 `amp_discriminator` 项

---

## AMP 观测格式

论文 §III-E 式(6) 定义判别器状态：

```
s_t = { v_t, ω_t, g_t, q_t, q̇_t }
```

其中 `v_t∈ℝ³`(基座线速度), `ω_t∈ℝ³`(角速度), `g_t∈ℝ³`(投影重力), `q_t∈ℝ²⁹`(关节位置), `q̇_t∈ℝ²⁹`(关节速度)。拼接得每个时间步 **dim = 3+3+3+29+29 = 67**。

判别器不消费单帧，而是消费**短序列**（论文里 `n` 帧历史）：

```
S_t = [s_{t-n}, …, s_t]    →   扁平 dim = 67 × (n+1)
```

当前 `AMPDiscriminator` 构造参数 `input_dim` 需要这个扁平后的总维度。

要启用 AMP 奖励，环境必须在 `extras["amp_obs"]` 中返回 `[B, input_dim]` 的张量。

### 配置示例（尚未启用）

```python
amp_cfg = {
    "input_dim": 67 * 5,        # 5 帧历史
    "hidden_dims": (256, 256),
    "activation": "elu",
    "reward_scale": 1.0,
    "grad_penalty_coeff": 10.0,
    "weight_decay": 1.0e-4,
    "learning_rate": 1.0e-3,
    "reward_coef": 0.5,         # style reward 混合系数（与 task reward 加权）
}
amp_cfg = None                  # ← 目前必须为 None，标签占位
```

传给 runner cfg 的 `algorithm` 字段。`RslRlPpoAlgorithmCfg` 未定义此字段（在 vendored mjlab 包中），但 PPO `__init__` 的 `**kwargs` 接收它。

---

## 如何启用（所需步骤）

1. **获取 Kuavo 重定向运动数据**：用 GMR (General Motion Retargeting, Araujo et al. 2025) 把 SMPL 行人/跑步轨迹或 MoCap 数据重定向到 Kuavo 运动学骨架。
2. **构建参考数据集 sampler**：实现 `sampler(n) → torch.Tensor [n, input_dim]`，每次调用返回 `n` 条从参考数据库均匀采样的（或按论文分 walk/run 两套数据集的）平坦状态序列。
3. **环境返回 AMP 观测**：在 env 的 `step()` 或 `compute_observations()` 中构造 `extras["amp_obs"]`。
4. **传递 `amp_cfg`**：在 runner config 的 `algorithm` 中传入非 None 的 `amp_cfg` dict。
5. **调用 `alg.set_amp_reference_sampler(sampler)`**：在 runner 构造 PPO 后 attach sampler。

### 注意事项

- **AMP 与 DeFM feature cache 互斥**：风格增强无法对齐缓存（同 symmetry augmentation）。启用 AMP 时需设 `share_cnn_encoders=False` 且关闭 feature cache。
- **分数据集训练**：论文把 walk 和 run 分两个独立策略训练，walk 数据来自 MPC + MoCap，run 来自 LAFAN。
- **对称增广**：AMP 的判别器观测可以做镜像增广（左右脚互换+关节符号反转）以倍增参考数据集——与 `process_env_step` 中的 mirror 逻辑联动。
