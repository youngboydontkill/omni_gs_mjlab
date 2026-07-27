# PPO 训练中的 `mean action std`

> 适用范围:本项目所有基于 `VelocityOnPolicyRunner` + rsl-rl PPO 的任务(`Kuavo-S45/S54-Flat/Rough`、`Kuavo-S54-Head-CNN-Rough`、`Kuavo-S54-Rough` 等)。本文解释训练日志中 `Policy/mean_noise_std`(亦常被简称 `mean action std`)的含义、作用与影响因素,作为调参与排障参考。

## 1. 定义

rsl-rl 的 actor-critic 使用**高斯策略**:

$$\pi(a\mid s) = \mathcal{N}\!\left(\mu_\theta(s),\, \sigma\right)$$

- $\mu_\theta(s)$:actor 网络(MLP / CNN / DeFM)输出的**动作均值**。本项目里它最终通过 `JointPositionAction` 缩放为关节目标位置的偏移并下发到 MuJoCo / MuJoCo Warp。
- $\sigma$:**动作标准差**,在 rsl-rl 默认实现里是一个**与状态无关的可学习参数** `self.std`(`nn.Parameter`,形状 `[action_dim]`),由 `RslRlPpoActorCriticCfg.init_noise_std` 初始化(本项目默认 `1.0`)。

`mean action std` 即把该 `std` 向量沿动作维度取均值后记录的标量,反映**当前策略的探索噪声幅度**。

采样与对数概率公式:

$$
a_t = \mu_\theta(s_t) + \sigma \odot \varepsilon,\quad \varepsilon \sim \mathcal{N}(0, I)
$$

$$
\log \pi(a\mid s) = -\tfrac{1}{2} \sum_i \left[\left(\tfrac{a_i-\mu_i}{\sigma_i}\right)^2 + 2\log\sigma_i + \log 2\pi\right]
$$

## 2. 它会影响什么

### 2.1 探索 vs. 利用
- **σ 大** → 采样动作偏离均值多,rollout 覆盖更广的状态空间,容易触发新的接触/姿态;但单步动作抖动大,机器人在仿真里"毛糙",reward 方差大。
- **σ 小** → 动作几乎等于 μ,策略接近确定性;reward 方差小,但探索弱,容易陷入局部最优(典型表现:站着不走、只学会摆一条腿)。

### 2.2 PPO 各项 loss / 指标
- **`log_prob`、`entropy`**:熵 $H = \tfrac{1}{2}\sum_i \log(2\pi e\,\sigma_i^2)$ 随 σ 单调上升;σ 塌缩 → 熵骤降。
- **重要性比 `ratio = exp(logπ_new - logπ_old)`**:σ 很小时,μ 的微小变化会让 $(a-\mu)/\sigma$ 剧烈变化,`ratio` 容易爆炸 → 频繁触发 `clip_param` → 梯度被裁,学不动。
- **`approx_kl` / `desired_kl`**:本项目用 `adaptive` LR(`desired_kl=0.01`)。两高斯之间 KL:

$$
\mathrm{KL}(\pi_{\text{old}}\Vert\pi_{\text{new}}) = \sum_i\left[\log\tfrac{\sigma_{\text{new},i}}{\sigma_{\text{old},i}} + \tfrac{\sigma_{\text{old},i}^2 + (\mu_{\text{new},i}-\mu_{\text{old},i})^2}{2\sigma_{\text{new},i}^2} - \tfrac{1}{2}\right]
$$

σ 很小时分母 $\sigma_{\text{new}}^2$ 变小,KL 对 μ 更新极敏感 → adaptive LR 会被反复砍半。

### 2.3 动作落到环境上的物理效果
σ 的物理量纲是**关节角度噪声的弧度数**(再乘以 action scale)。
- σ ≈ 0.5 时关节目标抖动远超 PD 跟踪带宽,机器人脚底打滑。
- σ ≲ 0.05 基本不再做随机探索,只剩 μ 在驱动。

### 2.4 与 DeFM 特征缓存的关系
`doc/defm_feature_cache_optimization.md` 描述的特征缓存只在 encoder **冻结**且未启用 symmetry augmentation 时生效。**σ 是 actor 头部参数,不影响 encoder 输出是否被复用,缓存机制本身也不直接动 σ。** 但 encoder 输出的稳定性会影响 μ,从而间接影响 σ 的更新方向。

## 3. 它受什么影响

### 3.1 初始化 `init_noise_std`
启动时 `self.std` 被填成该常数。本项目默认 `1.0`,对应 ~57° 关节噪声,前几百步必然乱抖,属于正常现象。

### 3.2 PPO 目标对 σ 的梯度
对单步 surrogate $L = \mathbb{E}[\text{ratio}\cdot A]$,
$\partial L / \partial \log\sigma$ 大致正比于:

$$
\left[\left(\tfrac{a-\mu}{\sigma}\right)^2 - 1\right]\cdot A
$$

含义:
- **优势 $A > 0$** 的样本里,采样动作偏离 μ 较远($(a-\mu)^2/\sigma^2 > 1$)时,梯度推 σ **变大**;反之推 σ **变小**。
- **优势 $A < 0$** 的样本规律相反。

σ 的走向因此取决于「好动作是大噪声采到的,还是 μ 附近采到的」。任务 reward 鼓励精细控制 → σ 收敛下降;reward 主要靠大幅探索撞上的 → σ 保持高位。

### 3.3 Entropy bonus
`RslRlPpoAlgorithmCfg.entropy_coef` 在 loss 上加 $-c_{\text{ent}} H(\pi)$,直接把 σ 往大推。本项目里该系数通常很小(~`0.01`),不足以独立维持 σ,只是减缓塌缩。

### 3.4 `desired_kl` + adaptive LR
KL 超过 `2 × desired_kl` 自动降 LR,反之升 LR。如果 σ 急剧下降导致 KL 爆掉,adaptive 机制会把 LR 砍到 ~`1e-5`,表现为"学不动",**实际是 σ 失控的副作用**。

### 3.5 `clip_param`
PPO 的 ratio 裁剪间接约束 σ:σ 过小 → ratio 频繁出区间 → 梯度被清零 → σ 反而停在某个低位不再下降。

### 3.6 Reward / observation 尺度变化
- 给 `actor_obs` 加新观测(如最近为 `S45-DeFM-Rough` 加入真实速度观测)会让 μ 暂时不稳,σ 通常**先回升再回落**。
- reward 重新整形(权重大调、新 reward term)同理:正向 reward 的"动作甜区"挪了位置,σ 先扩张再收敛。

### 3.7 Symmetry augmentation / 多 GPU
启用 symmetry 时同一 transition 被镜像复制,样本方差结构改变,σ 收敛速度通常变快(且会**禁用 DeFM 特征缓存**)。多 GPU(`torchrunx`)只是 reduce 梯度,不会单独动 σ。

## 4. 健康范围(经验)

| 阶段 | `mean action std` 典型量级 | 说明 |
| --- | --- | --- |
| 启动 (`init_noise_std=1.0`) | ≈ 1.0 | 完全随机,机器人乱抖 |
| 早期 (前 200–500 iter) | 0.6 – 0.9 | 站立 / 起步学习 |
| 中期 (能走但不稳) | 0.3 – 0.6 | 主体行走 reward 上升 |
| 收敛后 (Flat/Rough 双足) | 0.15 – 0.35 | 仍保留必要探索 |
| **异常塌缩** | < 0.05 | 通常伴随 KL ≈ 0、reward 停滞、`ratio` clip 比例骤降到 0 |
| **异常发散** | > 1.2 且持续上升 | 通常伴随 NaN / 触发 reset,参考 `doc/nan_debugging.md` |

## 5. 在本项目里怎么排查

定位日志:`logs/rsl_rl/<experiment_name>/<run>/`(tensorboard 或 wandb)。**孤立看 σ 容易误判,需联合下列指标:**

- `Policy/mean_noise_std`
- `Loss/entropy`
- `Loss/surrogate`
- `Loss/learning_rate`
- `Loss/approx_kl`

### 5.1 σ 塌缩 + reward 不动
可能原因与对策:
- 提高 `entropy_coef`,或回调 `init_noise_std`。
- 检查是否某个 reward term 一边倒地惩罚动作幅度,例如 `action_rate`、`dof_vel`、pose reward 的 `std` 设得过严。
- 检查 advantage 是否长期接近 0(critic 没学好) → critic 学习率 / `obs_groups` 配置 / `critic_depth` 是否正确接入。

### 5.2 σ 不收敛 / 发散
- reward 是否长期为负 → 推动策略不停大幅探索。
- 观测尺度是否爆炸(NaN / inf) → μ 抖动,触发"大噪声反而 advantage 更高"的反馈;先参考 `doc/nan_debugging.md`。
- 是否近期新增了观测项或大幅改动 reward → 给一段时间让 σ 重新稳定后再判断。

### 5.3 与模型类型的关系
- `MLPModel` / `CNNModel` / `DefmModel` 三种 actor 的 σ 行为**没有本质差异**——σ 只属于 actor 头部。
- encoder 是否冻结、特征是否缓存(参见 `doc/defm_feature_cache_optimization.md`)不直接影响 σ,但 encoder 输出稳定性会通过 μ 间接影响 σ 的梯度方向。

## 6. 相关文件与设置

| 位置 | 作用 |
| --- | --- |
| `packages/rsl_rl/rsl_rl/modules/actor_critic*.py` | `self.std` 定义、采样、`log_prob`、`entropy` 实现 |
| `packages/rsl_rl/rsl_rl/algorithms/ppo.py` | surrogate loss、KL、adaptive LR、entropy bonus |
| `src/omni_gs_playground/tasks/velocity/config/<robot>/rl_cfg.py` | `init_noise_std`、`entropy_coef`、`desired_kl`、`clip_param` 等 |
| `src/omni_gs_playground/tasks/velocity/rl/runner.py` | `VelocityOnPolicyRunner`,checkpoint / ONNX 导出 |

## 7. 参考

- `doc/defm_feature_cache_optimization.md` — DeFM 特征缓存何时启用、与 symmetry / 冻结 encoder 的关系。
- `doc/nan_debugging.md` — NaN / σ 发散场景的排障流程。
- `doc/terrain_curriculum_stuck.md` — 课程相关导致 reward 停滞、间接表现为 σ 异常的案例。
