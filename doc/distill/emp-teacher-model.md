# EMP Teacher 模型结构文档（蒸馏用）

> 任务: `Legged-Isaac-Velocity-Rough-Kuavo-S42-Emp-v0`
>
> 用途: 从 `.pt` checkpoint 加载训练好的 Teacher 模型，蒸馏到 Student 网络

---

## 一、Checkpoint 文件结构（.pt）

EMP runner 保存的 `.pt` 是一个 dict，key 如下：

```python
checkpoint = torch.load("model_X.pt")
checkpoint.keys()
# => dict_keys([
#     "model_state_dict",          # ActorCriticCNN 的 state_dict
#     "optimizer_state_dict",      # PPO optimizer
#     "discriminator_state_dict",  # Discriminator 的 state_dict
#     "iter",                      # 当前训练迭代数
#     "infos",                     # 附加信息 (可能为 None)
# ])
```

**蒸馏只需要 `model_state_dict`**（即 `ActorCriticCNN` 的参数）。

---

## 二、模型结构总览（ActorCriticCNN）

```
┌──────────────────────────────────────────────────────────────────┐
│  ActorCriticCNN                                                  │
│                                                                  │
│  1. actor_obs_normalizer: EmpiricalNormalization(423)            │
│     → 对 [command + policy] 这 423 维做经验归一化                     │
│                                                                  │
│  2. height_map_cnn: PolicyHeightMapCNN(H=9, W=7)                │
│     → 输入 height_scan_clip [B, 63], 输出 CNN 特征 [B, 64]        │
│                                                                  │
│  3. actor MLP: MLP(487 → 512 → 256 → 128 → 26, activation=elu) │
│     → 输入 = concat([归一化后obs(423), CNN特征(64)]) = [B, 487]    │
│     → 输出 = 26 维 action (关节目标位置)                              │
│                                                                  │
│  4.* critic: 蒸馏不需要, 忽略                                       │
│  5.* precise_height_map_cnn: 蒸馏不需要, 忽略                       │
└──────────────────────────────────────────────────────────────────┘
```

### 2.1 `PolicyHeightMapCNN` 细节

```python
PolicyHeightMapCNN(H=9, W=7, output_dim=64)
  ├── Conv2d(1→32, k=3, s=1, p=1) + LeakyReLU(0.2)  # → [B,32,9,7]
  ├── Conv2d(32→64, k=3, s=2, p=1) + LeakyReLU(0.2) # → [B,64,5,4]
  ├── AdaptiveAvgPool2d(1)                            # → [B,64,1,1]
  ├── FC(64→64) + LayerNorm(64)                      # → [B,64]
  └── return: [B, 64, T=1] → squeeze → [B, 64]
```

**前处理**（在 `forward` 内部）:
```python
# height_map: [B, 63, T]
min_height = height_map.min(dim=1, keepdim=True).values
height_map = height_map - min_height                      # 减最小值归一化
h = height_map.permute(0, 2, 1)                           # [B, T, 63]
h = h.reshape(B * T, 1, 7, 9)                             # → reshape 为 2D 网格
h = h.permute(0, 1, 3, 2).contiguous()                    # → [B, 1, 9, 7] (H,W)
```

---

## 三、Actor 输入结构（蒸馏时你需要提供的观测）

### 3.1 整体流程

```
obs_dict (TensorDict)
  ├── "command"    : [B, 3]    ─┐
  ├── "policy"     : [B, 420]  ─┤── concat(dim=-1) → [B, 423]
  └── "perception" : [B, 63]   ─┘── CNN → [B, 64]
                                            │
                                    concat(dim=-1) → [B, 487] → actor MLP → [B, 26]
```

蒸馏时等价于构造 `[B, 486]` 的 flat 向量：
- 前 423 维 → 过 `actor_obs_normalizer`
- 后 63 维 → 过 `height_map_cnn`
- concat → `actor`

### 3.2 Flat 向量排列（486 维）

```
位置       大小  内容
────────────────────────────────────────────────────────
0-2         3    cmd_vel: [vx, vy, wz] (当前帧)
────────────────────────────────────────────────────────
3-17        15   base_ang_vel: [t0(最老), ..., t4(最新)]，每帧 3 维
18-32       15   gravity:      [t0(最老), ..., t4(最新)]，每帧 3 维
33-162     130   joint_pos:    [t0(最老), ..., t4(最新)]，每帧 26 维
163-292    130   joint_vel:    [t0(最老), ..., t4(最新)]，每帧 26 维
293-422    130   action:       [t0(最老), ..., t4(最新)]，每帧 26 维
────────────────────────────────────────────────────────
423-485     63   height_scan_clip (CNN 输入, 仅当前帧)
```

这是 Isaac Lab/MJLab 在 `concatenate_terms=True` 与
`flatten_history_dim=True` 下的 **term-major** 布局。不要转换为五个连续
84 维帧的 frame-major 布局；checkpoint 的 `actor_obs_normalizer` 以该 term-major
维度顺序训练。

### 3.3 每个历史帧的 policy 内容

```
base_ang_vel(3) + gravity(3) + joint_pos(26) + joint_vel(26) + action(26) = 84
    0-2            3-5          6-31          32-57          58-83
```

### 3.4 关节顺序（checkpoint 的 USD/Lab 顺序）

MJCF 的环境原生顺序为：

```python
[leg_l1, leg_l2, leg_l3, leg_l4, leg_l5, leg_l6,
 leg_r1, leg_r2, leg_r3, leg_r4, leg_r5, leg_r6,
 zarm_l1, zarm_l2, zarm_l3, zarm_l4, zarm_l5, zarm_l6, zarm_l7,
 zarm_r1, zarm_r2, zarm_r3, zarm_r4, zarm_r5, zarm_r6, zarm_r7]
```

checkpoint 的 USD/Lab 顺序为：

| idx | 关节 | idx | 关节 |
|-----|------|-----|------|
| 0 | leg_l1_joint | 1 | leg_r1_joint |
| 2 | zarm_l1_joint | 3 | zarm_r1_joint |
| 4 | leg_l2_joint | 5 | leg_r2_joint |
| 6 | zarm_l2_joint | 7 | zarm_r2_joint |
| 8 | leg_l3_joint | 9 | leg_r3_joint |
| 10 | zarm_l3_joint | 11 | zarm_r3_joint |
| 12 | leg_l4_joint | 13 | leg_r4_joint |
| 14 | zarm_l4_joint | 15 | zarm_r4_joint |
| 16 | leg_l5_joint | 17 | leg_r5_joint |
| 18 | zarm_l5_joint | 19 | zarm_r5_joint |
| 20 | leg_l6_joint | 21 | leg_r6_joint |
| 22 | zarm_l6_joint | 23 | zarm_r6_joint |
| 24 | zarm_l7_joint | 25 | zarm_r7_joint |

MJCF 到 checkpoint USD/Lab 的重排索引：

```python
mjcf2lab = [0, 6, 12, 19, 1, 7, 13, 20, 2, 8, 14, 21,
            3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25]
lab2mjcf = [0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 21,
            2, 6, 10, 14, 18, 22, 24, 3, 7, 11, 15, 19, 23, 25]
```

### 3.5 动作输出（26 维）

`actor_out` 经 **scale=0.25** 后作为关节目标位置下发。动作维度顺序与上面的 checkpoint USD/Lab 顺序一致。

---

## 四、Height-Scan 数据格式

### 4.1 原始传感器

- 模式: `GridPattern(resolution=0.1, size=[1.6, 1.0])`
- 原始网格: 17(L) × 11(W) = **187 点**
- 传感器位置: base_link 上方 20m, 向下打点

### 4.2 裁剪（obs 中实际使用的）

```python
# height_scan = sensor_height(20) - hit_z - offset(0.5)   # [B, 187]
height_scan_2d = height_scan.view(-1, 11, 17)    # [B, W=11, L=17]
height_scan_clip = height_scan_2d[:, 2:9, 8:17]  # [B, 7, 9] 裁剪
height_scan_clip = height_scan_clip.flatten(1, 2) # [B, 63]
# 替换 NaN → 0.83, Inf → 1.378
# 最终 = raw_height - 0.5
```

- 最终形状: **[B, 63]** → 7行(宽度/ y轴) × 9列(长度/ x轴)
- 排列顺序: **C-major (row-major)**，即 `[r0c0, r0c1, ..., r0c8, r1c0, ...]`
- 覆盖范围: 前方约 0.8m × 宽 0.6m

---

## 五、蒸馏时需要的完整推理代码

```python
import torch
from rsl_rl.modules import ActorCriticCNN, PolicyHeightMapCNN


def build_teacher(checkpoint_path, device="cuda"):
    """从 .pt checkpoint 重建 teacher 模型。"""
    actor_cnn = PolicyHeightMapCNN(9, 7)
    critic_cnn = PolicyHeightMapCNN(9, 7)

    obs_groups = {
        "policy": ["command", "policy"],
        "critic": ["command", "privileged"],
        "perception": ["perception"],
        "precise_perception": ["precise_perception"],
        "discriminator": ["style"],
    }

    # dummy obs 用于 __init__ 计算维度
    dummy_obs = {
        "command": torch.zeros(1, 3),
        "policy": torch.zeros(1, 420),
        "privileged": torch.zeros(1, 197),
        "perception": torch.zeros(1, 63),
        "precise_perception": torch.zeros(1, 63),
        "style": torch.zeros(1, 116),
    }

    teacher = ActorCriticCNN(
        dummy_obs, obs_groups, num_actions=26,
        height_map_cnn=actor_cnn, precise_height_map_cnn=critic_cnn,
        actor_obs_normalization=True, critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128],
        activation="elu", init_noise_std=1.0, noise_std_type="log",
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    teacher.load_state_dict(checkpoint["model_state_dict"])
    teacher.eval()
    return teacher


@torch.no_grad()
def teacher_inference(teacher, cmd_vel, policy_obs, height_scan):
    """
    Args:
        cmd_vel:     [B, 3]    vx, vy, wz (当前帧)
        policy_obs:  [B, 420]  5帧历史, term-major 排列
        height_scan: [B, 63]   7×9 裁剪高程图
    Returns:
        actions:     [B, 26]   关节目标位置 (未缩放)
    """
    obs = teacher.actor_obs_normalizer(
        torch.cat([cmd_vel, policy_obs], dim=-1)
    )                                          # [B, 423]

    height_feat = teacher.height_map_cnn(
        height_scan.unsqueeze(-1)
    ).squeeze(-1)                              # [B, 64]

    obs_perceptive = torch.cat([obs, height_feat], dim=-1)  # [B, 487]
    return teacher.actor(obs_perceptive)       # [B, 26]
```

---

## 六、观测归一化器

`actor_obs_normalizer` 是 `EmpiricalNormalization(423)`:

```python
normalizer = teacher.actor_obs_normalizer
# normalizer.mean: [423], normalizer.var: [423], normalizer.count: int
```

蒸馏时 student 的输入范围需要与 teacher 一致，或者 student 也学一个归一化器。

---

## 七、蒸馏方案建议

Teacher 的 `act_inference` 完整推理流程：

```
actor_obs = concat([cmd_vel, policy_obs])                   # [B, 423]
norm_obs = actor_obs_normalizer(actor_obs)                  # [B, 423]
cnn_feat = height_map_cnn(height_scan.unsqueeze(-1))        # [B, 64]
final_input = concat([norm_obs, cnn_feat])                  # [B, 487]
action = actor(final_input)                                  # [B, 26]
```

建议的蒸馏 loss：

```python
loss = F.mse_loss(student_action, teacher_action.detach())
```

如果 student 没有 height-map CNN，也可以把 `cnn_feat` 作为 teacher 的中间表征来蒸馏：

```python
loss = (F.mse_loss(student_action, teacher_action.detach()) +
        0.1 * F.mse_loss(student_feat, cnn_feat.detach()))
```

**关于 history**: Teacher 需要 5 帧历史（term-major 排列）。如果你的 student 使用更少或没有历史帧，建议维护一个循环 buffer 缓存最近 5 帧观测来构造 teacher 输入。
