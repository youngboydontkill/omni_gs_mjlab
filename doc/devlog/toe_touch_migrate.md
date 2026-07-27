# toe_touch 迁移文档：脚尖立面惩罚（IsaacLab → MJLab）

> 姊妹文档：`doc/reward_migrate.md`（S45 EMP 奖励集整体重写）。本文件只记录
> `toe_touch`（惩罚脚尖接触/逼近立面）这一项从 Leju-IsaacLab 迁移到本项目
> MJLab 栈的改动与 API 差异。文档默认中文。

## 1. 迁移目标

在 Kuavo S45 的 EMP 奖励集中新增 `toe_touch`：当脚尖（前向）逼近或接触一个
**立面**（楼梯竖板、墙面）时给惩罚。每只脚挂一条**前向单射线** raycaster，用射线
测得的距离与脚长比较，越近惩罚越大。

作用范围：两个 S45 任务（CNN 版 `Kuavo-S45-Rough` 与 DeFM 版
`Kuavo-S45-DeFM-Rough`）——都经由 `_apply_s45_emp_rewards()` 共享同一套奖励，故
改动集中在该函数里。S54 系列不受影响。

## 2. 改了哪些文件

### 2.1 `src/omni_gs_playground/tasks/velocity/mdp/rewards.py`

- import 行补 `RayCastSensor`：
  `from mjlab.sensor import BuiltinSensor, ContactSensor, RayCastSensor`。
- 新增奖励函数 `toe_touch(env, sensor_name_l, sensor_name_r, feet_length=0.178, margin=0.01)`：
  - 通过 `env.scene[sensor_name]` 取 `RayCastSensor`（与 `height_scan` 同一取法）。
  - 单射线传感器 → 取 `sensor.data.distances[:, 0]`，形状 `[B]`。
  - 距离 → 惩罚的三段式（cubic ramp）：

    | 距离区间 | 惩罚 |
    | --- | --- |
    | `dist <= feet_length` | `1.0`（已到/越过立面） |
    | `feet_length < dist <= feet_length + margin` | `(1 - x)^3`，`x = (dist - feet_length) / margin` |
    | `dist > feet_length + margin` | `0.0`（无接触） |

  - 返回 `penalty_l + penalty_r`（左右脚各 `[0,1]`，求和）。

### 2.2 `src/omni_gs_playground/tasks/velocity/config/kuavo/env_cfgs.py`

在 `_apply_s45_emp_rewards()` 内：

1. import 行补 `GridPatternCfg, ObjRef, RayCastSensorCfg`。
2. 紧随 `undesired_body_contact_cfg` 之后，新增两个前向 raycaster 并 append 到
   `cfg.scene.sensors`：
   - `feet_l_forward_scanner` → `frame=ObjRef("body", "leg_l6_link", "robot")`
   - `feet_r_forward_scanner` → `frame=ObjRef("body", "leg_r6_link", "robot")`
   - 共用参数：
     `pattern=GridPatternCfg(size=(0.0, 0.0), resolution=0.01, direction=(1.0, 0.0, 0.0))`、
     `ray_alignment="base"`、`max_distance=1.0`、`exclude_parent_body=True`、
     `debug_vis=True`。
3. 在 `cfg.rewards.update({...})` 里注册奖励项：

   ```python
   "toe_touch": RewardTermCfg(
       func=mdp.toe_touch,
       weight=-1.0,   # 函数返回正值 penalty，本框架靠负权重变成惩罚
       params={
           "sensor_name_l": "feet_l_forward_scanner",
           "sensor_name_r": "feet_r_forward_scanner",
           "feet_length": 0.178,
           "margin": 0.01,
       },
   ),
   ```

   `weight=-1.0` 与同类接触/姿态惩罚（`feet_stumble`、`undesired_contacts`、
   `feet_too_near`）同量级。

## 3. 与 IsaacLab 的结构差异

用户给的原始代码是 IsaacLab（`RayCasterCfg` + `RayCaster`）风格；本项目是 MJLab 栈，
API 差别较大。逐项对照：

| 方面 | IsaacLab（原代码） | MJLab（本项目） |
| --- | --- | --- |
| 传感器配置类 | `RayCasterCfg` | `RayCastSensorCfg` |
| 传感器实例类 | `RayCaster` | `RayCastSensor` |
| 挂载方式 | `prim_path="{ENV_REGEX_NS}/Robot/leg_l6_link"` | `frame=ObjRef(type="body", name="leg_l6_link", entity="robot")` |
| **原点偏移** | `offset=OffsetCfg(pos=(0.0, 0.0, -0.05))` | **不支持** —— 射线原点恒为物理 frame 位置（body/site/geom 的 `xpos`） |
| 命中网格过滤 | `mesh_prim_paths=["/World/ground"]` | `include_geom_groups`（默认 `(0,1,2)`）+ `exclude_parent_body=True` |
| 单射线 pattern | `GridPatternCfg(resolution=0.01, size=(0.0, 0.0), direction=(1.0, 0.0, 0.0))` | 同名同参 `GridPatternCfg(size=(0.0,0.0), resolution=0.01, direction=(1.0,0.0,0.0))` |
| 距离取法 | `torch.norm(data.ray_hits_w[:,0,:] - data.pos_w, dim=-1)` | `data.distances[:, 0]`（框架直接给距离） |
| **未命中语义** | 命中点是有效世界坐标，靠 `nan_to_num` 兜 NaN/Inf | `distances == -1` 表示 miss / 超出 `max_distance` |
| 奖励入参风格 | `sensor_cfg: SceneEntityCfg` | `sensor_name: str`（与本仓库其它奖励一致） |
| 传感器接入场景 | scene 中声明 `RayCasterCfg` 字段 | append 到 `cfg.scene.sensors` 元组 |

### 3.1 未命中处理：从「norm」改为「distances」（重要正确性差异）

IsaacLab 原代码用 `norm(ray_hits_w - pos_w)` 算距离，靠 `nan_to_num` 处理 NaN/Inf。
**这个写法照搬到 MJLab 会出 bug**：MJLab 里未命中的射线，其 `hit_pos_w` 会**塌缩到
射线原点**（`origin + ray * max(distance, 0)`，miss 时 `distance` 被 clamp 到 0），
于是 `norm(hit - pos)` 恒为 `0`，会被误判成「距离 0 = 已接触」，从而每步误触发满额
惩罚。

因此迁移时改用框架提供的 `data.distances`：
- `distances < 0`（即 `-1`，miss 或超 `max_distance`）→ 视为无接触，映射到大距离
  `feet_length + margin + 1.0`；
- 再叠一层 `nan_to_num` 兜 NaN/Inf → 同样映射到该大距离。

cubic ramp 的数学与原代码完全一致（已用 mock 传感器单测：命中=1.0、边界=1.0、
中点 x=0.5→0.125、超出=0、miss/NaN/Inf 全=0）。

### 3.2 没有 `OffsetCfg`：射线原点在踝关节而非脚底

MJLab 的 `RayCastSensorCfg` 没有 IsaacLab 的原点偏移。IsaacLab 原代码把射线原点下移
`-0.05m`（贴近脚底）；MJLab 里射线只能从 `leg_[lr]6_link` 的 body frame 发出，而该
frame 在**踝关节**处，据 XML 脚底碰撞几何在 `z≈-0.0595`，即原点比脚底高约 5–6cm。

影响与取舍：前向射线仍能探到正前方的立面，且该偏移对左右脚对称，不引入左右偏置。
代价是触发距离的**基准点**比 IsaacLab 略高——`feet_length=0.178` 这个默认值是相对
`leg_6_link` frame 的前向可达距离，未按真实脚底重新标定。若可视化发现触发距离不对，
调 `feet_length` 即可（见验证）。

## 4. 验证

```bash
# 1) 任务仍能注册
uv run python scripts/list_envs.py --keyword Kuavo   # 6 个任务齐全

# 2) 两个 S45 config 实例化：传感器 + 奖励项均在位（已确认）
#    sensors 含 feet_l/r_forward_scanner；cfg.rewards["toe_touch"] weight=-1.0

# 3) reward 数学单测（mock 传感器）：三段式 + miss/NaN/Inf 全部通过

# 4) 可视化确认前向射线（debug_vis=True）并按需标定 feet_length
uv run python scripts/play.py Kuavo-S45-Rough --agent zero --num-envs 1 --viewer viser
```

## 5. 显存/性能说明

每只脚只有一条射线（`size=(0,0)`），两只脚共 2 条，`max_distance=1.0`。相对已有的
`terrain_scan`（16×10 网格）与深度相机，raycast 与存储开销可忽略；不进入 rollout
`extra` 缓存，不影响 DeFM 特征缓存链路。