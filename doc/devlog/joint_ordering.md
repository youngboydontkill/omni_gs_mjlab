# 受控关节顺序的配置与对齐机制(以 S45 为主)

> 适用范围:本项目所有 Kuavo 速度跟踪任务(`Kuavo-S45/S54-Flat/Rough`、`Kuavo-S45-DeFM-Rough`、`Kuavo-S54-Head-CNN-Rough`)。本文回答两个问题:**策略观测 / 动作里的关节维度顺序是什么?这个顺序由代码里的哪些环节决定、为什么观测与动作能逐关节对齐?** 以 `Kuavo-S45-Rough`(26 自由度)为主线,S54 / S54-Head 在末尾给出差异。

## 0. TL;DR

S45 的 26 个受控关节,在**观测**(`joint_pos` / `joint_vel` / `last_action`)、**动作输出**(`actions`)、**策略 ONNX 输入/输出**中**完全同序**:

```
左腿 leg_l1..l6  →  右腿 leg_r1..r6  →  左臂 zarm_l1..l7  →  右臂 zarm_r1..r7
```

这个顺序不是显式写死的,而是「**MJCF 物理定义顺序**」经两条不同的解析路径(`preserve_order=True` 与 `preserve_order=False`)各自收敛到同一结果。能对齐是策略可训练的前提。

| 段 | 关节(共 26) |
| --- | --- |
| 左腿 ×6 | `leg_l1_joint, leg_l2_joint, leg_l3_joint, leg_l4_joint, leg_l5_joint, leg_l6_joint` |
| 右腿 ×6 | `leg_r1_joint, leg_r2_joint, leg_r3_joint, leg_r4_joint, leg_r5_joint, leg_r6_joint` |
| 左臂 ×7 | `zarm_l1_joint, zarm_l2_joint, zarm_l3_joint, zarm_l4_joint, zarm_l5_joint, zarm_l6_joint, zarm_l7_joint` |
| 右臂 ×7 | `zarm_r1_joint, zarm_r2_joint, zarm_r3_joint, zarm_r4_joint, zarm_r5_joint, zarm_r6_joint, zarm_r7_joint` |

## 1. 顺序的三个决定因素

### 1.1 物理源头 — MJCF 关节定义顺序

`src/omni_gs_playground/assets/robots/kuavo/biped_s45/xml/biped_s45_collision.xml` 里 `<joint>` 的出现顺序即上表。`kuavo_s45_constants.py:14` 的 `get_spec()` 在编译前删掉 `zhead_1_joint` / `zhead_2_joint`(S45 无头),因此 `Entity.joint_names` 正好是这 26 个,物理顺序就是导出顺序的最终基准。

### 1.2 受控关节正则 — 决定「哪些」关节入选

`src/omni_gs_playground/tasks/velocity/config/kuavo/env_cfgs.py:34`:

```python
S45_CONTROLLED_JOINTS = (
  r"leg_[lr][1-6]_joint",
  r"zarm_[lr][1-7]_joint",
)
```

这是**正则模式**,不是关节名:`[lr]` 同时匹配左/右,`[1-6]` / `[1-7]` 匹配编号。它的职责只是从 26 个关节里圈出全部腿臂关节(排除 `zhead_*`),**不直接决定顺序**。

### 1.3 两条解析路径 — 决定「最终维度顺序」

同一组关节会被两套不同的 mjlab 解析逻辑消费,它们各自把正则映射成最终的索引序列:

| 路径 | 消费者 | 入口 | 关键参数 |
| --- | --- | --- | --- |
| **观测路径** | `joint_pos` / `joint_vel` / reward 的 `pose` 等 | `_controlled_joints_cfg()` → `SceneEntityCfg(joint_names=..., preserve_order=True)`(`env_cfgs.py:46`) | `preserve_order=True` |
| **动作路径** | `actions` 输出、`last_action` | `JointPositionActionCfg(actuator_names=(".*",))` → `find_joints_by_actuator_names(".*")` | `preserve_order=False` |

两条路径的实现对齐情况见下节。

## 2. 为什么两条路径能对齐(核心机制)

### 2.1 观测路径:`preserve_order=True`

`_controlled_joints_cfg()` 把正则包进 `SceneEntityCfg(..., preserve_order=True)`,解析时调用 mjlab 的 `resolve_matching_names`(`mjlab/utils/lab_api/string.py:178`)。该函数对多 key 的处理是:

1. 先按 target(物理 `joint_names`)顺序遍历,记录每个关节命中了第几个 query key;
2. 若 `preserve_order=True`,**按 query key 分组重排**——所有命中 key 0(`leg_*`)的在前,所有命中 key 1(`zarm_*`)的在后,组内保留 target(物理)顺序。

由于 XML 本身就是「全部腿在前、全部臂在后」,分组重排后结果 = 物理顺序。

### 2.2 动作路径:`preserve_order=False`

`JointPositionActionCfg` 用 `actuator_names=(".*",)` 匹配全部执行器,实际走 `Entity.find_joints_by_actuator_names`(`mjlab/entity/entity.py:525`):

```python
actuated_in_natural_order = [name for name in self.joint_names if name in actuated_joint_names_set]
_, matched_joint_names = self.find_joints(
    actuator_name_keys, joint_subset=actuated_in_natural_order, preserve_order=False
)
```

它先把 `joint_names` **按物理顺序**过滤到「有执行器」的关节,再 `preserve_order=False` 匹配——结果仍是物理顺序。

> 注意:`kuavo_s45_constants.py:47` 的 `_ACTUATOR_PARAMS` 按 `leg_[lr]1..6 → zarm_[lr]1..7` 定义执行器,但这一顺序**不参与最终 action 维度排序**;`find_joints_by_actuator_names` 会把它重映射回物理顺序。

### 2.3 结论

观测路径按 query key 分组(腿→臂),动作路径按物理顺序过滤——两条独立逻辑恰好都落到「物理顺序」,因此 `joint_pos` / `joint_vel` / `last_action` 反馈与 26 维动作输出逐关节对齐。若未来改动 MJCF 关节定义顺序或正则分组,必须重新核对这一对齐(见 §4 探针)。

## 3. 相关的「顺序」细节

- **action_scale**:`kuavo_s45_constants.py:106` 的 `KUAVO_S45_ACTION_SCALE` 是 dict,key 是 `leg_[lr]1_joint` 等 13 个正则(每个匹配 l/r 两个关节),值由 `0.25 * effort_limit / stiffness` 推导。经 `JointPositionAction.scale` 应用时同样对齐到上述 26 维物理顺序。
- **默认位姿 / 相对观测**:`HOME_KEYFRAME.joint_pos`(`kuavo_s45_constants.py:74`)用 `leg_[lr]3_joint: -0.4` 等正则 key 设默认角;观测里的 `joint_pos_rel` / `joint_vel_rel` 是相对该默认位姿的偏差,维度顺序仍是上表。
- **导出(ONNX)**:actor 的 `actions` 输出为 `[1, 26]`(deterministic mean),维度顺序与本表一致;观测输入 `obs` 中的 `joint_pos(26)` / `joint_vel(26)` / `actions(26)` 三段同样按本表排列。详见观测输入与导出形态的相关讨论。

## 4. 复现探针(可直接运行)

下列脚本实例化 S45 `Entity` 并打印三条顺序,用于在任何改动后快速核对观测/动作对齐:

```bash
uv run python - <<'PY'
from mjlab.entity.entity import Entity
from omni_gs_playground.assets.robots.kuavo.biped_s45.kuavo_s45_constants import get_kuavo_s45_robot_cfg

S45_CONTROLLED_JOINTS = (r"leg_[lr][1-6]_joint", r"zarm_[lr][1-7]_joint")

robot = Entity(get_kuavo_s45_robot_cfg())
print("[1] 物理全关节:", list(robot.joint_names))
_, obs = robot.find_joints(S45_CONTROLLED_JOINTS, preserve_order=True)
print("[2] 观测顺序   :", obs)
_, act = robot.find_joints_by_actuator_names(".*")
print("[3] 动作顺序   :", act)
print("[4] 观测==动作 ?", obs == act)
PY
```

最近一次实测输出:三条顺序均为 `leg_l1..l6 → leg_r1..r6 → zarm_l1..l7 → zarm_r1..r7`,`obs == act == True`。

## 5. S54 / S54-Head 的差异

机制完全相同(顺序 = 各自 MJCF 物理顺序),仅受控关节集合与正则不同:

| 任务 | `controlled_joints` 正则 | 自由度 | 相对 S45 的增减 |
| --- | --- | ---: | --- |
| `Kuavo-S45-*` | `leg_[lr][1-6]` + `zarm_[lr][1-7]` | 26 | — |
| `Kuavo-S54-*` | `leg_[lr][1-6]` + `waist_yaw` + `zarm_[lr][1-7]` | 27 | + `waist_yaw_joint` |
| `Kuavo-S54-Head-CNN-Rough` | S54 + `zhead_[12]` | 29 | + `waist_yaw_joint` + `zhead_1/2_joint` |

`waist_yaw_joint` / `zhead_*_joint` 在各自 MJCF 中的物理位置决定它们插入 26 维序列的位置(例如 `waist_yaw` 在 `leg` 与 `zarm` 之间)。改动机器人资产或正则后,用 §4 探针(替换对应 `controlled_joints` 与 `get_*_robot_cfg`)核对即可。
