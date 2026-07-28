"""
Kuavo S54 constants. 
需要注意, 这里没有像Unitree G1那样使用`reflected_inertia_from_two_stage_planetary`
来计算惯量, 因为Kuavo S54的XML已经直接给出了每个关节的惯量参数.
"""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

##
# MJCF and assets.
##

KUAVO_S54_XML: Path = Path(__file__).parent / "xml" / "biped_s54.xml"
assert KUAVO_S54_XML.exists()

KUAVO_S54_HEAD_GEOM_GROUP = 5

# Foot scanner geometry.  The S54 collision sole in ``biped_s54.xml`` spans
# approximately x=[-0.071, 0.173] m and y=[-0.050, 0.050] m in each
# ``leg_[lr]6_link`` frame.  Round the length to the 2.5 cm ray spacing and
# retain the physical 10 cm load-bearing width.  Dedicated sites keep this
# forward offset separate from the force/torque sensor frames at the ankle.
KUAVO_S54_SOLE_SCAN_SITE_NAMES = (
  "l_sole_scan_center",
  "r_sole_scan_center",
)
KUAVO_S54_SOLE_SCAN_CENTER = (0.051, 0.0, 0.0)
KUAVO_S54_SOLE_SCAN_SIZE = (0.25, 0.10)
KUAVO_S54_SOLE_SCAN_RESOLUTION = 0.025
KUAVO_S54_SOLE_SCAN_SHAPE = (5, 11)
# Furthest toe capsule center (0.165 m) plus its 8 mm collision radius.
KUAVO_S54_TOE_REACH = 0.173


def _add_sole_scan_sites(spec: mujoco.MjSpec) -> None:
  for body_name, site_name in zip(
    ("leg_l6_link", "leg_r6_link"), KUAVO_S54_SOLE_SCAN_SITE_NAMES
  ):
    spec.body(body_name).add_site(
      name=site_name,
      pos=KUAVO_S54_SOLE_SCAN_CENTER,
      size=(0.005,),
      group=3,
    )


def get_spec() -> mujoco.MjSpec:
  """Load the S54 spec with fixed head links and position-controlled joints."""
  spec = mujoco.MjSpec.from_file(str(KUAVO_S54_XML))
  for actuator in tuple(spec.actuators):
    spec.delete(actuator)
  for sensor in tuple(spec.sensors):
    if sensor.name.startswith(("zhead_1_joint_", "zhead_2_joint_")):
      spec.delete(sensor)
  spec.delete(spec.joint("zhead_1_joint"))
  spec.delete(spec.joint("zhead_2_joint"))
  _add_sole_scan_sites(spec)
  return spec


def get_spec_with_head() -> mujoco.MjSpec:
  """Load the S54 spec with movable head links and position-controlled joints."""
  spec = mujoco.MjSpec.from_file(str(KUAVO_S54_XML))
  for actuator in tuple(spec.actuators):
    spec.delete(actuator)
  # for body_name in ("zhead_1_link", "zhead_2_link"):
  #   for geom in spec.body(body_name).geoms:
  #     geom.group = KUAVO_S54_HEAD_GEOM_GROUP
  _add_sole_scan_sites(spec)
  return spec


##
# Actuator config.
##

# The gains, effort limits, armatures, and delay range match KuavoArticulationCfg
# in kuavo.py. The XML head motors are intentionally omitted because kuavo.py does
# not define S54 control parameters for them.
def _position_actuator(
  target_names_expr: tuple[str, ...],
  *,
  stiffness: float,
  damping: float,
  effort_limit: float,
  armature: float,
) -> BuiltinPositionActuatorCfg:
  return BuiltinPositionActuatorCfg(
    target_names_expr=target_names_expr,
    stiffness=stiffness,
    damping=damping,
    effort_limit=effort_limit,
    armature=armature,
    delay_min_lag=0,
    delay_max_lag=4,
  )


KUAVO_S54_LEG_1_ACTUATOR = _position_actuator(
  ("leg_[lr]1_joint",),
  stiffness=48.0,
  damping=5.0,
  effort_limit=100.0,
  armature=0.05,
)
KUAVO_S54_LEG_2_ACTUATOR = _position_actuator(
  ("leg_[lr]2_joint",),
  stiffness=48.0,
  damping=5.0,
  effort_limit=50.5,
  armature=0.025,
)
KUAVO_S54_LEG_3_ACTUATOR = _position_actuator(
  ("leg_[lr]3_joint",),
  stiffness=68.0,
  damping=5.0,
  effort_limit=100.0,
  armature=0.025,
)
KUAVO_S54_LEG_4_ACTUATOR = _position_actuator(
  ("leg_[lr]4_joint",),
  stiffness=68.0,
  damping=6.0,
  effort_limit=150.0,
  armature=0.05,
)
KUAVO_S54_LEG_5_ACTUATOR = _position_actuator(
  ("leg_[lr]5_joint",),
  stiffness=18.0,
  damping=7.5,
  effort_limit=70.0,
  armature=0.05,
)
KUAVO_S54_LEG_6_ACTUATOR = _position_actuator(
  ("leg_[lr]6_joint",),
  stiffness=18.0,
  damping=7.5,
  effort_limit=50.0,
  armature=0.05,
)
KUAVO_S54_WAIST_ACTUATOR = _position_actuator(
  ("waist_yaw_joint",),
  stiffness=30.0,
  damping=3.0,
  effort_limit=33.0,
  armature=0.025,
)
KUAVO_S54_ARM_1_ACTUATOR = _position_actuator(
  ("zarm_[lr]1_joint",),
  stiffness=30.0,
  damping=3.0,
  effort_limit=30.0,
  armature=0.025,
)
KUAVO_S54_ARM_2_ACTUATOR = _position_actuator(
  ("zarm_[lr]2_joint",),
  stiffness=30.0,
  damping=3.0,
  effort_limit=30.0,
  armature=0.02,
)
KUAVO_S54_ARM_3_ACTUATOR = _position_actuator(
  ("zarm_[lr]3_joint",),
  stiffness=15.0,
  damping=3.0,
  effort_limit=20.0,
  armature=0.02,
)
KUAVO_S54_ARM_4_ACTUATOR = _position_actuator(
  ("zarm_[lr]4_joint",),
  stiffness=30.0,
  damping=3.0,
  effort_limit=30.0,
  armature=0.02,
)
KUAVO_S54_ARM_5_ACTUATOR = _position_actuator(
  ("zarm_[lr]5_joint",),
  stiffness=15.0,
  damping=3.0,
  effort_limit=14.1,
  armature=0.01,
)
KUAVO_S54_ARM_6_ACTUATOR = _position_actuator(
  ("zarm_[lr]6_joint",),
  stiffness=15.0,
  damping=3.0,
  effort_limit=14.1,
  armature=0.01,
)
KUAVO_S54_ARM_7_ACTUATOR = _position_actuator(
  ("zarm_[lr]7_joint",),
  stiffness=15.0,
  damping=3.0,
  effort_limit=14.1,
  armature=0.01,
)
KUAVO_S54_HEAD_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(r"zhead_[12]_joint",),
  stiffness=10.0,
  damping=1.0,
)


##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.965),
  rot=(1.0, 0.0, 0.0, 0.0),
  joint_pos={
    # Legs — 参考 depth_loco_param.info defaultJointState (0-5: left, 6-11: right).
    "leg_[lr]1_joint": 0.0,
    "leg_[lr]2_joint": 0.0,
    "leg_[lr]3_joint": -0.24,
    "leg_[lr]4_joint": 0.5,
    "leg_[lr]5_joint": -0.26,
    "leg_[lr]6_joint": 0.0,
    # Waist.
    "waist_yaw_joint": 0.0,
    # Arms — 参考 defaultJointState (13-19: left, 20-26: right).
    "zarm_l1_joint": 0.126,
    "zarm_l2_joint": 0.1,
    "zarm_l3_joint": 0.0,
    "zarm_l4_joint": -0.27,
    "zarm_l5_joint": 0.0,
    "zarm_l6_joint": 0.0,
    "zarm_l7_joint": 0.0,
    "zarm_r1_joint": 0.126,
    "zarm_r2_joint": -0.1,
    "zarm_r3_joint": 0.0,
    "zarm_r4_joint": -0.27,
    "zarm_r5_joint": 0.0,
    "zarm_r6_joint": 0.0,
    "zarm_r7_joint": 0.0,
    # Head (used by head-controlled variant).
    "zhead_.*_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

KUAVO_S54_DEFAULT_BASE_HEIGHT = HOME_KEYFRAME.pos[2]


##
# Final config.
##

KUAVO_S54_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    KUAVO_S54_LEG_1_ACTUATOR,
    KUAVO_S54_LEG_2_ACTUATOR,
    KUAVO_S54_LEG_3_ACTUATOR,
    KUAVO_S54_LEG_4_ACTUATOR,
    KUAVO_S54_LEG_5_ACTUATOR,
    KUAVO_S54_LEG_6_ACTUATOR,
    KUAVO_S54_WAIST_ACTUATOR,
    KUAVO_S54_ARM_1_ACTUATOR,
    KUAVO_S54_ARM_2_ACTUATOR,
    KUAVO_S54_ARM_3_ACTUATOR,
    KUAVO_S54_ARM_4_ACTUATOR,
    KUAVO_S54_ARM_5_ACTUATOR,
    KUAVO_S54_ARM_6_ACTUATOR,
    KUAVO_S54_ARM_7_ACTUATOR,
  ),
  soft_joint_pos_limit_factor=0.95,
)
KUAVO_S54_HEAD_ARTICULATION = EntityArticulationInfoCfg(
  actuators=KUAVO_S54_ARTICULATION.actuators + (KUAVO_S54_HEAD_ACTUATOR,),
  soft_joint_pos_limit_factor=0.95,
)

# Ordered like the controlled joint/action vector: left leg, right leg, waist,
# left arm, right arm.  Values remain derived from the actuator definitions
# above so the SSR reward normalization cannot drift from robot physics.
_LEG_ACTUATORS = KUAVO_S54_ARTICULATION.actuators[:6]
_ARM_ACTUATORS = KUAVO_S54_ARTICULATION.actuators[7:]
KUAVO_S54_CONTROLLED_JOINT_EFFORT_LIMITS = tuple(
  actuator.effort_limit
  for actuator in (
    *_LEG_ACTUATORS,
    *_LEG_ACTUATORS,
    KUAVO_S54_WAIST_ACTUATOR,
    *_ARM_ACTUATORS,
    *_ARM_ACTUATORS,
  )
)
assert all(limit is not None for limit in KUAVO_S54_CONTROLLED_JOINT_EFFORT_LIMITS)

# S54's source model does not declare joint velocity limits.  Keep the SSR
# normalization/safety threshold centralized with the other robot control
# constants until a per-joint hardware limit table is available.
KUAVO_S54_JOINT_VELOCITY_LIMIT = 20.0


def get_kuavo_s54_robot_cfg() -> EntityCfg:
  """Get a fresh Kuavo S54 robot configuration instance."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(),
    spec_fn=get_spec,
    articulation=KUAVO_S54_ARTICULATION,
  )


def get_kuavo_s54_head_robot_cfg() -> EntityCfg:
  """Get a fresh Kuavo S54 configuration with controllable head joints."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(),
    spec_fn=get_spec_with_head,
    articulation=KUAVO_S54_HEAD_ARTICULATION,
  )


KUAVO_S54_ACTION_SCALE: dict[str, float] = {}
for actuator in KUAVO_S54_ARTICULATION.actuators:
  assert isinstance(actuator, BuiltinPositionActuatorCfg)
  assert actuator.effort_limit is not None
  for name_expr in actuator.target_names_expr:
    KUAVO_S54_ACTION_SCALE[name_expr] = (
      0.25 * actuator.effort_limit / actuator.stiffness
    )

KUAVO_S54_HEAD_ACTION_SCALE = {
  **KUAVO_S54_ACTION_SCALE,
  r"zhead_[12]_joint": 0.25,
}


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_kuavo_s54_robot_cfg())
  viewer.launch(robot.spec.compile())
