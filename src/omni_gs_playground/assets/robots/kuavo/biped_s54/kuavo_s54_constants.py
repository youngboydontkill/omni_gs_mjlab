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
  return spec


def get_spec_with_head() -> mujoco.MjSpec:
  """Load the S54 spec with movable head links and position-controlled joints."""
  spec = mujoco.MjSpec.from_file(str(KUAVO_S54_XML))
  for actuator in tuple(spec.actuators):
    spec.delete(actuator)
  # for body_name in ("zhead_1_link", "zhead_2_link"):
  #   for geom in spec.body(body_name).geoms:
  #     geom.group = KUAVO_S54_HEAD_GEOM_GROUP
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
  stiffness=60.0,
  damping=6.0,
  effort_limit=101.6,
  armature=0.05,
)
KUAVO_S54_LEG_2_ACTUATOR = _position_actuator(
  ("leg_[lr]2_joint",),
  stiffness=60.0,
  damping=6.0,
  effort_limit=56.8,
  armature=0.025,
)
KUAVO_S54_LEG_3_ACTUATOR = _position_actuator(
  ("leg_[lr]3_joint",),
  stiffness=80.0,
  damping=6.0,
  effort_limit=105.6,
  armature=0.025,
)
KUAVO_S54_LEG_4_ACTUATOR = _position_actuator(
  ("leg_[lr]4_joint",),
  stiffness=95.0,
  damping=6.0,
  effort_limit=224.0,
  armature=0.05,
)
KUAVO_S54_LEG_5_ACTUATOR = _position_actuator(
  ("leg_[lr]5_joint",),
  stiffness=55.0,
  damping=7.5,
  effort_limit=91.6,
  armature=0.05,
)
KUAVO_S54_LEG_6_ACTUATOR = _position_actuator(
  ("leg_[lr]6_joint",),
  stiffness=55.0,
  damping=7.5,
  effort_limit=57.0,
  armature=0.05,
)
KUAVO_S54_WAIST_ACTUATOR = _position_actuator(
  ("waist_yaw_joint",),
  stiffness=40.0,
  damping=4.0,
  effort_limit=81.6,
  armature=0.025,
)
KUAVO_S54_ARM_1_ACTUATOR = _position_actuator(
  ("zarm_[lr]1_joint",),
  stiffness=20.0,
  damping=3.0,
  effort_limit=52.8,
  armature=0.025,
)
KUAVO_S54_ARM_2_ACTUATOR = _position_actuator(
  ("zarm_[lr]2_joint",),
  stiffness=20.0,
  damping=3.0,
  effort_limit=60.0,
  armature=0.02,
)
KUAVO_S54_ARM_3_ACTUATOR = _position_actuator(
  ("zarm_[lr]3_joint",),
  stiffness=20.0,
  damping=3.0,
  effort_limit=45.6,
  armature=0.02,
)
KUAVO_S54_ARM_4_ACTUATOR = _position_actuator(
  ("zarm_[lr]4_joint",),
  stiffness=20.0,
  damping=3.0,
  effort_limit=60.0,
  armature=0.02,
)
KUAVO_S54_ARM_5_ACTUATOR = _position_actuator(
  ("zarm_[lr]5_joint",),
  stiffness=15.0,
  damping=3.0,
  effort_limit=11.0,
  armature=0.01,
)
KUAVO_S54_ARM_6_ACTUATOR = _position_actuator(
  ("zarm_[lr]6_joint",),
  stiffness=15.0,
  damping=3.0,
  effort_limit=11.0,
  armature=0.01,
)
KUAVO_S54_ARM_7_ACTUATOR = _position_actuator(
  ("zarm_[lr]7_joint",),
  stiffness=15.0,
  damping=3.0,
  effort_limit=11.0,
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
  pos=(0.0, 0.0, 0.925),
  rot=(1.0, 0.0, 0.0, 0.0),
  joint_pos={
    "leg_[lr]1_joint": 0.0,
    "leg_[lr]2_joint": 0.0,
    "leg_[lr]3_joint": -0.4,
    "leg_[lr]4_joint": 0.69,
    "leg_[lr]5_joint": -0.33,
    "leg_[lr]6_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "zarm_.*_joint": 0.0,
    "zhead_.*_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)


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
