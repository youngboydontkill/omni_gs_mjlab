"""Kuavo S45 robot constants."""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

KUAVO_S45_XML: Path = Path(__file__).parent / "xml" / "biped_s45_collision.xml"
assert KUAVO_S45_XML.exists()


def get_spec() -> mujoco.MjSpec:
  """Load the S45 collision spec with fixed head links."""
  spec = mujoco.MjSpec.from_file(str(KUAVO_S45_XML))
  for actuator in tuple(spec.actuators):
    spec.delete(actuator)
  for sensor in tuple(spec.sensors):
    if sensor.name.startswith(("zhead_1_joint_", "zhead_2_joint_")):
      spec.delete(sensor)
  spec.delete(spec.joint("zhead_1_joint"))
  spec.delete(spec.joint("zhead_2_joint"))
  return spec


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


# S45 uses the S54 position-control gains and its own XML effort/armature values.
_ACTUATOR_PARAMS = (
  ("leg_[lr]1_joint", 60.0, 6.0, 180.0, 0.12),
  ("leg_[lr]2_joint", 60.0, 6.0, 100.0, 0.0508),
  ("leg_[lr]3_joint", 80.0, 6.0, 100.0, 0.0508),
  ("leg_[lr]4_joint", 95.0, 6.0, 180.0, 0.12),
  ("leg_[lr]5_joint", 55.0, 7.5, 36.0, 0.05),
  ("leg_[lr]6_joint", 55.0, 7.5, 36.0, 0.05),
  ("zarm_[lr]1_joint", 20.0, 3.0, 100.0, 0.05),
  ("zarm_[lr]2_joint", 20.0, 3.0, 50.0, 0.05),
  ("zarm_[lr]3_joint", 20.0, 3.0, 39.0, 0.05),
  ("zarm_[lr]4_joint", 20.0, 3.0, 50.0, 0.05),
  ("zarm_[lr]5_joint", 15.0, 3.0, 12.0, 0.05),
  ("zarm_[lr]6_joint", 15.0, 3.0, 12.0, 0.05),
  ("zarm_[lr]7_joint", 15.0, 3.0, 12.0, 0.05),
)

KUAVO_S45_ACTUATORS = tuple(
  _position_actuator(
    (name_expr,),
    stiffness=stiffness,
    damping=damping,
    effort_limit=effort_limit,
    armature=armature,
  )
  for name_expr, stiffness, damping, effort_limit, armature in _ACTUATOR_PARAMS
)

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.84),
  rot=(1.0, 0.0, 0.0, 0.0),
  joint_pos={
    "leg_[lr]1_joint": 0.0,
    "leg_[lr]2_joint": 0.0,
    "leg_[lr]3_joint": -0.4,
    "leg_[lr]4_joint": 0.69,
    "leg_[lr]5_joint": -0.33,
    "leg_[lr]6_joint": 0.0,
    "zarm_.*_joint": 0.0,
    "zhead_.*_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

KUAVO_S45_ARTICULATION = EntityArticulationInfoCfg(
  actuators=KUAVO_S45_ACTUATORS,
  soft_joint_pos_limit_factor=0.95,
)


def get_kuavo_s45_robot_cfg() -> EntityCfg:
  """Get a fresh Kuavo S45 robot configuration instance."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(),
    spec_fn=get_spec,
    articulation=KUAVO_S45_ARTICULATION,
  )


KUAVO_S45_ACTION_SCALE: dict[str, float] = {}
for actuator in KUAVO_S45_ARTICULATION.actuators:
  assert isinstance(actuator, BuiltinPositionActuatorCfg)
  assert actuator.effort_limit is not None
  for name_expr in actuator.target_names_expr:
    KUAVO_S45_ACTION_SCALE[name_expr] = (
      0.25 * actuator.effort_limit / actuator.stiffness
    )


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_kuavo_s45_robot_cfg())
  viewer.launch(robot.spec.compile())
