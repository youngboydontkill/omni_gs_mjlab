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


# Teacher policy actuator parameters (from kuavo_s45_distill_ppo_runner_cfg).
# Matches the physical parameters used to train doc/model_48350.pt:
#   (name_expr, stiffness, damping, effort_limit, armature, frictionloss)
# velocity_limit, friction_static, activation_vel, friction_dynamic are not
# directly supported by BuiltinPositionActuatorCfg / MuJoCo position actuator.
_ACTUATOR_PARAMS = (
  ("leg_[lr]1_joint", 100.0, 4.0, 180.0, 0.05),
  ("leg_[lr]2_joint", 100.0, 4.0, 100.0, 0.025),
  ("leg_[lr]3_joint", 100.0, 4.0, 100.0, 0.025 ),
  ("leg_[lr]4_joint", 150.0, 8.0, 180.0, 0.05 ),
  ("leg_[lr]5_joint", 40.0, 4.0, 72.0, 0.05),
  ("leg_[lr]6_joint", 40.0, 4.0, 36.0, 0.05),
  ("zarm_[lr]1_joint", 30.0, 3.0, 100.0, 0.025),
  ("zarm_[lr]2_joint", 30.0, 3.0, 50.0, 0.02),
  ("zarm_[lr]3_joint", 30.0, 3.0, 36.0, 0.02),
  ("zarm_[lr]4_joint", 20.0, 3.0, 50.0, 0.02),
  ("zarm_[lr]5_joint", 10.0, 3.0, 12.0, 0.01),
  ("zarm_[lr]6_joint", 10.0, 3.0, 12.0, 0.01),
  ("zarm_[lr]7_joint", 10.0, 3.0, 12.0, 0.01),
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
    "leg_[lr]3_joint": -0.27,
    "leg_[lr]4_joint": 0.52,
    "leg_[lr]5_joint": -0.3,
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
    KUAVO_S45_ACTION_SCALE[name_expr] = 0.25


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_kuavo_s45_robot_cfg())
  viewer.launch(robot.spec.compile())
