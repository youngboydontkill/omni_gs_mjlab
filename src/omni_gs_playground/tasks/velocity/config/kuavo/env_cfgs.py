"""Kuavo velocity environment configurations."""

from copy import deepcopy

from omni_gs_playground.assets.robots.kuavo import (
  KUAVO_S45_ACTION_SCALE,
  KUAVO_S54_ACTION_SCALE,
  KUAVO_S54_HEAD_ACTION_SCALE,
  KUAVO_S54_HEAD_GEOM_GROUP,
  get_kuavo_s45_robot_cfg,
  get_kuavo_s54_head_robot_cfg,
  get_kuavo_s54_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.entity import EntityCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from omni_gs_playground.tasks.velocity import mdp
from omni_gs_playground.tasks.velocity.velocity_env_cfg import (
  make_kuavo_velocity_env_cfg as make_velocity_env_cfg,
)

ROOT_BODY = "base_link"
FOOT_BODIES = ("leg_l6_link", "leg_r6_link")
FOOT_SITES = ("l_ft_frame", "r_ft_frame")
S45_CONTROLLED_JOINTS = (
  r"leg_[lr][1-6]_joint",
  r"zarm_[lr][1-7]_joint",
)
S54_CONTROLLED_JOINTS = (
  r"leg_[lr][1-6]_joint",
  "waist_yaw_joint",
  r"zarm_[lr][1-7]_joint",
)
HEAD_CONTROLLED_JOINTS = S54_CONTROLLED_JOINTS + (r"zhead_[12]_joint",)


def _controlled_joints_cfg(
  joint_names: tuple[str, ...],
) -> SceneEntityCfg:
  return SceneEntityCfg(
    "robot",
    joint_names=joint_names,
    preserve_order=True,
  )


def _kuavo_rough_env_cfg(
  *,
  robot_cfg: EntityCfg,
  action_scale: dict[str, float],
  controlled_joints: tuple[str, ...],
  viewer_body: str,
  has_waist: bool,
  depth_camera_parent_body: str,
  depth_camera_pos: tuple[float, float, float],
  play: bool,
) -> ManagerBasedRlEnvCfg:
  """Create a Kuavo rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  # cfg.sim.nconmax = 48
  # 这里感觉非常影响显存阿
  cfg.sim.nconmax = 64
  # cfg.sim.nconmax = 128  # 每个 world 分配的 contact
  # cfg.sim.njmax = 2048  # 每个 world 分配的 constraint

  cfg.scene.entities = {"robot": robot_cfg}
  depth_camera = next(
    sensor for sensor in cfg.scene.sensors or () if sensor.name == "depth"
  )
  depth_camera.parent_body = depth_camera_parent_body
  depth_camera.pos = depth_camera_pos

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=FOOT_BODIES,
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern=ROOT_BODY, entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern=ROOT_BODY, entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = action_scale

  cfg.viewer.body_name = viewer_body

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  controlled_joints_terms = (
    cfg.observations["actor"].terms["joint_pos"],
    cfg.observations["actor"].terms["joint_vel"],
    cfg.observations["critic"].terms["joint_pos"],
    cfg.observations["critic"].terms["joint_vel"],
  )
  for term in controlled_joints_terms:
    term.params["asset_cfg"] = _controlled_joints_cfg(controlled_joints)

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = FOOT_SITES

  cfg.events["reset_robot_joints"].params["asset_cfg"] = _controlled_joints_cfg(
    controlled_joints
  )
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = ".*"
  cfg.events["base_com"].params["asset_cfg"].body_names = (ROOT_BODY,)

  cfg.rewards["pose"].params["asset_cfg"] = _controlled_joints_cfg(controlled_joints)
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    r"leg_[lr]1_joint": 0.15,  # Hip roll.
    r"leg_[lr]2_joint": 0.15,  # Hip yaw.
    r"leg_[lr]3_joint": 0.5,  # Hip pitch.
    r"leg_[lr]4_joint": 0.5,  # Knee.
    r"leg_[lr]5_joint": 0.15,  # Ankle pitch.
    r"leg_[lr]6_joint": 0.1,  # Ankle roll.
    r"zarm_[lr]1_joint": 0.15,  # Shoulder pitch.
    r"zarm_[lr][2-4]_joint": 0.1,  # Shoulder roll/yaw and elbow.
    r"zarm_[lr][5-7]_joint": 0.1,  # Wrist.
  }
  if has_waist:
    cfg.rewards["pose"].params["std_walking"]["waist_yaw_joint"] = 0.15
  cfg.rewards["pose"].params["std_running"] = {
    r"leg_[lr]1_joint": 0.25,
    r"leg_[lr]2_joint": 0.25,
    r"leg_[lr]3_joint": 0.5,
    r"leg_[lr]4_joint": 0.5,
    r"leg_[lr]5_joint": 0.25,
    r"leg_[lr]6_joint": 0.1,
    r"zarm_[lr]1_joint": 0.25,
    r"zarm_[lr][2-4]_joint": 0.1,
    r"zarm_[lr][5-7]_joint": 0.1,
  }
  if has_waist:
    cfg.rewards["pose"].params["std_running"]["waist_yaw_joint"] = 0.25

  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = (ROOT_BODY,)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (ROOT_BODY,)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = FOOT_SITES
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = FOOT_SITES
  cfg.rewards["stand_still"].params["asset_cfg"] = _controlled_joints_cfg(
    controlled_joints
  )
  cfg.rewards["joint_acc_l2"].params["asset_cfg"] = _controlled_joints_cfg(
    controlled_joints
  )
  cfg.rewards["joint_pos_limits"].params["asset_cfg"] = _controlled_joints_cfg(
    controlled_joints
  )
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def kuavo_s45_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Kuavo S45 rough terrain velocity configuration."""
  return _kuavo_rough_env_cfg(
    robot_cfg=get_kuavo_s45_robot_cfg(),
    action_scale=KUAVO_S45_ACTION_SCALE,
    controlled_joints=S45_CONTROLLED_JOINTS,
    viewer_body=ROOT_BODY,
    has_waist=False,
    depth_camera_parent_body="robot/base_link",
    depth_camera_pos=(0.0787, 0.0, 0.082951),
    play=play,
  )


def _kuavo_s54_base_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the shared Kuavo S54 rough terrain configuration."""
  return _kuavo_rough_env_cfg(
    robot_cfg=get_kuavo_s54_robot_cfg(),
    action_scale=KUAVO_S54_ACTION_SCALE,
    controlled_joints=S54_CONTROLLED_JOINTS,
    viewer_body="waist_yaw",
    has_waist=True,
    depth_camera_parent_body="robot/waist_yaw",
    depth_camera_pos=(0.0987, 0.0, -0.028449),
    play=play,
  )


def _separate_depth_observations(
  cfg: ManagerBasedRlEnvCfg,
  *,
  normalize: bool,
) -> None:
  """Move depth terms into independent 2D observation groups."""
  actor_depth = cfg.observations["actor"].terms.pop("depth")
  critic_depth = cfg.observations["critic"].terms.pop("depth")
  for depth_term in (actor_depth, critic_depth):
    depth_term.params["flatten"] = False
    depth_term.params["normalize"] = normalize

  cfg.observations["actor"].history_length = 5
  cfg.observations["actor"].flatten_history_dim = True
  cfg.observations["critic"].history_length = 1
  cfg.observations["actor_depth"] = ObservationGroupCfg(
    terms={"depth": deepcopy(actor_depth)},
    concatenate_terms=True,
    enable_corruption=cfg.observations["actor"].enable_corruption,
    history_length=None,
  )
  cfg.observations["critic_depth"] = ObservationGroupCfg(
    terms={"depth": deepcopy(critic_depth)},
    concatenate_terms=True,
    enable_corruption=False,
    history_length=None,
  )


def kuavo_s54_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the 27-joint S54 rough terrain task with waist-mounted DeFM depth."""
  cfg = _kuavo_s54_base_rough_env_cfg(play=play)
  depth_camera = next(
    sensor for sensor in cfg.scene.sensors or () if sensor.name == "depth"
  )
  depth_camera.width = 42
  depth_camera.height = 42
  _separate_depth_observations(cfg, normalize=False)
  return cfg


def kuavo_s54_head_cnn_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the rough-terrain S54 task with controllable head and CNN observations."""
  cfg = _kuavo_s54_base_rough_env_cfg(play=play)
  cfg.scene.entities = {"robot": get_kuavo_s54_head_robot_cfg()}

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = KUAVO_S54_HEAD_ACTION_SCALE

  controlled_joints_cfg = _controlled_joints_cfg(HEAD_CONTROLLED_JOINTS)
  controlled_joints_terms = (
    cfg.observations["actor"].terms["joint_pos"],
    cfg.observations["actor"].terms["joint_vel"],
    cfg.observations["critic"].terms["joint_pos"],
    cfg.observations["critic"].terms["joint_vel"],
  )
  for term in controlled_joints_terms:
    term.params["asset_cfg"] = controlled_joints_cfg

  cfg.events["reset_robot_joints"].params["asset_cfg"] = controlled_joints_cfg
  cfg.rewards["pose"].params["asset_cfg"] = controlled_joints_cfg
  cfg.rewards["stand_still"].params["asset_cfg"] = controlled_joints_cfg
  cfg.rewards["joint_acc_l2"].params["asset_cfg"] = controlled_joints_cfg
  cfg.rewards["joint_pos_limits"].params["asset_cfg"] = controlled_joints_cfg
  cfg.rewards["pose"].params["std_standing"] = {
    **{joint_names: 0.05 for joint_names in S54_CONTROLLED_JOINTS},
    r"zhead_[12]_joint": 0.15,
  }
  for pose_std_name in ("std_walking", "std_running"):
    cfg.rewards["pose"].params[pose_std_name][r"zhead_[12]_joint"] = 0.15

  depth_camera = next(
    sensor for sensor in cfg.scene.sensors or () if sensor.name == "depth"
  )
  depth_camera.parent_body = "robot/zhead_2_link"
  depth_camera.pos = (0.133839, 0.0475, 0.050077)
  depth_camera.quat = (0.577506, 0.408028, -0.408028, -0.577506)
  # depth_camera.pos = (0.093839, 0.0475, 0.050077)
  # depth_camera.enabled_geom_groups = tuple(
  #   group for group in range(6) if group != KUAVO_S54_HEAD_GEOM_GROUP
  # )

  _separate_depth_observations(cfg, normalize=True)

  return cfg


def kuavo_s54_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Kuavo S54 flat terrain velocity configuration."""
  cfg = _kuavo_s54_base_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (0.0, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg


def kuavo_s45_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Kuavo S45 flat terrain velocity configuration."""
  cfg = kuavo_s45_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (0.0, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg
