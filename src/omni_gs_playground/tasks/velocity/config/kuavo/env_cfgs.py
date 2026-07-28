"""Kuavo velocity environment configurations."""

import math
from copy import deepcopy

from omni_gs_playground.assets.robots.kuavo import (
  KUAVO_S45_ACTION_SCALE,
  KUAVO_S54_ACTION_SCALE,
  KUAVO_S54_CONTROLLED_JOINT_EFFORT_LIMITS,
  KUAVO_S54_DEFAULT_BASE_HEIGHT,
  KUAVO_S54_HEAD_ACTION_SCALE,
  KUAVO_S54_HEAD_GEOM_GROUP,
  KUAVO_S54_JOINT_VELOCITY_LIMIT,
  KUAVO_S54_SOLE_SCAN_RESOLUTION,
  KUAVO_S54_SOLE_SCAN_SITE_NAMES,
  KUAVO_S54_SOLE_SCAN_SIZE,
  KUAVO_S54_TOE_REACH,
  get_kuavo_s45_robot_cfg,
  get_kuavo_s54_head_robot_cfg,
  get_kuavo_s54_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr as mjlab_dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.entity import EntityCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  GridPatternCfg,
  ObjRef,
  RayCastSensorCfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from omni_gs_playground.tasks.velocity import mdp
from omni_gs_playground.tasks.velocity.velocity_env_cfg import make_kuavo_velocity_env_cfg

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
  depth_camera_quat: tuple[float, float, float, float] | None = None,
) -> ManagerBasedRlEnvCfg:
  """Create a Kuavo rough terrain velocity configuration.

  ``depth_camera_quat`` is optional: when ``None`` the depth sensor keeps the
  shared S54-derived orientation from :func:`make_kuavo_velocity_env_cfg`.
  Pass a ``(w, x, y, z)`` quaternion to override it for robots whose real-world
  camera mount uses a different pitch (e.g. S45 waist camera at 0.698 rad
  vs. the S54 head camera at ~0.593 rad encoded in the default quat).
  """
  cfg = make_kuavo_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 1000
  # nconmax bumped from 64 to 128: rough terrain + new undesired_body_contact
  # sensor + EMP rewards push contact-count peaks past 64 on certain
  # sub-terrains, which silently fails the MuJoCo solver and propagates
  # NaN/Inf into qpos/qvel (and therefore into actor + critic observations).
  cfg.sim.nconmax = 128
  # cfg.sim.nconmax = 128  # 每个 world 分配的 contact
  # cfg.sim.njmax = 2048  # 每个 world 分配的 constraint

  cfg.scene.entities = {"robot": robot_cfg}
  depth_camera = next(
    sensor for sensor in cfg.scene.sensors or () if sensor.name == "depth"
  )
  depth_camera.parent_body = depth_camera_parent_body
  depth_camera.pos = depth_camera_pos
  if depth_camera_quat is not None:
    depth_camera.quat = depth_camera_quat

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
    cfg.events.pop("depth_camera_pitch", None)
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

  # Depth camera extrinsic domain randomization: resample pitch per episode.
  cfg.events["depth_camera_pitch"] = EventTermCfg(
    func=mjlab_dr.cam_quat,
    mode="reset",
    params={
      "roll_range": (0.0, 0.0),
      "pitch_range": (-0.15, 0.15),  # ±~8.6°
      "yaw_range": (0.0, 0.0),
      "asset_cfg": SceneEntityCfg("robot", camera_names=("depth",)),
    },
  )

  return cfg


def kuavo_s45_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Kuavo S45 rough terrain velocity configuration.

  Reward set is ported from Leju-IsaacLab ``emp_env_cfg.py``: starts from the
  shared Kuavo rough config and then overrides weights / drops a few default
  terms / injects EMP-specific terms. Only this task is affected; the S54
  variants reuse ``_kuavo_rough_env_cfg`` unchanged.

  The depth camera observation is hoisted into independent
  ``actor_depth`` / ``critic_depth`` groups (normalized to [-1, 1]) so the
  CNN encoder configured in :func:`kuavo_s45_ppo_runner_cfg` consumes it as
  a 2D input. This matches the layout used by ``Kuavo-S54-Head-CNN-Rough``.
  """
  cfg = _kuavo_rough_env_cfg(
    robot_cfg=get_kuavo_s45_robot_cfg(),
    action_scale=KUAVO_S45_ACTION_SCALE,
    controlled_joints=S45_CONTROLLED_JOINTS,
    viewer_body=ROOT_BODY,
    has_waist=False,
    # Waist camera pose from
    # gauss-gym-Leju/resources/robots/biped_s45/urdf/biped_s45_waist_cam.urdf:
    #   <joint name="waist_camera" type="fixed">
    #     <origin xyz="0.168717483101422 0 0.01355599662743" rpy="0 0.698 0"/>
    #     <parent link="base_link"/>
    #   </joint>
    # The quat below = R_y(0.698) composed with the MuJoCo camera optical
    # rotation (camera looks down its local -Z, with +X right, +Y up); the
    # default S54 quat encodes a 0.593 rad pitch and would tilt the S45
    # waist cam 6° too shallow if reused.
    depth_camera_parent_body="robot/base_link",
    depth_camera_pos=(0.168717483101422, 0.0, 0.01355599662743),
    depth_camera_quat=(0.6830, 0.1830, -0.1830, -0.6830),  # 70°: (0.6964, 0.1228, -0.1228, -0.6964) 40°: 0.6408367, 0.29887844, -0.29887844, -0.6408367 60°:0.6830, 0.1830, -0.1830, -0.6830
    play=play,
  )
  _apply_s45_emp_rewards(cfg)
  _separate_depth_observations(cfg, normalize=True)
  return cfg

def kuavo_s45_simple_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = _kuavo_rough_env_cfg(play=play)
  assert cfg.scene.terrain is not None
  gen = cfg.scene.terrain.terrain_generator
  assert gen is not None
  import mjlab.terrains as terrain_gen
  gen.sub_terrains = {
    "stairs_up": terrain_gen.BoxPyramidStairsTerrainCfg(
      proportion=0.5,
      step_height_range=(0.02, 0.14),
      step_width=0.32,
      platform_width=2.0,
      border_width=0.8,
    ),
    "stairs_down": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
      proportion=0.5,
      step_height_range=(0.02, 0.14),
      step_width=0.32,
      platform_width=2.0,
      border_width=0.8,
    ),
  }
  gen.curriculum = True
  return cfg


def kuavo_s45_stairs_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Kuavo S45 stairs-only velocity configuration.

  Restricts the terrain generator to just stair terrain types (ascending and
  descending), sharing all S45 EMP rewards / depth observation / simulation
  parameters with the full rough config.  Used for training a stairs-specialist
  teacher policy.
  """
  cfg = kuavo_s45_rough_env_cfg(play=play)
  assert cfg.scene.terrain is not None
  gen = cfg.scene.terrain.terrain_generator
  assert gen is not None
  import mjlab.terrains as terrain_gen
  gen.sub_terrains = {
    "stairs_up": terrain_gen.BoxPyramidStairsTerrainCfg(
      proportion=0.5,
      step_height_range=(0.02, 0.14),
      step_width=0.32,
      platform_width=2.0,
      border_width=0.8,
    ),
    "stairs_down": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
      proportion=0.5,
      step_height_range=(0.02, 0.14),
      step_width=0.32,
      platform_width=2.0,
      border_width=0.8,
    ),
  }
  gen.curriculum = True
  return cfg


def kuavo_s45_slope_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Kuavo S45 slope-only velocity configuration.

  Restricts the terrain generator to just slope terrain types (up and down),
  sharing all S45 EMP rewards / depth observation / simulation parameters with
  the full rough config.  Used for training a slope-specialist teacher policy.
  """
  cfg = kuavo_s45_rough_env_cfg(play=play)
  assert cfg.scene.terrain is not None
  gen = cfg.scene.terrain.terrain_generator
  assert gen is not None
  import mjlab.terrains as terrain_gen
  gen.sub_terrains = {
    "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
      proportion=0.5,
      slope_range=(0.0, 0.35),
      platform_width=2.0,
      border_width=0.25,
      inverted=False,
    ),
    "slope_down": terrain_gen.HfPyramidSlopedTerrainCfg(
      proportion=0.5,
      slope_range=(0.0, 0.35),
      platform_width=2.0,
      border_width=0.25,
      inverted=True,
    ),
  }
  gen.curriculum = True
  return cfg


# ---------------------------------------------------------------------------
# S45 EMP-style reward port (Leju-IsaacLab emp_env_cfg.py).
# ---------------------------------------------------------------------------

_S45_FEET_GROUND_SENSOR = "feet_ground_contact"
_S45_UNDESIRED_CONTACT_SENSOR = "undesired_body_contact"
_S45_FEET_NAMES: tuple[str, ...] = FOOT_BODIES
_S45_ANKLE_JOINTS: tuple[str, ...] = (
  "leg_l5_joint",
  "leg_l6_joint",
  "leg_r5_joint",
  "leg_r6_joint",
)


def _scene_cfg(**kwargs) -> SceneEntityCfg:
  return SceneEntityCfg("robot", **kwargs)


def _apply_s45_emp_rewards(
  cfg: ManagerBasedRlEnvCfg,
  controlled_joints: tuple[str, ...] = S45_CONTROLLED_JOINTS,
) -> None:
  """In-place: rewrite ``cfg.rewards`` to match Leju-IsaacLab S42 EMP setup."""

  # 1) New sensor: any contact between non-foot bodies and the world.
  undesired_body_contact_cfg = ContactSensorCfg(
    name=_S45_UNDESIRED_CONTACT_SENSOR,
    primary=ContactMatch(
      mode="body",
      pattern=(r"leg_[lr][1-5]_link", "base_link", r"zarm_[lr][1-7]_link"),
      entity="robot",
    ),
    secondary=None,
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (undesired_body_contact_cfg,)

  # Forward-pointing single-ray raycasters on each toe. Attached to the foot
  # body (leg_[lr]6_link) and aimed along +X in the base frame; used by the
  # `toe_touch` reward to penalize a toe approaching / contacting a vertical
  # face (stair riser, wall). MJLab's RayCastSensorCfg has no origin offset
  # (unlike IsaacLab's OffsetCfg): the ray starts at the foot body frame (ankle
  # joint), a few cm above the sole; the forward ray still detects vertical
  # faces ahead and the offset is symmetric across both feet.
  _toe_scanner_kwargs = dict(
    ray_alignment="base",
    pattern=GridPatternCfg(size=(0.0, 0.0), resolution=0.01, direction=(1.0, 0.0, 0.0)),
    max_distance=1.0,
    exclude_parent_body=True,
    debug_vis=True,
  )
  feet_l_forward_scanner = RayCastSensorCfg(
    name="feet_l_forward_scanner",
    frame=ObjRef(type="body", name=_S45_FEET_NAMES[0], entity="robot"),
    **_toe_scanner_kwargs,
  )
  feet_r_forward_scanner = RayCastSensorCfg(
    name="feet_r_forward_scanner",
    frame=ObjRef(type="body", name=_S45_FEET_NAMES[1], entity="robot"),
    **_toe_scanner_kwargs,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_l_forward_scanner,
    feet_r_forward_scanner,
  )

  # Downward raycast clusters under each foot for the edge-contact penalty
  # (Hiking in the Wild §III-C, raycast-normal approximation — see
  # mdp.edge_contact_penalty). Cover most of the sole instead of only its
  # center so a heel/toe overhang cannot escape the edge detector.
  _edge_scanner_kwargs = dict(
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(0.16, 0.08), resolution=0.02),  # 9x5, downward
    max_distance=1.0,
    exclude_parent_body=True,
    debug_vis=False,
  )
  feet_l_edge_scanner = RayCastSensorCfg(
    name="feet_l_edge_scanner",
    frame=ObjRef(type="body", name=_S45_FEET_NAMES[0], entity="robot"),
    **_edge_scanner_kwargs,
  )
  feet_r_edge_scanner = RayCastSensorCfg(
    name="feet_r_edge_scanner",
    frame=ObjRef(type="body", name=_S45_FEET_NAMES[1], entity="robot"),
    **_edge_scanner_kwargs,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_l_edge_scanner,
    feet_r_edge_scanner,
  )

  # Verify air-time tracking is on for the feet sensor.
  feet_ground = next(
    s for s in cfg.scene.sensors or () if s.name == _S45_FEET_GROUND_SENSOR
  )
  assert feet_ground.track_air_time, (
    "feet_ground_contact must have track_air_time=True for "
    "feet_air_time_positive_biped"
  )

  # 2) Drop terms the source replaces / does not use.
  for key in ("foot_clearance", "foot_slip", "soft_landing"):
    cfg.rewards.pop(key, None)

  # 3) Re-weight surviving terms.
  # P1 (see doc/terrain_curriculum_stuck.md §3): strengthen the forward
  # signal to pull the policy out of the "stand still" local optimum.
  cfg.rewards["track_linear_velocity"].weight = 8.0
  cfg.rewards["track_angular_velocity"].weight = 3.0
  cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.25)
  cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.25)
  cfg.rewards["action_rate_l2"].weight = -0.01  # -0.005 -> -0.01(踝 roll 扭动)
  cfg.rewards["body_orientation_l2"].weight = -3.0
  # body_ang_vel (-0.05), joint_acc_l2 (-2.5e-7), joint_pos_limits (-10.0),
  # is_terminated (-200.0), stand_still (-1.0), self_collisions (-1.0) are
  # kept as-is (already match source intent).

  # 4) Inject EMP-specific terms.
  non_ankle_torque_joints = (r"leg_[lr][1-5]_joint", r"zarm_[lr][1-7]_joint")
  ankle_torque_joints = (r"leg_[lr]6_joint",)
  hip_joints = (r"leg_[lr][12]_joint",)
  arm_joints = (r"zarm_[lr][1-7]_joint",)
  all_controlled = _controlled_joints_cfg(controlled_joints)

  cfg.rewards.update({
    "dof_vel_l2": RewardTermCfg(
      func=mdp.joint_vel_l2,
      weight=-2.0e-3,
      params={"asset_cfg": all_controlled},
    ),
    "dof_torques_l2": RewardTermCfg(
      func=mdp.joint_torques_l2,
      weight=-1.0e-5,
      params={
        "asset_cfg": _scene_cfg(actuator_names=non_ankle_torque_joints),
      },
    ),
    "dof_torques_ankle_l2": RewardTermCfg(
      func=mdp.joint_torques_l2,
      weight=-1.0e-4,  # -1e-5 -> -1e-4: prevent ankle-roll torque from being free.
      params={
        "asset_cfg": _scene_cfg(actuator_names=ankle_torque_joints),
      },
    ),
    # "ankle_roll_vel_l2": RewardTermCfg(
    #   # 直接惩罚踝 roll 角速度(扭动的最直接代理量)。action_rate_l2 作用于全维
    #   # raw action、无 asset_cfg,无法 scoped 到踝,故用 joint_vel_l2 + leg_6 专门
    #   # 约束 roll 角速度。ankle_torque_joints = (r"leg_[lr]6_joint",) 即踝 roll。
    #   func=mdp.joint_vel_l2,
    #   weight=-2.0e-2,
    #   params={
    #     "asset_cfg": _scene_cfg(
    #       joint_names=ankle_torque_joints, preserve_order=True
    #     ),
    #   },
    # ),
    "dof_power_l2": RewardTermCfg(
      func=mdp.joint_power_l2,
      weight=-2.0e-5,
      params={
        "asset_cfg": _scene_cfg(
          joint_names=controlled_joints,
          actuator_names=controlled_joints,
          preserve_order=True,
        ),
      },
    ),
    "action_smoothness_l2": RewardTermCfg(
      func=mdp.action_acc_l2,
      weight=-2.0e-2,  # -0.01 -> -0.02 压高频动作抖动(踝 roll 扭动)
    ),
    "feet_air_time": RewardTermCfg(
      func=mdp.feet_air_time_positive_biped,
      weight=4.0,
      params={
        "command_name": "twist",
        "threshold": 0.5,
        "sensor_name": _S45_FEET_GROUND_SENSOR,
        "command_threshold": 0.01,
        "asset_cfg": _scene_cfg(),
      },
    ),
    "feet_slide": RewardTermCfg(
      func=mdp.feet_slide,
      weight=-0.1,
      params={
        "sensor_name": _S45_FEET_GROUND_SENSOR,
        "asset_cfg": _scene_cfg(body_names=_S45_FEET_NAMES),
      },
    ),
    "feet_contact_without_cmd": RewardTermCfg(
      func=mdp.feet_contact_without_cmd,
      weight=0.4,
      params={
        "command_name": "twist",
        "sensor_name": _S45_FEET_GROUND_SENSOR,
        "command_threshold": 0.01,
        "asset_cfg": _scene_cfg(),
      },
    ),
    # Arms are 14/26 DoF; at 3.0 this term alone paid ~1.4/step without
    # requiring any walking, anchoring the static "stand still" optimum.
    # Keep it as a posture prior, not the dominant task signal.
    "track_default_arm_pos": RewardTermCfg(
      func=mdp.track_default_arm_pos,
      weight=0.0,
      params={
        "asset_cfg": _scene_cfg(joint_names=arm_joints, preserve_order=True),
        "alpha": 5.0,
      },
    ),
    "joint_deviation_hip": RewardTermCfg(
      func=mdp.joint_deviation_l1,
      weight=-0.1,
      params={
        "asset_cfg": _scene_cfg(joint_names=hip_joints, preserve_order=True),
      },
    ),
    # P1: halved alongside track_default_arm_pos so the arm-posture bonus and
    # its regularizer stay in proportion after the rebalance.
    "joint_deviation_arms": RewardTermCfg(
      func=mdp.joint_deviation_l1,
      weight=-0.0,
      params={
        "asset_cfg": _scene_cfg(joint_names=arm_joints, preserve_order=True),
      },
    ),
    "contact_force": RewardTermCfg(
      func=mdp.contact_force_violation,
      weight=-0.001,
      params={
        "sensor_name": _S45_FEET_GROUND_SENSOR,
        "threshold": 900.0,
        "violation_max": 300.0,
      },
    ),
    "feet_stumble": RewardTermCfg(
      func=mdp.feet_stumble,
      weight=-1.0,
      params={"sensor_name": _S45_FEET_GROUND_SENSOR},
    ),
    "no_feet_contact": RewardTermCfg(
      func=mdp.no_feet_contact,
      weight=-0.1,
      params={
        "command_name": "twist",
        "sensor_name": _S45_FEET_GROUND_SENSOR,
        "command_threshold": 0.2,
        "force_threshold": 5.0,
      },
    ),
    "illegal_dof_barrier": RewardTermCfg(
      func=mdp.illegal_dof_pos_barrier,
      weight=-0.1,
      params={
        "asset_cfg": _scene_cfg(
          joint_names=_S45_ANKLE_JOINTS, preserve_order=True
        ),
      },
    ),
    # P1: -5.0 -> -1.0. At -5.0 this was a steep trap the std=1.3 noisy policy
    # could not avoid, so it suppressed every gait that risks narrow foot spacing.
    "feet_too_near": RewardTermCfg(
      func=mdp.feet_too_near_humanoid,
      weight=-1.0,
      params={
        "asset_cfg": _scene_cfg(),
        "threshold": 0.15,
        "feet_names": _S45_FEET_NAMES,
      },
    ),
    # P1: -10.0 -> -2.0. Soften the steepest gait trap so the policy can
    # explore airborne phases without a catastrophic penalty; the P0 std drop
    # already removes most of the jitter that triggered this term.
    "fly": RewardTermCfg(
      func=mdp.fly,
      weight=-2.0,
      params={
        "sensor_name": _S45_FEET_GROUND_SENSOR,
        "threshold": 1.0,
      },
    ),
    "undesired_contacts": RewardTermCfg(
      func=mdp.undesired_contacts,
      weight=-1.0,
      params={
        "sensor_name": _S45_UNDESIRED_CONTACT_SENSOR,
        "threshold": 1.0,
      },
    ),
    # Penalize a toe approaching / contacting a vertical face (stair riser,
    # wall) via the per-foot forward raycasters added above.
    "toe_touch": RewardTermCfg(
      func=mdp.toe_touch,
      weight=-5.0,
      params={
        "sensor_name_l": "feet_l_forward_scanner",
        "sensor_name_r": "feet_r_forward_scanner",
        "feet_length": 0.178,
        "margin": 0.01,
      },
    ),
    # Foothold safety: penalize loading a foot on a terrain edge, scaled by foot
    # speed (Hiking in the Wild §III-C, raycast-normal approximation via the
    # downward per-foot edge scanners). Complements toe_touch: toe_touch keeps
    # the toe off vertical faces, edge_contact keeps the sole centered on flat
    # ground. Conservative starting weight; not the paper's Warp point-mesh
    # penalty (mjlab terrain exposes no trimesh — see mdp.edge_contact_penalty).
    # "edge_contact": RewardTermCfg(
    #   func=mdp.edge_contact_penalty,
    #   weight=-0.5,
    #   params={
    #     "sensor_name_l": "feet_l_edge_scanner",
    #     "sensor_name_r": "feet_r_edge_scanner",
    #     "ground_sensor_name": _S45_FEET_GROUND_SENSOR,
    #     "asset_cfg": _scene_cfg(body_names=_S45_FEET_NAMES),
    #     "height_threshold": 0.03,
    #     "normal_threshold": 0.5,
    #   },
    # ),
  })


def _enable_s45_foothold_reward(cfg: ManagerBasedRlEnvCfg) -> None:
  """Add the terrain-edge signal used for distillation evaluation/fine-tuning.

  In pure BC this term is diagnostic only. The hybrid PPO fine-tuning task uses
  it as a low-weight complement to ``toe_touch``: toe rays detect stair risers,
  while the downward clusters detect a loaded sole straddling a terrain edge.
  """
  cfg.rewards["edge_contact"] = RewardTermCfg(
    func=mdp.edge_contact_penalty,
    weight=-2.0,
    params={
      "sensor_name_l": "feet_l_edge_scanner",
      "sensor_name_r": "feet_r_edge_scanner",
      "ground_sensor_name": _S45_FEET_GROUND_SENSOR,
      "asset_cfg": _scene_cfg(body_names=_S45_FEET_NAMES),
      "height_threshold": 0.03,
      "normal_threshold": 0.5,
    },
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
    depth_camera_pos=(0.0987, 0.0, -0.028449), # 0.09538, 0.0, -0.01491
    play=play,
  )

# 增加时间延迟
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

  # 论文「Hiking in the Wild」§III-B2 F_sim 深度退化链：仅对 actor 深度加噪，
  # critic 走干净米制深度（非对称 actor-critic）。play 分支已把 actor
  # enable_corruption 置 False（见 _kuavo_rough_env_cfg），此处据此关噪，
  # 保证回放拿干净深度。保守默认：小 range-gaussian σ + 低概率白区/blur/OOD。
  # actor_corrupt = bool(cfg.observations["actor"].enable_corruption)
  # actor_depth.params.update(
  #   corrupt=actor_corrupt,
  #   noise_std=0.02,             # 米制 2cm range-dependent 高斯噪声
  #   noise_range=(0.15, 3.0),    # 仅有效感知带内加噪
  #   white_prob=0.10,            # 10% 帧出现双目失配白区
  #   white_max_blocks=2,
  #   white_block_size=(8, 8),
  #   blur_prob=0.20,             # 20% 帧运动模糊
  #   blur_kernel=3,
  #   blur_sigma=0.8,
  #   ood_prob=0.005,             # 0.5% 帧整帧失效
  # )
  # critic_depth.params.update(corrupt=False)

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


def kuavo_s54_rough_cnn_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create S54 rough terrain CNN task with S45 EMP rewards and waist camera.

  The depth camera is mounted on the waist with a 59.6° downward pitch,
  matching the real ``waist_camera_link`` body in ``biped_s54.xml``
  (``quat=(0.868, 0, 0.497, 0)`` = R_y(59.6°)), composed with the MuJoCo
  camera optical rotation:

    q = 0.5·(cos(θ/2)+sin(θ/2), cos(θ/2)-sin(θ/2),
             -cos(θ/2)+sin(θ/2), -cos(θ/2)-sin(θ/2))
    θ = 59.6° → (0.682369, 0.185395, -0.185395, -0.682369)

  In play mode the ``depth_camera_pitch`` domain-randomisation event is
  removed so the extrinsics stay fixed at the design value.
  """
  cfg = _kuavo_s54_base_rough_env_cfg(play=play)
  depth_camera = next(
    sensor for sensor in cfg.scene.sensors or () if sensor.name == "depth"
  )
  depth_camera.quat = (0.682369, 0.185395, -0.185395, -0.682369)
  _apply_s45_emp_rewards(cfg, controlled_joints=S54_CONTROLLED_JOINTS)
  _separate_depth_observations(cfg, normalize=True)
  return cfg


def kuavo_s54_rough_ssr_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the isolated Kuavo-S54 port of SSR (without style rewards)."""
  cfg = _kuavo_s54_base_rough_env_cfg(play=play)
  controlled = _controlled_joints_cfg(S54_CONTROLLED_JOINTS)
  feet = SceneEntityCfg("robot", site_names=FOOT_SITES, preserve_order=True)
  sole_centers = SceneEntityCfg(
    "robot", site_names=KUAVO_S54_SOLE_SCAN_SITE_NAMES, preserve_order=True
  )

  depth_camera = next(
    sensor for sensor in cfg.scene.sensors or () if sensor.name == "depth"
  )
  depth_camera.width = 42
  depth_camera.height = 42
  depth_camera.quat = (0.682369, 0.185395, -0.185395, -0.682369)
  _separate_depth_observations(cfg, normalize=True)

  base_height_scan = RayCastSensorCfg(
    name="ssr_base_height_scan",
    frame=ObjRef(type="body", name=ROOT_BODY, entity="robot"),
    ray_alignment="world",
    pattern=GridPatternCfg(size=(0.0, 0.0), resolution=0.01),
    max_distance=2.0,
    exclude_parent_body=True,
  )
  body_height_scan = RayCastSensorCfg(
    name="ssr_body_height_scan",
    frame=ObjRef(type="body", name=ROOT_BODY, entity="robot"),
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(0.8, 0.8), resolution=0.1),
    max_distance=2.0,
    exclude_parent_body=True,
  )
  foot_height_scan = RayCastSensorCfg(
    name="ssr_foot_height_scan",
    frame=tuple(
      ObjRef(type="site", name=site_name, entity="robot")
      for site_name in KUAVO_S54_SOLE_SCAN_SITE_NAMES
    ),
    ray_alignment="yaw",
    pattern=GridPatternCfg(
      size=KUAVO_S54_SOLE_SCAN_SIZE,
      resolution=KUAVO_S54_SOLE_SCAN_RESOLUTION,
    ),
    max_distance=1.0,
    exclude_parent_body=True,
  )
  foothold_planning_scan = RayCastSensorCfg(
    name="ssr_foothold_planning_scan",
    frame=ObjRef(type="body", name=ROOT_BODY, entity="robot"),
    ray_alignment="yaw",
    # The planning raster is interpolated at the S54 sole's 2.5 cm support
    # points. The compact next-step window keeps 1024-env ray graphs tractable;
    # candidates outside it are conservatively treated as unsupported.
    pattern=GridPatternCfg(size=(1.0, 0.6), resolution=0.05),
    max_distance=2.0,
    exclude_parent_body=True,
  )
  toe_scanner_kwargs = dict(
    ray_alignment="base",
    pattern=GridPatternCfg(
      size=(0.0, 0.0),
      resolution=0.01,
      direction=(1.0, 0.0, 0.0),
    ),
    max_distance=1.0,
    exclude_parent_body=True,
    debug_vis=False,
  )
  feet_l_forward_scanner = RayCastSensorCfg(
    name="feet_l_forward_scanner",
    frame=ObjRef(type="body", name=FOOT_BODIES[0], entity="robot"),
    **toe_scanner_kwargs,
  )
  feet_r_forward_scanner = RayCastSensorCfg(
    name="feet_r_forward_scanner",
    frame=ObjRef(type="body", name=FOOT_BODIES[1], entity="robot"),
    **toe_scanner_kwargs,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    base_height_scan,
    body_height_scan,
    foot_height_scan,
    foothold_planning_scan,
    feet_l_forward_scanner,
    feet_r_forward_scanner,
  )

  actor_depth = cfg.observations["actor_depth"].terms["depth"]
  actor_depth.params.update(flatten=False, normalize=True)
  cfg.observations["actor"] = ObservationGroupCfg(
    terms={
      "proprioception": ObservationTermCfg(
        func=mdp.ssr_proprioception,
        params={
          "command_name": "twist",
          "angular_velocity_sensor": "robot/BodyGyro",
          "asset_cfg": controlled,
        },
      )
    },
    concatenate_terms=True,
    enable_corruption=True,
    history_length=5,
    flatten_history_dim=True,
  )
  cfg.observations["actor_depth"] = ObservationGroupCfg(
    terms={"depth": deepcopy(actor_depth)},
    concatenate_terms=True,
    enable_corruption=not play,
  )

  critic = cfg.observations["critic"]
  critic.terms.pop("depth", None)
  critic.terms["body_height_map"] = ObservationTermCfg(
    func=mdp.ssr_height_map,
    params={"sensor_name": body_height_scan.name, "miss_value": 2.0},
  )
  critic.terms["foot_height_maps"] = ObservationTermCfg(
    func=mdp.ssr_height_map,
    params={"sensor_name": foot_height_scan.name, "miss_value": 1.0},
  )
  cfg.observations["ssr_body_heights"] = ObservationGroupCfg(
    terms={
      "height_map": ObservationTermCfg(
        func=mdp.ssr_height_map,
        params={"sensor_name": body_height_scan.name, "miss_value": 2.0},
      )
    },
    concatenate_terms=True,
    enable_corruption=False,
  )
  cfg.observations["ssr_foot_heights"] = ObservationGroupCfg(
    terms={
      "height_maps": ObservationTermCfg(
        func=mdp.ssr_height_map,
        params={"sensor_name": foot_height_scan.name, "miss_value": 1.0},
      )
    },
    concatenate_terms=True,
    enable_corruption=False,
  )
  cfg.observations["ssr_base_velocity"] = ObservationGroupCfg(
    terms={
      "base_velocity": ObservationTermCfg(
        func=mdp.builtin_sensor,
        params={"sensor_name": "robot/BodyVel"},
      )
    },
    concatenate_terms=True,
    enable_corruption=False,
  )
  cfg.observations["ssr_foothold_terrain"] = ObservationGroupCfg(
    terms={
      "planning_map": ObservationTermCfg(
        func=mdp.ssr_foothold_planning_map,
        params={"sensor_name": foothold_planning_scan.name},
      )
    },
    concatenate_terms=True,
    enable_corruption=False,
  )
  cfg.observations["ssr_foothold_geometry"] = ObservationGroupCfg(
    terms={
      "geometry": ObservationTermCfg(
        func=mdp.ssr_foothold_geometry,
        params={
          "contact_sensor_name": "feet_ground_contact",
          "asset_cfg": sole_centers,
        },
      )
    },
    concatenate_terms=True,
    enable_corruption=False,
  )

  cfg.rewards = {
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=1.0,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=0.8,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "orientation": RewardTermCfg(
      func=mdp.ssr_orientation_reward,
      weight=0.5,
      params={"variance": 0.01},
    ),
    "angular_velocity_xy": RewardTermCfg(
      func=mdp.ssr_angular_velocity_reward,
      weight=0.25,
      params={"variance": 0.25},
    ),
    "base_height": RewardTermCfg(
      func=mdp.ssr_base_height_reward,
      weight=0.4,
      params={
        "sensor_name": base_height_scan.name,
        "target_height": KUAVO_S54_DEFAULT_BASE_HEIGHT,
        "variance": 0.01,
      },
    ),
    "action_rate": RewardTermCfg(func=mdp.ssr_action_rate, weight=-0.12),
    "action_smoothness": RewardTermCfg(
      func=mdp.ssr_action_smoothness, weight=-0.06
    ),
    "joint_velocity": RewardTermCfg(
      func=mdp.ssr_joint_velocity,
      weight=-0.96,
      params={
        "velocity_limit": KUAVO_S54_JOINT_VELOCITY_LIMIT,
        "asset_cfg": controlled,
      },
    ),
    "joint_torque": RewardTermCfg(
      func=mdp.ssr_joint_torque,
      weight=-0.6,
      params={
        "effort_limits": KUAVO_S54_CONTROLLED_JOINT_EFFORT_LIMITS,
        "asset_cfg": controlled,
      },
    ),
    "joint_deviation": RewardTermCfg(
      func=mdp.ssr_joint_deviation, weight=-1.8, params={"asset_cfg": controlled}
    ),
    "joint_position_limits": RewardTermCfg(
      func=mdp.ssr_joint_position_limit,
      weight=-1.5,
      params={"asset_cfg": controlled},
    ),
    "joint_velocity_limits": RewardTermCfg(
      func=mdp.ssr_joint_velocity_limit,
      weight=-6.0,
      params={
        "velocity_limit": KUAVO_S54_JOINT_VELOCITY_LIMIT,
        "asset_cfg": controlled,
      },
    ),
    "joint_torque_limits": RewardTermCfg(
      func=mdp.ssr_joint_torque_limit,
      weight=-6.0,
      params={
        "effort_limits": KUAVO_S54_CONTROLLED_JOINT_EFFORT_LIMITS,
        "asset_cfg": controlled,
      },
    ),
    "stand_still": RewardTermCfg(
      func=mdp.ssr_stand_still,
      weight=-0.12,
      params={
        "command_name": "twist",
        "command_threshold": 0.15,
        "asset_cfg": controlled,
      },
    ),
    "single_support": RewardTermCfg(
      func=mdp.ssr_single_support,
      weight=0.2,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.15,
      },
    ),
    "impact_velocity": RewardTermCfg(
      func=mdp.ssr_impact_velocity,
      weight=-1.3,
      params={"sensor_name": "feet_ground_contact", "asset_cfg": feet},
    ),
    "contact_slippage": RewardTermCfg(
      func=mdp.ssr_contact_slippage,
      weight=-0.2,
      params={"sensor_name": "feet_ground_contact", "asset_cfg": feet},
    ),
    "feet_air_time": RewardTermCfg(
      func=mdp.ssr_excess_air_time,
      weight=-2.0,
      params={"sensor_name": "feet_ground_contact", "target": 0.4},
    ),
    "feet_stumble": RewardTermCfg(
      func=mdp.feet_stumble,
      weight=-2.0,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "toe_touch": RewardTermCfg(
      func=mdp.toe_touch,
      # SSR velocity tracking is 8x lighter than S45 EMP, so do not copy its -5.0.
      weight=-1.0,
      params={
        "sensor_name_l": feet_l_forward_scanner.name,
        "sensor_name_r": feet_r_forward_scanner.name,
        "feet_length": KUAVO_S54_TOE_REACH,
        "margin": 0.015,
      },
    ),
    "feet_lateral_distance": RewardTermCfg(
      func=mdp.ssr_feet_lateral_distance,
      weight=0.08,
      params={
        "minimum_distance": 0.22,
        "variance": 0.03,
        "asset_cfg": feet,
      },
    ),
  }
  return cfg


def kuavo_s54_rough_blind_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create S54 rough terrain blind task — proprioception only, no depth camera.

  Builds on the S54 rough base with EMP rewards but strips all depth-related
  input (sensor, observation terms, camera-pitch randomisation).  The RL model
  uses ``MLPModel`` with 5-frame stacked proprioception.

  This is the rough-terrain counterpart of ``kuavo_s45_flat_blind_env_cfg``;
  unlike that flat variant the terrain generator / curriculum remain active.
  """
  cfg = _kuavo_s54_base_rough_env_cfg(play=play)
  _apply_s45_emp_rewards(cfg, controlled_joints=S54_CONTROLLED_JOINTS)

  # Blind: strip depth input (sensor + observations + events).
  for group in cfg.observations.values():
    group.terms.pop("depth", None)
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "depth"
  )
  cfg.events.pop("depth_camera_pitch", None)

  # Blind task: stack proprioception history (5-frame observation).
  for group in cfg.observations.values():
    group.history_length = 5

  # Feet height bonus: encourage lifting the feet higher during swing, which
  # helps clear obstacles on rough terrain — the blind policy cannot see them.
  cfg.rewards["feet_height"] = RewardTermCfg(
    func=mdp.feet_height,
    weight=1.0,
    params={
      "sensor_name": _S45_FEET_GROUND_SENSOR,
      "target_height": 0.2,
      "command_name": "twist",
      "command_threshold": 0.1,
      "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITES),
    },
  )

  return cfg


def kuavo_s45_rough_defm_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """S45 rough task with DeFM-aligned 42x42 depth and split depth obs groups.

  Mirrors the S54 DeFM setup: starts from the shared Kuavo rough config,
  applies the EMP-style reward port, resizes the depth camera to 42x42
  (so DeFM ViT-S14 patches align), and hoists the depth obs into independent
  ``actor_depth`` / ``critic_depth`` groups with raw metric depth (the DeFM
  encoder expects un-normalized depth, unlike the CNN encoder in
  :func:`kuavo_s45_rough_env_cfg`).
  """
  cfg = _kuavo_rough_env_cfg(
    robot_cfg=get_kuavo_s45_robot_cfg(),
    action_scale=KUAVO_S45_ACTION_SCALE,
    controlled_joints=S45_CONTROLLED_JOINTS,
    viewer_body=ROOT_BODY,
    has_waist=False,
    # Same URDF-derived waist camera pose as kuavo_s45_rough_env_cfg.
    depth_camera_parent_body="robot/base_link",
    depth_camera_pos=(0.168717483101422, 0.0, 0.01355599662743),
    depth_camera_quat=(0.6408367, 0.29887844, -0.29887844, -0.6408367),
    play=play,
  )
  _apply_s45_emp_rewards(cfg)
  # Bilateral symmetry loss (legs only): penalize left/right swing-amplitude
  # asymmetry via an EMA of deviation-from-default per side. Targets the
  # observed "one leg swings higher" failure without a gait phase or mirror-sign
  # table. Scoped to the DeFM task only; the CNN S45-Rough task (which also
  # calls _apply_s45_emp_rewards) is unaffected. Reward-level term, so it stays
  # compatible with the DeFM feature cache (unlike the PPO Symmetry extension).

  # cfg.rewards["leg_amplitude_symmetry"] = RewardTermCfg(
  #   func=mdp.bilateral_amplitude_symmetry,
  #   weight=-1.0,
  #   params={
  #     "asset_cfg": _scene_cfg(
  #       joint_names=r"leg_[lr][1-6]_joint", preserve_order=True
  #     ),
  #     "command_name": "twist",
  #     "command_threshold": 0.01,
  #     "alpha": 0.05,
  #   },
  # )

  # Expose true base linear velocity to the actor (in addition to the gyro-only
  # angular velocity it already sees). DeFM depth features alone are not enough
  # for the policy to recover its own body velocity, and feeding ground-truth
  # base_lin_vel measurably improves velocity tracking on rough terrain. We
  # reuse the critic's term (same robot/BodyVel sensor + ±0.5 m/s noise) so the
  # actor observation matches the privileged signal up to noise corruption.
  # cfg.observations["actor"].terms["base_lin_vel"] = deepcopy(
  #   cfg.observations["critic"].terms["base_lin_vel"]
  # )
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


def kuavo_s45_flat_blind_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Kuavo S45 flat terrain velocity configuration — blind (no depth input).

  Pure proprioception: inherits the full EMP reward set from
  :func:`kuavo_s45_rough_env_cfg` but strips the depth camera sensor and all
  depth observation terms. The RL model (:func:`kuavo_s45_flat_blind_ppo_runner_cfg`)
  uses ``MLPModel`` with only 1D ``actor`` / ``critic`` observation groups.

  Flat terrain overrides match :func:`kuavo_s45_flat_env_cfg`.
  """
  cfg = _kuavo_rough_env_cfg(
    robot_cfg=get_kuavo_s45_robot_cfg(),
    action_scale=KUAVO_S45_ACTION_SCALE,
    controlled_joints=S45_CONTROLLED_JOINTS,
    viewer_body=ROOT_BODY,
    has_waist=False,
    depth_camera_parent_body="robot/base_link",
    depth_camera_pos=(0.168717483101422, 0.0, 0.01355599662743),
    depth_camera_quat=(0.6408367, 0.29887844, -0.29887844, -0.6408367),
    play=play,
  )
  _apply_s45_emp_rewards(cfg)

  # Blind: strip depth input (sensor + observations).
  for group in cfg.observations.values():
    group.terms.pop("depth", None)
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "depth"
  )
  cfg.events.pop("depth_camera_pitch", None)

  # Blind task: stack proprioception history (5-frame observation).
  for group in cfg.observations.values():
    group.history_length = 5

  # Flat terrain overrides.
  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = 128

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.0, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

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
    twist_cmd.ranges.lin_vel_x = (-1.0, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg


def kuavo_s45_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Kuavo S45 flat terrain velocity configuration."""
  cfg = kuavo_s45_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = 128

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.0, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg


def kuavo_s45_rough_distill_env_cfg(
  play: bool = False,
  adaptive_terrain_curriculum: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Kuavo S45 rough terrain configuration for teacher-student distillation.

  Builds on :func:`kuavo_s45_rough_env_cfg` (EMP rewards + depth camera + CNN
  encoder for the student) and adds the ``terrain_scan`` ray-cast sensor plus
  three teacher-specific observation groups consumed by the frozen EMP teacher.

  Teacher observation groups (no corruption, no noise):

  * ``teacher_proprio`` — 5-frame history of 5 proprioceptive terms (84 dims
    per frame → 420 dims) matching the teacher's ``policy_obs`` input.
  * ``teacher_cmd`` — current velocity command ``[B, 3]``.
  * ``teacher_height`` — cropped 7×9 height scan ``[B, 63]``.

  The student continues to use the original ``actor`` / ``actor_depth`` groups
  (depth camera + CNN encoder, unchanged from S45-Rough).
  """
  cfg = kuavo_s45_rough_env_cfg(play=play)
  _enable_s45_foothold_reward(cfg)

  # During scheduled teacher intervention, displacement-based curriculum would
  # mostly measure the teacher and quickly push an untrained student to hard
  # terrain. Keep the BC data distribution stationary instead: environments
  # retain their initially sampled levels. Hybrid PPO fine-tuning opts back in.
  if not adaptive_terrain_curriculum:
    cfg.curriculum.pop("terrain_levels", None)

  # --- Re-add the terrain_scan sensor (removed by make_kuavo_velocity_env_cfg) ---
  terrain_scan = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="base_link", entity="robot"),
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
    max_distance=5.0,
    exclude_parent_body=True,
    # Isaac Lab's EMP RayCaster intersects only /World/ground. Limit this
    # individual height-scan query to terrain group 0. Parent-body exclusion
    # only excludes base_link and is not a terrain-only filtering mechanism.
    include_geom_groups=(0,),
    debug_vis=True,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (terrain_scan,)

  # --- Teacher observation groups ---

  # 1) teacher_proprio: 5-frame history of 84-dim per-frame proprioception.
  teacher_proprio_terms = {
    key: deepcopy(cfg.observations["actor"].terms[key])
    for key in ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions")
  }
  for term in teacher_proprio_terms.values():
    term.noise = None
  cfg.observations["teacher_proprio"] = ObservationGroupCfg(
    terms=teacher_proprio_terms,
    concatenate_terms=True,
    enable_corruption=False,
    history_length=5,
    flatten_history_dim=True,
  )

  # 2) teacher_cmd: current command.
  teacher_cmd_term = deepcopy(cfg.observations["actor"].terms["command"])
  teacher_cmd_term.noise = None
  cfg.observations["teacher_cmd"] = ObservationGroupCfg(
    terms={"command": teacher_cmd_term},
    concatenate_terms=True,
    enable_corruption=False,
    history_length=1,
  )

  # 3) teacher_height: cropped 63-dim height scan matching Isaac Lab's
  # height_scan_no_nan_clip preprocessing, including its final [-1, 1] clip.
  cfg.observations["teacher_height"] = ObservationGroupCfg(
    terms={"height_scan": ObservationTermCfg(
      func=mdp.teacher_height_obs,
      params={"sensor_name": "terrain_scan", "offset": 0.5},
      clip=(-1.0, 1.0),
      scale=1.0,
    )},
    concatenate_terms=True,
    enable_corruption=False,
    history_length=1,
  )
  cfg.observations["teacher_height_valid"] = ObservationGroupCfg(
    terms={"valid": ObservationTermCfg(
      func=mdp.teacher_height_valid_obs,
      params={"sensor_name": "terrain_scan"},
    )},
    concatenate_terms=True,
    enable_corruption=False,
    history_length=1,
  )

  # --- Restrict command ranges to match EMP teacher training distribution ---
  #
  # The frozen EMP teacher (trained on Isaac Lab S46) was trained with:
  #   vx ∈ [0.0, 1.0],  vy = 0.0,  wz ∈ [-0.8, 0.8]
  # Non-zero vy or negative vx would be out-of-distribution and cause the
  # teacher to produce unreasonable actions.
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_x = (0.0, 1.0)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (-0.8, 0.8)
  # Teacher was trained with always moving forward — no standing.
  twist_cmd.rel_standing_envs = 0.0

  return cfg
