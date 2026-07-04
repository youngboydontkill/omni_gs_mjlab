"""Kuavo velocity environment configurations."""

import math
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
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  GridPatternCfg,
  ObjRef,
  RayCastSensorCfg,
)
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
  depth_camera_quat: tuple[float, float, float, float] | None = None,
) -> ManagerBasedRlEnvCfg:
  """Create a Kuavo rough terrain velocity configuration.

  ``depth_camera_quat`` is optional: when ``None`` the depth sensor keeps the
  shared S54-derived orientation from :func:`make_kuavo_velocity_env_cfg`.
  Pass a ``(w, x, y, z)`` quaternion to override it for robots whose real-world
  camera mount uses a different pitch (e.g. S45 waist camera at 0.698 rad
  vs. the S54 head camera at ~0.593 rad encoded in the default quat).
  """
  cfg = make_velocity_env_cfg()

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
    depth_camera_quat=(0.6408367, 0.29887844, -0.29887844, -0.6408367),
    play=play,
  )
  _apply_s45_emp_rewards(cfg)
  _separate_depth_observations(cfg, normalize=False)
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


def _apply_s45_emp_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
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
  cfg.rewards["action_rate_l2"].weight = -0.005
  cfg.rewards["body_orientation_l2"].weight = -3.0
  # body_ang_vel (-0.05), joint_acc_l2 (-2.5e-7), joint_pos_limits (-10.0),
  # is_terminated (-200.0), stand_still (-1.0), self_collisions (-1.0) are
  # kept as-is (already match source intent).

  # 4) Inject EMP-specific terms.
  non_ankle_torque_joints = (r"leg_[lr][1-5]_joint", r"zarm_[lr][1-7]_joint")
  ankle_torque_joints = (r"leg_[lr]6_joint",)
  hip_joints = (r"leg_[lr][12]_joint",)
  arm_joints = (r"zarm_[lr][1-7]_joint",)
  all_controlled = _controlled_joints_cfg(S45_CONTROLLED_JOINTS)

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
      weight=-1.0e-5,
      params={
        "asset_cfg": _scene_cfg(actuator_names=ankle_torque_joints),
      },
    ),
    "dof_power_l2": RewardTermCfg(
      func=mdp.joint_power_l2,
      weight=-2.0e-5,
      params={
        "asset_cfg": _scene_cfg(
          joint_names=S45_CONTROLLED_JOINTS,
          actuator_names=S45_CONTROLLED_JOINTS,
          preserve_order=True,
        ),
      },
    ),
    "action_smoothness_l2": RewardTermCfg(
      func=mdp.action_acc_l2,
      weight=-0.01,
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
    # P1: 3.0 -> 1.0. Arms are 14/26 DoF; at 3.0 this term alone paid ~1.4/step
    # without requiring any walking, anchoring the static "stand still" optimum.
    "track_default_arm_pos": RewardTermCfg(
      func=mdp.track_default_arm_pos,
      weight=3.0,
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
      weight=-0.1,
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
      weight=-1.0,
      params={
        "sensor_name_l": "feet_l_forward_scanner",
        "sensor_name_r": "feet_r_forward_scanner",
        "feet_length": 0.178,
        "margin": 0.01,
      },
    ),
  })


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
  cfg.rewards["leg_amplitude_symmetry"] = RewardTermCfg(
    func=mdp.bilateral_amplitude_symmetry,
    weight=-1.0,
    params={
      "asset_cfg": _scene_cfg(
        joint_names=r"leg_[lr][1-6]_joint", preserve_order=True
      ),
      "command_name": "twist",
      "command_threshold": 0.01,
      "alpha": 0.05,
    },
  )
  # Expose true base linear velocity to the actor (in addition to the gyro-only
  # angular velocity it already sees). DeFM depth features alone are not enough
  # for the policy to recover its own body velocity, and feeding ground-truth
  # base_lin_vel measurably improves velocity tracking on rough terrain. We
  # reuse the critic's term (same robot/BodyVel sensor + ±0.5 m/s noise) so the
  # actor observation matches the privileged signal up to noise corruption.
  cfg.observations["actor"].terms["base_lin_vel"] = deepcopy(
    cfg.observations["critic"].terms["base_lin_vel"]
  )
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

  # Blind task: stack proprioception history (5-frame observation).
  # for group in cfg.observations.values():
  #   group.history_length = 5

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
