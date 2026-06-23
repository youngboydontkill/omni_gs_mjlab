from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + (2 * z_error)
  return torch.exp(-lin_vel_error / std**2)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  ang_vel_error = z_error + (0.05 * xy_error)
  return torch.exp(-ang_vel_error / std**2)


def body_orientation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward flat base orientation (robot being upright).

  If asset_cfg has body_ids specified, computes the projected gravity
  for that specific body. Otherwise, uses the root link projected gravity.
  """
  asset: Entity = env.scene[asset_cfg.name]

  # If body_ids are specified, compute projected gravity for that body.
  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
    body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    gravity_w = asset.data.gravity_vec_w  # [3]
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)  # [B, 3]
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
  else:
    # Use root link projected gravity.
    xy_squared = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  return xy_squared


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.squeeze(-1)


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
  return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize whole-body angular momentum to encourage natural arm swing."""
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
  return angmom_magnitude_sq


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float = 0.4,
  command_name: str | None = None,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Reward feet air time."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  air_time = sensor_data.current_air_time
  contact_time = sensor_data.current_contact_time
  in_contact = contact_time > 0.0
  in_mode_time = torch.where(in_contact, contact_time, air_time)
  single_stance = torch.mean(in_contact.float(), dim=1) == 0.5
  mode_time = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
  error = torch.abs(mode_time - threshold)
  reward = torch.clamp(threshold - error, min=0.0)
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command > command_threshold).float()
      reward *= scale
  return reward


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  command_name: str | None = None,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target clearance height, weighted by foot velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  delta = torch.abs(foot_z - target_height)  # [B, N]
  cost = torch.sum(delta * vel_norm, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


def feet_gait(
        env: ManagerBasedRlEnv,
        period: float,
        offset: list[float],
        threshold: float,
        command_threshold: float,
        command_name: str,
        sensor_name: str,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    is_contact = sensor.data.current_contact_time > 0
    global_phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
    offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype).view(1, -1)
    leg_phase = (global_phase + offsets) % 1.0
    is_stance = (leg_phase < threshold)
    reward = (is_stance == is_contact).float().mean(dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command > command_threshold).float()
            reward *= scale
    return reward


class feet_swing_height:
  """Penalize deviation from target swing height, evaluated at landing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.sensor_name = cfg.params["sensor_name"]
    self.site_names = cfg.params["asset_cfg"].site_names
    self.peak_heights = torch.zeros(
      (env.num_envs, len(self.site_names)), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    foot_heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class variable_posture:
  """Penalize deviation from default pose with speed-dependent tolerance.

  Uses per-joint standard deviations to control how much each joint can deviate
  from default pose. Smaller std = stricter (less deviation allowed), larger
  std = more forgiving. The reward is: exp(-mean(error² / std²))

  Three speed regimes (based on linear + angular command velocity):
    - std_standing (speed < walking_threshold): Tight tolerance for holding pose.
    - std_walking (walking_threshold <= speed < running_threshold): Moderate.
    - std_running (speed >= running_threshold): Loose tolerance for large motion.

  Tune std values per joint based on how much motion that joint needs at each
  speed. Map joint name patterns to std values, e.g. {".*knee.*": 0.35}.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Unused.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))


def stand_still(
        env: ManagerBasedRlEnv,
        command_name: str,
        command_threshold: float = 0.1,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.square(diff_angle), dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command <= command_threshold).float()
            reward *= scale
    return reward


# ---------------------------------------------------------------------------
# Rewards ported from Leju-IsaacLab emp_env_cfg.py
# (covers ~14 new terms for Kuavo S45 EMP-style training).
# ---------------------------------------------------------------------------


def _command_speed_norm(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """L2 norm of the linear (xy) + |yaw| command magnitude."""
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  return torch.norm(command[:, :3], dim=1)


def _upright_gate(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """clamp(-grav_z, 0, 0.7) / 0.7 — zeroes rewards while falling."""
  asset: Entity = env.scene[asset_cfg.name]
  grav_z = asset.data.projected_gravity_b[:, 2]
  return torch.clamp(-grav_z, min=0.0, max=0.7) / 0.7


def feet_air_time_positive_biped(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  sensor_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward biped single-stance air time up to threshold.

  Mirrors Leju-IsaacLab's feet_air_time_positive_biped: keeps one foot in
  the air at a time, rewards the swing duration up to a cap, gates by the
  command magnitude, and damps the reward while the robot is tipping.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  air_time = sensor.data.current_air_time
  contact_time = sensor.data.current_contact_time
  assert air_time is not None and contact_time is not None, (
    f"Sensor '{sensor_name}' must have track_air_time=True."
  )
  in_contact = contact_time > 0.0
  in_mode_time = torch.where(in_contact, contact_time, air_time)
  single_stance = torch.sum(in_contact.int(), dim=1) == 1
  reward = torch.min(
    torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1
  )[0]
  reward = torch.clamp(reward, max=threshold)
  cmd_norm = _command_speed_norm(env, command_name)
  reward = reward * (cmd_norm > command_threshold).float()
  reward = reward * _upright_gate(env, asset_cfg)
  return reward


def feet_slide(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot body XY velocity while in contact (force-gated >1N)."""
  asset: Entity = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  if sensor.data.force_history is not None:
    force_mag = torch.norm(sensor.data.force_history, dim=-1)
    in_contact = (force_mag > 1.0).any(dim=-1)
  else:
    assert sensor.data.force is not None
    in_contact = torch.norm(sensor.data.force, dim=-1) > 1.0
  body_vel_xy = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :2]
  vel_norm = torch.norm(body_vel_xy, dim=-1)
  num = min(vel_norm.shape[1], in_contact.shape[1])
  return torch.sum(vel_norm[:, :num] * in_contact[:, :num].float(), dim=1)


def feet_contact_without_cmd(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward first ground contact while command ~zero (stand-still anchor)."""
  sensor: ContactSensor = env.scene[sensor_name]
  first_contact = sensor.compute_first_contact(env.step_dt).float()
  reward = torch.sum(first_contact, dim=-1)
  cmd_norm = _command_speed_norm(env, command_name)
  reward = reward * (cmd_norm < command_threshold).float()
  reward = reward * _upright_gate(env, asset_cfg)
  return reward


def track_default_arm_pos(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  alpha: float = 5.0,
) -> torch.Tensor:
  """Exp reward exp(-alpha * sum((q - q_default)^2)) on selected joints."""
  asset: Entity = env.scene[asset_cfg.name]
  q = asset.data.joint_pos[:, asset_cfg.joint_ids]
  q_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  sq_dist = torch.sum(torch.square(q - q_default), dim=1)
  return torch.exp(-alpha * sq_dist)


def contact_force_violation(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float,
  violation_max: float = float("inf"),
) -> torch.Tensor:
  """Penalize contact force exceeding threshold (clipped at violation_max)."""
  sensor: ContactSensor = env.scene[sensor_name]
  if sensor.data.force_history is not None:
    force_mag = torch.norm(sensor.data.force_history, dim=-1).max(dim=-1)[0]
  else:
    assert sensor.data.force is not None
    force_mag = torch.norm(sensor.data.force, dim=-1)
  violation = (force_mag - threshold).clamp(min=0.0, max=violation_max)
  return torch.sum(violation, dim=1)


def feet_stumble(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalise contacts where lateral force >> vertical force (stumble)."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  lateral = torch.norm(sensor.data.force[..., :2], dim=-1)
  vertical = torch.abs(sensor.data.force[..., 2])
  return torch.any(lateral > 3.0 * vertical, dim=1).float()


def no_feet_contact(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  command_threshold: float = 0.2,
  force_threshold: float = 5.0,
) -> torch.Tensor:
  """Penalise no-foot-contact while command is small (lagging response)."""
  sensor: ContactSensor = env.scene[sensor_name]
  if sensor.data.force_history is not None:
    force_mag = torch.norm(sensor.data.force_history, dim=-1).max(dim=-1)[0]
  else:
    assert sensor.data.force is not None
    force_mag = torch.norm(sensor.data.force, dim=-1)
  in_contact = force_mag > force_threshold
  no_contact = torch.sum(in_contact.int(), dim=1) == 0
  cmd_norm = _command_speed_norm(env, command_name)
  return torch.where(no_contact & (cmd_norm < command_threshold), 1.0, 0.0)


# Convex feasible region for the (leg_l5, leg_l6, leg_r5, leg_r6) parallel
# ankle subsystem. Copied verbatim from Leju-IsaacLab. Stored as plain tuples
# so the tensor is built lazily on the correct device.
_ANKLE_FEASIBLE_REGION = (
  (0.87, 0.5, -0.40),
  (-1.0, 0.0, -0.87),
  (-0.63, -0.78, -0.94),
  (0.87, -0.5, -0.4),
  (0.0, -1.0, -0.8),
  (-0.63, 0.78, -0.94),
  (0.0, 1.0, -0.8),
)


def illegal_dof_pos_barrier(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Log-barrier penalty for the ankle parallel-mechanism feasible region.

  Expects asset_cfg.joint_ids to resolve to exactly 4 joints in this order:
  leg_l5_joint, leg_l6_joint, leg_r5_joint, leg_r6_joint.
  """
  asset: Entity = env.scene[asset_cfg.name]
  ankle = asset.data.joint_pos[:, asset_cfg.joint_ids]  # [B, 4]
  feasible = torch.tensor(
    _ANKLE_FEASIBLE_REGION, device=ankle.device, dtype=ankle.dtype
  )  # [7, 3]
  eps = 0.05
  left = -feasible[:, -1] - torch.matmul(ankle[:, :2], feasible[:, :-1].T)
  right = -feasible[:, -1] - torch.matmul(ankle[:, 2:], feasible[:, :-1].T)
  left = torch.where((left < -eps).any(dim=-1, keepdim=True), 0.0, left)
  right = torch.where((right < -eps).any(dim=-1, keepdim=True), 0.0, right)
  max_penalty = 25.0
  l = torch.clamp(-torch.log(left + eps), min=0.0, max=max_penalty)
  r = torch.clamp(-torch.log(right + eps), min=0.0, max=max_penalty)
  return (l + r).sum(dim=1)


def feet_too_near_humanoid(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  threshold: float,
  feet_names: tuple[str, ...],
) -> torch.Tensor:
  """Penalize feet drifting too close in the lateral (body Y) direction."""
  asset: Entity = env.scene[asset_cfg.name]
  body_ids, _ = asset.find_bodies(feet_names)
  feet = asset.data.body_link_pos_w[:, body_ids, :]  # [B, 2, 3]
  root = asset.data.root_link_pos_w  # [B, 3]
  rel = feet - root.unsqueeze(1)
  quat = asset.data.root_link_quat_w[:, None, :].expand(-1, 2, -1)
  body = quat_apply_inverse(quat, rel)
  dist = torch.abs(body[:, 0, 1] - body[:, 1, 1])
  return (threshold - dist).clamp(min=0.0)


def fly(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float = 1.0,
) -> torch.Tensor:
  """Penalize all feet simultaneously airborne (force below threshold)."""
  sensor: ContactSensor = env.scene[sensor_name]
  if sensor.data.force_history is not None:
    force_mag = torch.norm(sensor.data.force_history, dim=-1).max(dim=-1)[0]
  else:
    assert sensor.data.force is not None
    force_mag = torch.norm(sensor.data.force, dim=-1)
  return (torch.sum((force_mag > threshold).int(), dim=-1) < 0.5).float()


def joint_deviation_l1(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """L1 deviation of selected joints from their default positions."""
  asset: Entity = env.scene[asset_cfg.name]
  diff = (
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  )
  return torch.sum(torch.abs(diff), dim=1)


def undesired_contacts(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float = 1.0,
) -> torch.Tensor:
  """Count (body, substep) pairs on undesired bodies exceeding threshold.

  Uses force history when available so brief mid-substep collisions count.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    return (force_mag > threshold).sum(dim=(1, 2)).float()
  if data.force is not None:
    force_mag = torch.norm(data.force, dim=-1)  # [B, N]
    return (force_mag > threshold).sum(dim=1).float()
  assert data.found is not None
  return data.found.sum(dim=1).float()


def joint_power_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalise |tau * qd| per joint (mechanical power).

  Despite the _l2 name this sums absolute power, not squared power.
  Uses actuator_force (indexed by actuator_ids) and joint_vel (indexed by
  joint_ids). Defensively trims to the shorter of the two because the caller
  may filter by joint regex on both — on Kuavo the mapping is 1:1 so the
  trim has no effect in practice.
  """
  asset: Entity = env.scene[asset_cfg.name]
  tau = asset.data.actuator_force[:, asset_cfg.actuator_ids]
  qd = asset.data.joint_vel[:, asset_cfg.joint_ids]
  num = min(tau.shape[1], qd.shape[1])
  return torch.sum(torch.abs(tau[:, :num] * qd[:, :num]), dim=1)

