from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def foot_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (num_envs, num_sites)
  # Site positions can briefly become NaN/Inf if the physics solver fails
  # (e.g. on the first frame of a respawned env or when a foot site lands on
  # a degenerate ray query). Clamp to a safe range so the critic observation
  # never propagates NaN into PPO.
  return torch.nan_to_num(z, nan=0.0, posinf=1.0, neginf=0.0)


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def phase(env: ManagerBasedRlEnv, period: float, command_name: str) -> torch.Tensor:
    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    stand_mask = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    phase = torch.where(stand_mask.unsqueeze(1), torch.zeros_like(phase), phase)
    return phase


def depth_image_obs(
    env: ManagerBasedRlEnv,
    sensor_name: str = "depth",
    near: float = 0.15,
    far: float = 5.0,
    flatten: bool = True,
    normalize: bool = True,
) -> torch.Tensor:
    # [num_envs, H, W, 1]
    depth = env.scene[sensor_name].data.depth

    # [num_envs, H, W]
    depth = depth.squeeze(-1)

    # 防止 inf / nan 进入策略网络
    depth = torch.nan_to_num(
        depth,
        nan=far,
        posinf=far,
        neginf=near,
    )

    # 裁剪深度；DeFM 使用米制深度，其他视觉模型默认归一化到 [-1, 1]。
    depth = depth.clamp(near, far)
    if normalize:
        depth = (depth - near) / (far - near)
        depth = depth * 2.0 - 1.0

    if flatten:
        # 给 MLP policy 用: [num_envs, H * W]
        return depth.reshape(env.num_envs, -1)

    # 给视觉 encoder policy 用: [num_envs, 1, H, W]
    return depth.unsqueeze(1)
