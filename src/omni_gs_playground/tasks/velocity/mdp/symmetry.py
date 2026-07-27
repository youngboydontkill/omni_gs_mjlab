"""Bilateral symmetry transforms for the Kuavo-S54 SSR task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

from omni_gs_playground.assets.robots.kuavo import KUAVO_S54_SOLE_SCAN_SHAPE

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_JOINT_PERM = (
  6, 7, 8, 9, 10, 11,
  0, 1, 2, 3, 4, 5,
  12,
  20, 21, 22, 23, 24, 25, 26,
  13, 14, 15, 16, 17, 18, 19,
)
_JOINT_SIGNS = (
  -1.0, -1.0, 1.0, 1.0, 1.0, -1.0,
  -1.0, -1.0, 1.0, 1.0, 1.0, -1.0,
  -1.0,
  1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0,
  1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0,
)
_SOLE_ROWS, _SOLE_COLS = KUAVO_S54_SOLE_SCAN_SHAPE
_FOOT_HEIGHT_START = 186
_FOOT_HEIGHT_SIZE = 2 * _SOLE_ROWS * _SOLE_COLS
_FOOT_HEIGHT_END = _FOOT_HEIGHT_START + _FOOT_HEIGHT_SIZE


def _mirror_joints(value: torch.Tensor) -> torch.Tensor:
  perm = torch.as_tensor(_JOINT_PERM, device=value.device)
  signs = value.new_tensor(_JOINT_SIGNS)
  return value[..., perm] * signs


def _mirror_proprioception(value: torch.Tensor) -> torch.Tensor:
  frames = value.reshape(*value.shape[:-1], 5, 90).clone()
  frames[..., 0:3] *= frames.new_tensor((-1.0, 1.0, -1.0))
  frames[..., 3:6] *= frames.new_tensor((1.0, -1.0, 1.0))
  frames[..., 6:9] *= frames.new_tensor((1.0, -1.0, -1.0))
  for start in (9, 36, 63):
    frames[..., start : start + 27] = _mirror_joints(
      frames[..., start : start + 27]
    )
  return frames.flatten(start_dim=-2)


def _mirror_critic(value: torch.Tensor) -> torch.Tensor:
  mirrored = value.clone()
  mirrored[..., 0:3] *= value.new_tensor((-1.0, 1.0, -1.0))
  mirrored[..., 3:6] *= value.new_tensor((1.0, -1.0, 1.0))
  mirrored[..., 6:9] *= value.new_tensor((1.0, -1.0, -1.0))
  for start in (9, 36, 63):
    mirrored[..., start : start + 27] = _mirror_joints(
      value[..., start : start + 27]
    )
  mirrored[..., 90:93] *= value.new_tensor((1.0, -1.0, 1.0))
  for start in (93, 95, 97):
    mirrored[..., start : start + 2] = value[..., start : start + 2].flip(-1)
  forces = value[..., 99:105].reshape(*value.shape[:-1], 2, 3)
  forces = forces.flip(-2) * value.new_tensor((1.0, -1.0, 1.0))
  mirrored[..., 99:105] = forces.flatten(start_dim=-2)
  body = value[..., 105:186].reshape(*value.shape[:-1], 9, 9)
  mirrored[..., 105:186] = body.flip(-2).flatten(start_dim=-2)
  feet = value[..., _FOOT_HEIGHT_START:_FOOT_HEIGHT_END].reshape(
    *value.shape[:-1], 2, _SOLE_ROWS, _SOLE_COLS
  )
  mirrored[..., _FOOT_HEIGHT_START:_FOOT_HEIGHT_END] = (
    feet.flip(-3).flip(-2).flatten(start_dim=-3)
  )
  return mirrored


def kuavo_s54_ssr_symmetry(
  env: ManagerBasedRlEnv,
  obs: TensorDict | None,
  actions: torch.Tensor | None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
  """Append sagittally mirrored SSR observations and actions for PPO."""
  del env
  augmented_obs = None
  if obs is not None:
    mirrored = obs.clone()
    mirrored["actor"] = _mirror_proprioception(obs["actor"])
    mirrored["actor_depth"] = obs["actor_depth"].flip(-1)
    mirrored["critic"] = _mirror_critic(obs["critic"])
    mirrored["ssr_body_heights"] = obs["ssr_body_heights"].reshape(
      *obs.batch_size, 9, 9
    ).flip(-2).flatten(start_dim=-2)
    mirrored["ssr_foot_heights"] = obs["ssr_foot_heights"].reshape(
      *obs.batch_size, 2, _SOLE_ROWS, _SOLE_COLS
    ).flip(-3).flip(-2).flatten(start_dim=-3)
    mirrored["ssr_base_velocity"] = (
      obs["ssr_base_velocity"] * obs["ssr_base_velocity"].new_tensor((1.0, -1.0, 1.0))
    )
    augmented_obs = torch.cat((obs, mirrored), dim=0)

  augmented_actions = None
  if actions is not None:
    augmented_actions = torch.cat((actions, _mirror_joints(actions)), dim=0)
  return augmented_obs, augmented_actions
