from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

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


def _range_gaussian_noise(
    depth: torch.Tensor,
    noise_std: float,
    noise_range: tuple[float, float],
) -> torch.Tensor:
    """Range-dependent additive Gaussian noise (paper §III-B2, F_sim step 2).

    Additive noise ``N(0, noise_std^2)`` is injected only into pixels whose
    (pre-normalization, metric) depth falls inside ``noise_range = [d_min,
    d_max]``. Pixels outside the valid sensing band (too near / too far) keep
    their value, emulating precision decay of a physical RGB-D sensor.

    ``depth`` is expected in the metric domain ``[near, far]`` here so the mask
    is meaningful; ``noise_std`` is therefore in meters.
    """
    d_min, d_max = noise_range
    noise = torch.randn_like(depth) * noise_std
    in_band = (depth >= d_min) & (depth <= d_max)
    return torch.where(in_band, depth + noise, depth)


def _disparity_white_regions(
    depth: torch.Tensor,
    far: float,
    prob: float,
    max_blocks: int,
    block_size: tuple[int, int],
) -> torch.Tensor:
    """Disparity artifact synthesis (paper §III-B2, F_sim step 3).

    With per-env probability ``prob`` a few contiguous rectangular "white
    regions" are set to ``far`` (max range), emulating binocular matching
    failures in over-exposed / textureless areas. Vectorized across the batch:
    each env independently decides whether to receive artifacts and where.

    Args:
      depth: ``[B, H, W]`` metric depth.
      far: value written into masked regions.
      prob: per-env probability of applying any white region this step.
      max_blocks: max number of rectangular blocks per affected env.
      block_size: ``(max_h, max_w)`` upper bound on each block's size in pixels.
    """
    b, h, w = depth.shape
    device = depth.device
    active = torch.rand(b, device=device) < prob  # [B]
    if not bool(active.any()):
        return depth

    max_bh, max_bw = block_size
    max_bh = max(1, min(max_bh, h))
    max_bw = max(1, min(max_bw, w))

    rows = torch.arange(h, device=device).view(1, 1, h, 1)  # [1,1,H,1]
    cols = torch.arange(w, device=device).view(1, 1, 1, w)  # [1,1,1,W]

    # Sample block geometry: [B, max_blocks].
    bh = torch.randint(1, max_bh + 1, (b, max_blocks), device=device)
    bw = torch.randint(1, max_bw + 1, (b, max_blocks), device=device)
    top = (torch.rand(b, max_blocks, device=device) * (h - bh).clamp(min=0).float()).long()
    left = (torch.rand(b, max_blocks, device=device) * (w - bw).clamp(min=0).float()).long()

    top_ = top.view(b, max_blocks, 1, 1)
    left_ = left.view(b, max_blocks, 1, 1)
    bot_ = (top + bh).view(b, max_blocks, 1, 1)
    right_ = (left + bw).view(b, max_blocks, 1, 1)

    # Per-block boolean mask, OR-reduced over blocks -> [B, H, W].
    block_mask = (
        (rows >= top_) & (rows < bot_) & (cols >= left_) & (cols < right_)
    ).any(dim=1)
    mask = block_mask & active.view(b, 1, 1)
    return torch.where(mask, depth.new_full((), far), depth)


def _gaussian_blur(
    depth: torch.Tensor,
    kernel_size: int,
    sigma: float,
) -> torch.Tensor:
    """Separable Gaussian blur (paper §III-B2, F_sim step 4) via depthwise conv.

    Simulates optical / motion blur. Applied to the whole batch (cheap on the
    42x42 depth maps used here); gating by probability is done by the caller.
    """
    k = kernel_size | 1  # force odd
    coords = torch.arange(k, device=depth.device, dtype=depth.dtype) - (k - 1) / 2.0
    g = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
    g = g / g.sum()
    kernel = torch.outer(g, g).view(1, 1, k, k)
    x = depth.unsqueeze(1)  # [B,1,H,W]
    x = F.conv2d(x, kernel, padding=k // 2)
    return x.squeeze(1)


def _ood_dropout(
    depth: torch.Tensor,
    prob: float,
    near: float,
    far: float,
) -> torch.Tensor:
    """OOD perturbation (paper §III-B2, F_sim step 6).

    With per-env Bernoulli probability ``prob`` the entire frame is replaced by
    uniform random depth in ``[near, far]``, forcing the policy to tolerate
    transient perceptual blackouts / glitches.
    """
    b = depth.shape[0]
    active = (torch.rand(b, device=depth.device) < prob).view(b, 1, 1)
    rand_frame = torch.rand_like(depth) * (far - near) + near
    return torch.where(active, rand_frame, depth)


def depth_image_obs(
    env: ManagerBasedRlEnv,
    sensor_name: str = "depth",
    near: float = 0.15,
    far: float = 5.0,
    flatten: bool = True,
    normalize: bool = True,
    corrupt: bool = False,
    noise_std: float = 0.0,
    noise_range: tuple[float, float] | None = None,
    white_prob: float = 0.0,
    white_max_blocks: int = 3,
    white_block_size: tuple[int, int] = (10, 10),
    blur_prob: float = 0.0,
    blur_kernel: int = 3,
    blur_sigma: float = 0.8,
    ood_prob: float = 0.0,
) -> torch.Tensor:
    """Depth observation with optional realistic sensor-noise synthesis.

    Base pipeline (always on): squeeze -> nan_to_num -> clip to ``[near, far]``
    -> optional normalize to ``[-1, 1]`` -> optional flatten. This matches the
    original behaviour exactly when ``corrupt=False`` (all synthesis stages are
    skipped), so existing tasks are bitwise-unchanged unless they opt in.

    When ``corrupt=True`` the paper's ``F_sim`` degradation chain (Hiking in the
    Wild, §III-B2) is applied in the **metric** domain, right after clip and
    before normalization:

      1. range-dependent Gaussian noise (``noise_std`` in meters, only inside
         ``noise_range``; defaults to ``[near, far]`` when unset),
      2. disparity "white regions" set to ``far`` (``white_prob``),
      3. Gaussian blur (``blur_prob`` / ``blur_kernel`` / ``blur_sigma``),
      4. out-of-distribution whole-frame dropout to random depth (``ood_prob``).

    ``corrupt`` is intended to be driven by the observation group's
    ``enable_corruption`` flag: actor groups train with noise, critic groups and
    play mode (both ``enable_corruption=False``) receive clean depth. Each stage
    is additionally a no-op when its probability / std is 0, so noise can be
    tuned per task. DeFM tasks use ``normalize=False`` (metric depth); CNN tasks
    use ``normalize=True``.
    """
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

    # 论文 §III-B2 F_sim 退化链：仅在 corruption 开启时作用于米制域。
    if corrupt:
        band = noise_range if noise_range is not None else (near, far)
        if noise_std > 0.0:
            depth = _range_gaussian_noise(depth, noise_std, band)
        if white_prob > 0.0:
            depth = _disparity_white_regions(
                depth, far, white_prob, white_max_blocks, white_block_size
            )
        if blur_prob > 0.0:
            blurred = _gaussian_blur(depth, blur_kernel, blur_sigma)
            do_blur = (torch.rand(depth.shape[0], device=depth.device) < blur_prob)
            depth = torch.where(do_blur.view(-1, 1, 1), blurred, depth)
        if ood_prob > 0.0:
            depth = _ood_dropout(depth, ood_prob, near, far)
        # 加噪后再次裁剪，保证落在 [near, far]，normalize 不越界。
        depth = depth.clamp(near, far)

    if normalize:
        depth = (depth - near) / (far - near)
        depth = depth * 2.0 - 1.0

    if flatten:
        # 给 MLP policy 用: [num_envs, H * W]
        return depth.reshape(env.num_envs, -1)

    # 给视觉 encoder policy 用: [num_envs, 1, H, W]
    return depth.unsqueeze(1)
