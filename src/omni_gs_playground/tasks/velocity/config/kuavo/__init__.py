from mjlab.tasks.registry import register_mjlab_task
from omni_gs_playground.tasks.velocity.rl import (
  AMPVelocityOnPolicyRunner,
  VelocityOnPolicyRunner,
)
from rsl_rl.runners import DistillationRunner

from .env_cfgs import (
  kuavo_s45_flat_blind_env_cfg,
  kuavo_s45_flat_env_cfg,
  kuavo_s45_rough_defm_env_cfg,
  kuavo_s45_rough_distill_env_cfg,
  kuavo_s45_rough_env_cfg,
  kuavo_s45_slope_env_cfg,
  kuavo_s45_stairs_env_cfg,
  kuavo_s54_flat_env_cfg,
  kuavo_s54_head_cnn_rough_env_cfg,
  kuavo_s54_rough_env_cfg,
)
from .rl_cfg import (
  kuavo_s45_amp_ppo_runner_cfg,
  kuavo_s45_defm_ppo_runner_cfg,
  kuavo_s45_distill_ppo_runner_cfg,
  kuavo_s45_flat_blind_ppo_runner_cfg,
  kuavo_s45_ppo_runner_cfg,
  kuavo_s54_flat_ppo_runner_cfg,
  kuavo_s54_head_cnn_ppo_runner_cfg,
  kuavo_s54_head_moe_ppo_runner_cfg,
  kuavo_s54_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Kuavo-S45-Rough",
  env_cfg=kuavo_s45_rough_env_cfg(),
  play_env_cfg=kuavo_s45_rough_env_cfg(play=True),
  rl_cfg=kuavo_s45_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Kuavo-S45-DeFM-Rough",
  env_cfg=kuavo_s45_rough_defm_env_cfg(),
  play_env_cfg=kuavo_s45_rough_defm_env_cfg(play=True),
  rl_cfg=kuavo_s45_defm_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Kuavo-S45-Flat",
  env_cfg=kuavo_s45_flat_env_cfg(),
  play_env_cfg=kuavo_s45_flat_env_cfg(play=True),
  rl_cfg=kuavo_s45_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Kuavo-S45-Flat-Blind",
  env_cfg=kuavo_s45_flat_blind_env_cfg(),
  play_env_cfg=kuavo_s45_flat_blind_env_cfg(play=True),
  rl_cfg=kuavo_s45_flat_blind_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Kuavo-S54-Rough",
  env_cfg=kuavo_s54_rough_env_cfg(),
  play_env_cfg=kuavo_s54_rough_env_cfg(play=True),
  rl_cfg=kuavo_s54_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Kuavo-S54-Head-CNN-Rough",
  env_cfg=kuavo_s54_head_cnn_rough_env_cfg(),
  play_env_cfg=kuavo_s54_head_cnn_rough_env_cfg(play=True),
  rl_cfg=kuavo_s54_head_cnn_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Kuavo-S54-Head-MoE-Rough",
  env_cfg=kuavo_s54_head_cnn_rough_env_cfg(),
  play_env_cfg=kuavo_s54_head_cnn_rough_env_cfg(play=True),
  rl_cfg=kuavo_s54_head_moe_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Kuavo-S54-Flat",
  env_cfg=kuavo_s54_flat_env_cfg(),
  play_env_cfg=kuavo_s54_flat_env_cfg(play=True),
  rl_cfg=kuavo_s54_flat_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Kuavo-S45-AMP-Rough",
  env_cfg=kuavo_s45_rough_env_cfg(),
  play_env_cfg=kuavo_s45_rough_env_cfg(play=True),
  rl_cfg=kuavo_s45_amp_ppo_runner_cfg(),
  runner_cls=AMPVelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Kuavo-S45-Stairs",
  env_cfg=kuavo_s45_stairs_env_cfg(),
  play_env_cfg=kuavo_s45_stairs_env_cfg(play=True),
  rl_cfg=kuavo_s45_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Kuavo-S45-Slope",
  env_cfg=kuavo_s45_slope_env_cfg(),
  play_env_cfg=kuavo_s45_slope_env_cfg(play=True),
  rl_cfg=kuavo_s45_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Kuavo-S45-Rough-Distill",
  env_cfg=kuavo_s45_rough_distill_env_cfg(),
  play_env_cfg=kuavo_s45_rough_distill_env_cfg(play=True),
  rl_cfg=kuavo_s45_distill_ppo_runner_cfg(),
  runner_cls=DistillationRunner,
)
