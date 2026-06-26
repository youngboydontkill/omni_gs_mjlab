"""RL configuration for Kuavo velocity tasks."""

from dataclasses import dataclass
from typing import Any

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


@dataclass
class RslRlDefmModelCfg(RslRlModelCfg):
  """RSL-RL model configuration with DeFM-specific options."""

  defm_cfg: dict[str, Any] | None = None


def _kuavo_base_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the shared MLP runner configuration for Kuavo velocity tasks."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="kuavo_s54_velocity",
    save_interval=1000,
    num_steps_per_env=24,
    max_iterations=24001,
  )


def kuavo_s54_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the DeFM runner configuration for the 27-joint S54 rough task."""
  defm_cfg = {
    "model_name": "defm_vit_s14",
    "pretrained": True,
    "trainable": False,
    "target_size": 42,
    "token_feature_dim": 32,
  }
  cfg = _kuavo_base_ppo_runner_cfg()
  cfg.actor = RslRlDefmModelCfg(
    class_name="DefmModel",
    hidden_dims=cfg.actor.hidden_dims,
    activation=cfg.actor.activation,
    obs_normalization=cfg.actor.obs_normalization,
    distribution_cfg=cfg.actor.distribution_cfg,
    defm_cfg=defm_cfg,
  )
  cfg.critic = RslRlDefmModelCfg(
    class_name="DefmModel",
    hidden_dims=cfg.critic.hidden_dims,
    activation=cfg.critic.activation,
    obs_normalization=cfg.critic.obs_normalization,
    defm_cfg=defm_cfg,
  )
  cfg.algorithm.share_cnn_encoders = True
  cfg.obs_groups = {
    "actor": ("actor", "actor_depth"),
    "critic": ("critic", "critic_depth"),
  }
  cfg.experiment_name = "kuavo_s54_defm_velocity"
  return cfg


def kuavo_s45_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the CNN RL runner configuration for the Kuavo S45 rough task.

  Mirrors :func:`kuavo_s54_head_cnn_ppo_runner_cfg`: actor and critic both use
  ``CNNModel`` with two conv layers consuming the normalized depth image as a
  separate 2D observation group (``actor_depth`` / ``critic_depth``).
  Encoder weights are NOT shared between actor and critic so the value head
  can specialise on its full privileged input set.
  """
  cnn_cfg = {
    "output_channels": (16, 32),
    "kernel_size": (5, 3),
    "stride": (2, 2),
    "padding": "zeros",
    "global_pool": "avg",
  }
  cfg = _kuavo_base_ppo_runner_cfg()
  cfg.actor.class_name = "CNNModel"
  cfg.actor.cnn_cfg = cnn_cfg
  cfg.critic.class_name = "CNNModel"
  cfg.critic.cnn_cfg = cnn_cfg
  cfg.algorithm.share_cnn_encoders = False
  cfg.obs_groups = {
    "actor": ("actor", "actor_depth"),
    "critic": ("critic", "critic_depth"),
  }
  # P0 (see doc/terrain_curriculum_stuck.md §5/§6): the state-independent
  # scalar std never decays because the entropy bonus dominates its gradient
  # (advantage normalization nullifies the policy-loss gradient on std). Cut
  # entropy_coef 10x and halve init_std so the policy can commit to a gait
  # instead of injecting ~0.1 rad/step joint noise that keeps causing falls.
  cfg.algorithm.entropy_coef = 5.0e-3
  cfg.actor.distribution_cfg["init_std"] = 0.5

  cfg.experiment_name = "kuavo_s45_velocity"
  return cfg


def kuavo_s45_defm_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the DeFM runner configuration for the 23-joint S45 rough task.

  Mirrors :func:`kuavo_s54_ppo_runner_cfg`: actor/critic both use the frozen
  ``defm_vit_s14`` encoder with 42x42 depth + the project's 384->32 channel
  averaging, encoders are shared between actor and critic, and the PPO
  algorithm enables feature caching during rollout.
  """
  defm_cfg = {
    "model_name": "defm_vit_s14",
    "pretrained": True,
    "trainable": False,
    "target_size": 42,
    "token_feature_dim": 32,
  }
  cfg = _kuavo_base_ppo_runner_cfg()
  cfg.actor = RslRlDefmModelCfg(
    class_name="DefmModel",
    hidden_dims=cfg.actor.hidden_dims,
    activation=cfg.actor.activation,
    obs_normalization=cfg.actor.obs_normalization,
    distribution_cfg=cfg.actor.distribution_cfg,
    defm_cfg=defm_cfg,
  )
  cfg.critic = RslRlDefmModelCfg(
    class_name="DefmModel",
    hidden_dims=cfg.critic.hidden_dims,
    activation=cfg.critic.activation,
    obs_normalization=cfg.critic.obs_normalization,
    defm_cfg=defm_cfg,
  )
  cfg.algorithm.share_cnn_encoders = True
  cfg.obs_groups = {
    "actor": ("actor", "actor_depth"),
    "critic": ("critic", "critic_depth"),
  }
  # std 从「状态无关标量参数」改为「actor MLP 头输出的状态相关 std」(mean‖std)。
  # 根因修复:状态无关 std 在 advantage normalization 下 policy-loss 对其净梯度≈0,
  # entropy_coef 会单方面把 std 顶高不衰减(2026-06-25 run 卡在 ~1.59)。状态相关后
  # policy-loss 重新对 std 产生梯度,策略自信时主动把 std 拉低。详见
  # doc/action_std_in_ppo.md。导出走 deterministic mean,缓存/ONNX 均不受影响。
  cfg.actor.distribution_cfg["class_name"] = "HeteroscedasticGaussianDistribution"
  cfg.actor.distribution_cfg["init_std"] = 0.5  # 原本 1.0;降低早期探索噪声,减少乱抖摔
  cfg.experiment_name = "kuavo_s45_defm_velocity"
  return cfg


def kuavo_s54_flat_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the MLP runner configuration for the S54 flat task."""
  return _kuavo_base_ppo_runner_cfg()


def kuavo_s54_head_cnn_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the CNN RL runner configuration for the head-controlled S54 task."""
  cnn_cfg = {
    "output_channels": (16, 32),
    "kernel_size": (5, 3),
    "stride": (2, 2),
    "padding": "zeros",
    "global_pool": "avg",
  }
  cfg = _kuavo_base_ppo_runner_cfg()
  cfg.actor.class_name = "CNNModel"
  cfg.actor.cnn_cfg = cnn_cfg
  cfg.critic.class_name = "CNNModel"
  cfg.critic.cnn_cfg = cnn_cfg
  cfg.algorithm.share_cnn_encoders = False
  cfg.obs_groups = {
    "actor": ("actor", "actor_depth"),
    "critic": ("critic", "critic_depth"),
  }
  cfg.experiment_name = "kuavo_s54_head_cnn_velocity"
  return cfg
