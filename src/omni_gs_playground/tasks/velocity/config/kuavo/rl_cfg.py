"""RL configuration for Kuavo velocity tasks."""

from dataclasses import asdict, dataclass
from typing import Any

from omni_gs_playground.assets.robots.kuavo import (
  KUAVO_S54_SOLE_SCAN_RESOLUTION,
  KUAVO_S54_SOLE_SCAN_SIZE,
)
from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


@dataclass
class RslRlDefmModelCfg(RslRlModelCfg):
  """RSL-RL model configuration with DeFM-specific options."""

  defm_cfg: dict[str, Any] | None = None


@dataclass
class RslRlMoEModelCfg(RslRlModelCfg):
  """RSL-RL model configuration with Mixture-of-Experts options.

  ``MoEModel`` inherits ``CNNModel`` so it also consumes ``cnn_cfg`` for the
  depth encoder; ``moe_cfg`` configures the gated expert head that sits between
  the encoder latent and the policy MLP.
  """

  moe_cfg: dict[str, Any] | None = None


@dataclass
class RslRlSsrModelCfg(RslRlModelCfg):
  """RSL-RL model configuration for the SSR cross-modal actor."""

  ssr_cfg: dict[str, Any] | None = None


@dataclass
class RslRlSsrPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """PPO configuration carrying SSR's bilateral data augmentation."""

  symmetry_cfg: dict[str, Any] | None = None
  foothold_cfg: dict[str, Any] | None = None


@dataclass
class RslRlAmpOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
  """Runner config that carries ``amp_cfg`` at the top level.

  ``RslRlPpoAlgorithmCfg`` does not have an ``amp_cfg`` field, so the config
  dict produced by ``asdict()`` would strip it.  By hoisting ``amp_cfg`` into
  the runner-level dataclass, ``asdict()`` preserves it.  The AMP runner
  (:class:`~omni_gs_playground.tasks.velocity.rl.runner.AMPVelocityOnPolicyRunner`)
  then injects it into ``cfg["algorithm"]["amp_cfg"]`` before PPO construction.
  """

  amp_cfg: dict[str, Any] | None = None


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

  # 根因修复(2026-07-06 run 复盘,踝 roll 扭动):该 run 的 Policy/mean_std 从
  # 0.5 单调涨到 1.377 且不收敛(异常发散,见 doc/action_std_in_ppo.md §4/§5.2),
  # 把高频噪声注入所有 action 维;对 kp=8 的软踝 roll,刚性关节能滤掉的噪声在踝
  # 上表现为可见扭动。仅降 init_std 上次已证明无效(std 仍涨到 1.377)。切到状态
  # 相关 std(actor 头输出 mean‖std)后,policy-loss 重新对 std 产生梯度,策略自信时
  # 主动拉低 std。与 S45-DeFM-Rough 一致(见 kuavo_s45_defm_ppo_runner_cfg 与
  # doc/action_std_in_ppo.md)。CNNModel 经 MLPModel 路径兼容;导出走 deterministic
  # mean,ONNX/缓存不受影响。init_std 0.3 进一步压低早期踝噪声(0.3*0.25=0.075 rad
  # < 踝 roll ±0.262 rad 范围)。
  # cfg.actor.distribution_cfg["class_name"] = "HeteroscedasticGaussianDistribution"
  # cfg.actor.distribution_cfg["init_std"] = 0.3
  # # 截断极端采样:原 clip_actions=None,高 std 下 raw action 可远超关节范围。
  # # 6.0 对所有关节 offset(scale=0.25 -> ±1.5 rad)都安全,仅作卫生截断。
  # cfg.clip_actions = 6.0

  cfg.experiment_name = "kuavo_s45_velocity"
  return cfg


def kuavo_s45_flat_blind_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the MLP runner configuration for the blind S45 flat task.

  Pure proprioception: ``MLPModel`` (default) consumes only the 1D
  ``actor`` / ``critic`` observation groups. No depth encoder.
  """
  cfg = _kuavo_base_ppo_runner_cfg()
  # MLPModel is the default class_name — no override needed.
  # Default obs_groups = {"actor": ("actor",), "critic": ("critic",)}.
  cfg.experiment_name = "kuavo_s45_flat_blind_velocity"
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


def kuavo_s54_rough_cnn_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the CNN runner for S54 rough with waist camera + EMP rewards."""
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
  cfg.algorithm.entropy_coef = 5.0e-3
  cfg.actor.distribution_cfg["init_std"] = 0.5
  cfg.experiment_name = "kuavo_s54_cnn_velocity"
  return cfg


def kuavo_s54_rough_ssr_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the paper-aligned SSR runner for the 27-DoF Kuavo-S54."""
  cfg = _kuavo_base_ppo_runner_cfg()
  cfg.actor = RslRlSsrModelCfg(
    class_name="SSRModel",
    hidden_dims=(1024, 512, 128),
    activation="elu",
    obs_normalization=True,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
      "std_range": (0.2, 5.0),
    },
    ssr_cfg={
      "history_length": 5,
      "proprio_group": "actor",
      "depth_group": "actor_depth",
      "foot_height_group": "ssr_foot_heights",
      "body_height_group": "ssr_body_heights",
      "velocity_group": "ssr_base_velocity",
    },
  )
  cfg.critic = RslRlModelCfg(
    hidden_dims=(512, 256, 128),
    activation="elu",
    obs_normalization=True,
  )
  cfg.obs_groups = {
    "actor": ("actor", "actor_depth"),
    "critic": ("critic",),
  }
  cfg.algorithm = RslRlSsrPpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=5.0e-3,
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=5.0e-4,
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
    symmetry_cfg={
      "data_augmentation_func": (
        "omni_gs_playground.tasks.velocity.mdp:kuavo_s54_ssr_symmetry"
      ),
      "use_data_augmentation": True,
      "use_mirror_loss": False,
      "mirror_loss_coeff": 0.0,
    },
    foothold_cfg={
      "state_group": "critic",
      "terrain_group": "ssr_foothold_terrain",
      "geometry_group": "ssr_foothold_geometry",
      "map_size": (1.0, 0.6),
      "map_resolution": 0.05,
      "sole_size": KUAVO_S54_SOLE_SCAN_SIZE,
      "sole_resolution": KUAVO_S54_SOLE_SCAN_RESOLUTION,
      "hidden_dims": (512, 256, 128),
      "activation": "elu",
      "learning_rate": 5.0e-4,
      "replay_capacity": 32768,
      "batch_size": 1024,
      "updates_per_iteration": 4,
      "max_pending_steps": 32,
      "train_min_samples": 1024,
      "reward_min_samples": 4096,
      "reward_min_updates": 100,
      "reward_min_terrain_level": 3.5,
      "reward_weight": 1.5,
      "reward_variance": 0.10,
      "height_threshold": 0.025,
      "min_std": 0.02,
      "max_std": 0.25,
      "max_target_distance": 1.5,
    },
  )
  cfg.num_steps_per_env = 24
  cfg.max_iterations = 20001
  cfg.experiment_name = "kuavo_s54_rough_ssr"
  return cfg


def kuavo_s54_rough_blind_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the MLP runner configuration for the blind S54 rough task.

  Pure proprioception: ``MLPModel`` (default) consumes only the 1D
  ``actor`` / ``critic`` observation groups.  No depth encoder.
  """
  cfg = _kuavo_base_ppo_runner_cfg()
  cfg.experiment_name = "kuavo_s54_blind_velocity"
  return cfg


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


def kuavo_s54_head_moe_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the MoE RL runner configuration for the head-controlled S54 task.

  迁移「Hiking in the Wild」§I 的 Mixture-of-Experts 策略（MoE-Loco 风格）：在
  CNN 深度 encoder 之后插入门控软加权专家头，再进策略 MLP。复用 Head-CNN 任务的
  env（归一化深度 + actor_depth/critic_depth 分组）与相同 CNN encoder 配置，仅把
  actor/critic 的 ``class_name`` 换成 ``MoEModel`` 并加 ``moe_cfg``。

  显存：``num_experts`` 个专家 MLP 参数量线性增长、前向 FLOPs ≈ ×num_experts；
  encoder 不变，rollout 缓存维度不变。默认 4 专家、专家隐层较窄以控参数量。
  encoder 不共享（``share_cnn_encoders=False``），与 Head-CNN 一致，让 critic 头
  在其完整特权输入上单独学习。
  """
  cnn_cfg = {
    "output_channels": (16, 32),
    "kernel_size": (5, 3),
    "stride": (2, 2),
    "padding": "zeros",
    "global_pool": "avg",
  }
  moe_cfg = {
    "num_experts": 4,
    "expert_hidden_dims": (256,),
    "gate_hidden_dims": (64,),
    "moe_output_dim": 256,
  }
  cfg = _kuavo_base_ppo_runner_cfg()
  cfg.actor = RslRlMoEModelCfg(
    class_name="MoEModel",
    hidden_dims=cfg.actor.hidden_dims,
    activation=cfg.actor.activation,
    obs_normalization=cfg.actor.obs_normalization,
    distribution_cfg=cfg.actor.distribution_cfg,
    cnn_cfg=cnn_cfg,
    moe_cfg=moe_cfg,
  )
  cfg.critic = RslRlMoEModelCfg(
    class_name="MoEModel",
    hidden_dims=cfg.critic.hidden_dims,
    activation=cfg.critic.activation,
    obs_normalization=cfg.critic.obs_normalization,
    cnn_cfg=cnn_cfg,
    moe_cfg=moe_cfg,
  )
  cfg.algorithm.share_cnn_encoders = False
  cfg.obs_groups = {
    "actor": ("actor", "actor_depth"),
    "critic": ("critic", "critic_depth"),
  }
  cfg.experiment_name = "kuavo_s54_head_moe_velocity"
  return cfg


def kuavo_s45_amp_ppo_runner_cfg() -> RslRlAmpOnPolicyRunnerCfg:
  """Create the CNN+AMP runner configuration for the Kuavo S45 rough task.

  Based on :func:`kuavo_s45_ppo_runner_cfg` (CNN encoder, non-shared), with
  an AMP discriminator that provides a style reward during rollout and is
  trained against retargeted motion-capture reference sequences.

  AMP is incompatible with DeFM feature caching; the CNN baseline already
  uses ``share_cnn_encoders=False`` so no extra mitigation is needed.
  """
  base = kuavo_s45_ppo_runner_cfg()

  amp_cfg = {
    "input_dim": 61 * 4,            # STATE_DIM(61) × seq_len(4)
    "hidden_dims": (256, 256),
    "activation": "elu",
    "reward_scale": 1.0,
    "grad_penalty_coeff": 10.0,
    "weight_decay": 1.0e-4,
    "learning_rate": 1.0e-3,
    "reward_coef": 0.5,             # style ← task reward mixing weight
    "seq_len": 4,
    "motion_data_dir": "/home/hitcsc/YX/GMR/motion_data/kuavo_s45_locomotion_pkl_v2/csv",
    "motion_dt": 1.0 / 30.0,
    # Only B4 through B15 (walk / turn) — exclude hop / leap / side_step / crouch.
    "include_keywords": ("b4", "b5", "b9", "b10", "b11", "b13", "b14", "b15"),
    "exclude_keywords": (),
  }

  return RslRlAmpOnPolicyRunnerCfg(
    seed=base.seed,
    num_steps_per_env=base.num_steps_per_env,
    max_iterations=base.max_iterations,
    obs_groups=base.obs_groups,
    save_interval=base.save_interval,
    experiment_name="kuavo_s45_amp_velocity",
    run_name=base.run_name,
    logger=base.logger,
    wandb_project=base.wandb_project,
    wandb_tags=base.wandb_tags,
    resume=base.resume,
    load_run=base.load_run,
    load_checkpoint=base.load_checkpoint,
    clip_actions=base.clip_actions,
    upload_model=base.upload_model,
    actor=base.actor,
    critic=base.critic,
    algorithm=base.algorithm,
    amp_cfg=amp_cfg,
  )


def kuavo_s45_distill_ppo_runner_cfg() -> dict:
  """Create the distillation runner configuration for the S45 rough task.

  This config uses the :class:`~rsl_rl.algorithms.Distillation` algorithm with:

  * **Student**: ``CNNModel`` (same as :func:`kuavo_s45_ppo_runner_cfg`).
    ``obs_groups = ("actor", "actor_depth")`` — depth camera + proprioception.
  * **Teacher**: ``EMPTeacherModel`` — frozen ``ActorCriticCNN`` loaded from
    ``doc/model_48350.pt``.  ``obs_groups = ("teacher_cmd", "teacher_proprio",
    "teacher_height")`` — command + 5-frame proprio history + height scan.

  The config is returned as a plain dict because the Distillation algorithm's
  ``construct_algorithm()`` expects ``"student"`` / ``"teacher"`` keys rather
  than the PPO ``"actor"`` / ``"critic"`` keys enforced by
  :class:`~mjlab.rl.RslRlOnPolicyRunnerCfg`.
  """
  cnn_cfg = {
    "output_channels": (16, 32),
    "kernel_size": (5, 3),
    "stride": (2, 2),
    "padding": "zeros",
    "global_pool": "avg",
  }

  return {
    "student": {
      "class_name": "CNNModel",
      "hidden_dims": (512, 256, 128),
      "activation": "elu",
      "obs_normalization": True,
      "distribution_cfg": {
        "class_name": "GaussianDistribution",
        "init_std": 0.5,
        "std_type": "scalar",
      },
      "cnn_cfg": cnn_cfg,
    },
    "teacher": {
      "class_name": "EMPTeacherModel",
      "checkpoint_path": "doc/model_48350.pt",
    },
    "algorithm": {
      "class_name": "Distillation",
      "num_learning_epochs": 5,
      "gradient_length": 15,
      "learning_rate": 1.0e-3,
      "max_grad_norm": 1.0,
      # Huber is less sensitive to rare recovery-action outliers. Weight the 12
      # leg joints above the 14 arm joints so the easy arm targets cannot
      # dominate the foothold-relevant behavior loss.
      "loss_type": "huber",
      "action_loss_weights": (1.5,) * 12 + (0.5,) * 14,
      # The teacher scan and camera depth are not isomorphic views. Keep global
      # latent matching off and reconstruct the local forward terrain instead.
      "latent_loss_coef": 0.0,
      "terrain_reconstruction_loss_coef": 0.1,
      "terrain_gradient_loss_coef": 0.05,
      "terrain_target_obs_group": "teacher_height",
      "terrain_mask_obs_group": "teacher_height_valid",
      "terrain_target_shape": (7, 9),
      # Per-environment DAgger-style intervention. Start from safe teacher
      # states and smoothly hand execution to the deploy-time student policy.
      "teacher_intervention_start": 1.0,
      "teacher_intervention_end": 0.05,
      "teacher_intervention_decay_updates": 12000,
      # Keep rollouts on the deploy-time mean action. The environment state
      # distribution must remain close to the frozen teacher for BC targets to
      # be meaningful.
      "student_rollout_stochastic": False,
    },
    "obs_groups": {
      "student": ("actor", "actor_depth"),
      "teacher": ("teacher_cmd", "teacher_proprio", "teacher_height"),
    },
    "num_steps_per_env": 24,
    "max_iterations": 24001,
    "save_interval": 1000,
    "experiment_name": "kuavo_s45_distill_velocity",
    "seed": 42,
  }


def kuavo_s45_distill_finetune_ppo_runner_cfg() -> dict:
  """Create hybrid PPO + teacher-BC configuration for reward fine-tuning.

  Load a distillation checkpoint with ``scripts/train.py --checkpoint-file``.
  PPO initializes the actor from ``student_state_dict``, keeps a fresh critic
  and optimizer, and decays the frozen-teacher behavior regularizer while task
  reward learns closed-loop recovery and foothold safety.
  """
  cfg = asdict(kuavo_s45_ppo_runner_cfg())
  cfg["teacher"] = {
    "class_name": "EMPTeacherModel",
    "checkpoint_path": "doc/model_48350.pt",
  }
  cfg["obs_groups"]["teacher"] = (
    "teacher_cmd",
    "teacher_proprio",
    "teacher_height",
  )
  cfg["algorithm"].update({
    "learning_rate": 1.0e-4,
    "entropy_coef": 5.0e-4,
    # Preserve the foothold-relevant leg policy while the fresh critic settles.
    # Keep the coefficient fixed for 3k updates, then decay over 12k updates.
    "behavior_loss_coef_start": 0.3,
    "behavior_loss_coef_end": 0.05,
    "behavior_loss_hold_updates": 3000,
    "behavior_loss_decay_updates": 12000,
    "behavior_loss_type": "huber",
    "behavior_action_weights": (1.5,) * 12 + (0.5,) * 14,
  })
  # Cross-algorithm loading intentionally keeps this configured PPO std rather
  # than the untrained 0.5 std stored by deterministic BC.
  cfg["actor"]["distribution_cfg"]["init_std"] = 0.15
  cfg["max_iterations"] = 5001
  cfg["save_interval"] = 500
  cfg["experiment_name"] = "kuavo_s45_distill_finetune_velocity"
  return cfg


def kuavo_s45_distill_finetune_curriculum_ppo_runner_cfg() -> dict:
  """Continue hybrid fine-tuning after the fixed-terrain stabilization phase."""
  cfg = kuavo_s45_distill_finetune_ppo_runner_cfg()
  cfg["max_iterations"] = 10001
  cfg["experiment_name"] = "kuavo_s45_distill_finetune_curriculum_velocity"
  return cfg
