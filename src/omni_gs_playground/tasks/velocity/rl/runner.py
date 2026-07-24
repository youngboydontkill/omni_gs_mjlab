import os

import torch
import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner
from rsl_rl.utils.onnx_export import (
  ReorderObsOnnx,
  build_history_interleave_perm,
  export_policy_to_onnx as _export_onnx,
)


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def _build_history_interleave_perm(self) -> torch.Tensor | None:
    """Compute frame-major to term-major permutation.

    Delegates to :func:`~rsl_rl.utils.onnx_export.build_history_interleave_perm`.
    """
    return build_history_interleave_perm(
      self.env.unwrapped.observation_manager, self.alg.get_policy()
    )

  def export_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    """Export the policy to ONNX with a deploy-ordered (frame-major) obs input.

    Delegates to :func:`~rsl_rl.utils.onnx_export.export_policy_to_onnx`.
    """
    _export_onnx(
      self.env.unwrapped.observation_manager,
      self.alg.get_policy(),
      path,
      filename,
      verbose,
    )

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = "policy.onnx"
    self.export_policy_to_onnx(policy_path, filename)
    run_name: str = (
      wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
    )  # type: ignore[assignment]
    onnx_path = os.path.join(policy_path, filename)
    metadata = get_base_metadata(self.env.unwrapped, run_name)
    attach_metadata_to_onnx(onnx_path, metadata)
    if self.logger.logger_type in ["wandb"]:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class AMPVelocityOnPolicyRunner(VelocityOnPolicyRunner):
  """Velocity runner with AMP (Adversarial Motion Priors) style-reward training.

  Before constructing PPO the runner:

  1. Extracts ``amp_cfg`` from the top-level config dict (placed there by
     ``asdict(RslRlAmpOnPolicyRunnerCfg)``) and injects it into
     ``train_cfg["algorithm"]["amp_cfg"]`` so PPO can build the discriminator.
  2. Reads the resolved joint IDs from the observation manager and builds an
     :class:`~omni_gs_playground.tasks.velocity.amp.amp_obs.AMPStateComputer`.
  3. Re-wraps the environment with
     :class:`~omni_gs_playground.tasks.velocity.amp.env_wrapper.AMPVecEnvWrapper`
     so every ``step()`` appends ``extras["amp_obs"]``.
  4. Calls ``super().__init__()`` — PPO receives ``amp_cfg`` and constructs the
     discriminator.
  5. Loads the retargeted motion dataset and attaches a reference sampler so the
     discriminator is trained during ``update()``.

  When ``amp_cfg`` is absent the runner falls back to vanilla
  ``VelocityOnPolicyRunner`` behaviour.
  """

  env: RslRlVecEnvWrapper

  def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu"):
    # -- extract amp_cfg from the top level ----------------------------------
    amp_cfg: dict | None = train_cfg.pop("amp_cfg", None)

    if amp_cfg is not None:
      amp_cfg = dict(amp_cfg)

      # Resolve joint IDs from the already-initialised observation manager.
      # ``asset_cfg.joint_ids`` is a ``list[int]`` when a subset of joints is
      # selected, or ``slice(None)`` when all joints are selected (optimisation).
      obs_manager = env.unwrapped.observation_manager
      joint_pos_cfg = obs_manager.get_term_cfg("actor", "joint_pos")
      joint_ids: list[int] | slice = joint_pos_cfg.params["asset_cfg"].joint_ids
      if isinstance(joint_ids, slice):
        n_joints = env.unwrapped.scene["robot"].data.joint_pos.shape[1]
        joint_ids = list(range(n_joints))

      from omni_gs_playground.tasks.velocity.amp.amp_obs import (
        AMPStateComputer,
      )
      from omni_gs_playground.tasks.velocity.amp.env_wrapper import (
        AMPVecEnvWrapper,
      )

      state_computer = AMPStateComputer(
        env.unwrapped,
        torch.tensor(joint_ids, dtype=torch.long),
      )

      seq_len = amp_cfg.pop("seq_len", 4)
      motion_data_dir = amp_cfg.pop("motion_data_dir", None)
      motion_dt = amp_cfg.pop("motion_dt", 1.0 / 30.0)
      include_kw = amp_cfg.pop("include_keywords", None)
      exclude_kw = amp_cfg.pop("exclude_keywords", None)

      clip_actions = train_cfg.get("clip_actions")
      env = AMPVecEnvWrapper(
        env.unwrapped,
        state_computer,
        seq_len=seq_len,
        clip_actions=clip_actions,
      )

      # Inject into algorithm dict so PPO.__init__ receives it.
      train_cfg["algorithm"]["amp_cfg"] = amp_cfg

    # -- standard PPO construction -------------------------------------------
    super().__init__(env, train_cfg, log_dir, device)

    # -- attach reference sampler (must happen AFTER PPO is built) ------------
    if amp_cfg is not None and motion_data_dir is not None:
      from omni_gs_playground.tasks.velocity.amp.motion_loader import (
        AMPSampler,
        MotionDataset,
      )

      dataset = MotionDataset(
        csv_dir=motion_data_dir,
        seq_len=seq_len,
        dt=motion_dt,
        include_keywords=include_kw,
        exclude_keywords=exclude_kw,
      )
      self.alg.set_amp_reference_sampler(AMPSampler(dataset))
