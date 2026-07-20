import math
import os

import torch
import torch.nn as nn
import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner


class _ReorderObsOnnx(nn.Module):
  """Wrap an ONNX-export model so its 1D ``obs`` input is time-major.

  Training concatenates each observation term's ``history_length`` frames
  contiguously, i.e. the exported ``obs`` vector is *term-major*::

      [A1, A2, A3, A4, A5, B1, B2, B3, B4, B5]

  (A/B are terms such as ``joint_pos`` / ``joint_vel``; digits are frames,
  oldest→newest). Deployment instead provides the history *frame-major*
  (interleaved)::

      [A1, B1, A2, B2, A3, B3, A4, B4, A5, B5]

  This wrapper gathers the deployment-ordered ``obs`` back into the
  term-major order the trained normalizer / MLP expects, then delegates to
  the original export model. Any extra inputs (e.g. depth) pass through
  untouched, so the ONNX input/output signature is unchanged.
  """

  def __init__(self, inner: nn.Module, perm: torch.Tensor) -> None:
    super().__init__()
    self.inner = inner
    self.register_buffer("perm", perm)

  def forward(self, obs: torch.Tensor, *rest: torch.Tensor) -> torch.Tensor:
    return self.inner(obs.index_select(-1, self.perm), *rest)

  def get_dummy_inputs(self):
    return self.inner.get_dummy_inputs()  # type: ignore[operator]

  @property
  def input_names(self):
    return self.inner.input_names  # type: ignore[attr-defined]

  @property
  def output_names(self):
    return self.inner.output_names  # type: ignore[attr-defined]


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def _build_history_interleave_perm(self) -> torch.Tensor | None:
    """Map a deploy-ordered (frame-major) 1D obs vector to training order.

    Training lays out the actor 1D observation as term-major, frame-major
    within each term. Deployment feeds it frame-major (all terms at frame 0,
    then all terms at frame 1, ...). This returns a permutation ``perm`` such
    that ``training_obs = deploy_obs.index_select(-1, perm)``, or ``None`` when
    the two layouts already coincide (e.g. no history / ``history_length==1``).

    The interleaving is applied per 1D observation group and only when all
    terms in that group share the same ``history_length > 1`` (the case for the
    ``actor`` proprioception group, where ``_separate_depth_observations`` /
    the blind config set a uniform ``history_length=5``). Non-uniform or
    history-free groups are left as identity blocks.
    """
    obs_manager = self.env.unwrapped.observation_manager
    actor = self.alg.get_policy()

    perm: list[int] = []
    offset = 0
    for group in actor.obs_groups:
      term_names = obs_manager.active_terms[group]
      term_dims = obs_manager.group_obs_term_dim[group]
      bases: list[int] = []
      histories: list[int] = []
      for name, dim in zip(term_names, term_dims):
        term_cfg = obs_manager.get_term_cfg(group, name)
        total = int(math.prod(dim))
        if term_cfg.history_length > 0 and term_cfg.flatten_history_dim:
          hist = term_cfg.history_length
        else:
          hist = 1
        bases.append(total // hist)
        histories.append(hist)

      group_dim = sum(b * h for b, h in zip(bases, histories))
      uniform_hist = len(set(histories)) == 1
      if uniform_hist and histories and histories[0] > 1:
        hist = histories[0]
        per_frame = sum(bases)
        term_off = 0
        for base in bases:
          for f in range(hist):
            for k in range(base):
              # training index (term-major, frame-major within term)
              # <- deploy index (frame-major, terms interleaved per frame)
              perm.append(offset + f * per_frame + term_off + k)
          term_off += base
      else:
        # No history (or non-uniform): layouts already match.
        perm.extend(range(offset, offset + group_dim))
      offset += group_dim

    perm_t = torch.tensor(perm, dtype=torch.long)
    if torch.equal(perm_t, torch.arange(perm_t.numel())):
      return None
    return perm_t

  def export_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    """Export the policy to ONNX with a deploy-ordered (frame-major) obs input.

    Mirrors the base implementation but wraps the export model in
    :class:`_ReorderObsOnnx` so the ONNX ``obs`` input accepts the interleaved
    history layout used at deployment. Keeps ``dynamo=False`` to avoid the
    dynamic-axes deprecation warnings on torch>=2.9.
    """
    onnx_model = self.alg.get_policy().as_onnx(verbose=verbose)
    perm = self._build_history_interleave_perm()
    if perm is not None:
      onnx_model = _ReorderObsOnnx(onnx_model, perm)
    onnx_model.to("cpu")
    onnx_model.eval()
    os.makedirs(path, exist_ok=True)
    torch.onnx.export(
      onnx_model,
      onnx_model.get_dummy_inputs(),  # type: ignore[operator]
      os.path.join(path, filename),
      export_params=True,
      opset_version=18,
      verbose=verbose,
      input_names=onnx_model.input_names,  # type: ignore[arg-type]
      output_names=onnx_model.output_names,  # type: ignore[arg-type]
      dynamic_axes={},
      dynamo=False,
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
