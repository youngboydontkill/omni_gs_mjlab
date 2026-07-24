"""Script to play RL agent with RSL-RL."""

import os
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  no_terminations: bool = False
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


def _agent_get(agent: Any, key: str, default: Any = None) -> Any:
  """Read a runner setting from either a dataclass or a plain dict config."""
  return agent.get(key, default) if isinstance(agent, dict) else getattr(agent, key, default)


def _agent_to_dict(agent: Any) -> dict[str, Any]:
  """Serialize a runner config without mutating nested distillation settings."""
  return deepcopy(agent) if isinstance(agent, dict) else asdict(agent)


def _is_distillation_cfg(agent: Any) -> bool:
  """Return True when the task uses the Distillation algorithm."""
  algorithm = _agent_get(agent, "algorithm", None)
  if isinstance(algorithm, dict):
    return algorithm.get("class_name") == "Distillation"
  return getattr(algorithm, "class_name", None) == "Distillation"


def _checkpoint_load_cfg(agent: Any) -> dict[str, bool]:
  """Select the runner load keys that match the checkpoint layout."""
  if _is_distillation_cfg(agent):
    # Distill checkpoints store student/teacher, not PPO actor/critic.
    return {
      "student": True,
      "teacher": False,
      "optimizer": False,
      "iteration": False,
    }
  return {"actor": True}


def _patch_zero_velocity_viser_gui(env: ManagerBasedRlEnv) -> None:
  """Allow Viser to build a slider for an intentionally fixed command axis.

  Viser requires every command-axis maximum to be at least 0.1. Distill play keeps
  ``lin_vel_y=(0, 0)`` to match the EMP teacher distribution, so temporarily widen
  the range only while the GUI controls are constructed, then restore it.
  """
  for term_name in env.command_manager.active_terms:
    term = env.command_manager.get_term(term_name)
    if not isinstance(term, UniformVelocityCommand):
      continue

    original_create_gui = term.create_gui
    original_compute = term.compute
    zero_axis_indices = [
      index
      for index, axis in enumerate(("lin_vel_x", "lin_vel_y", "ang_vel_z"))
      if getattr(term.cfg.ranges, axis) == (0.0, 0.0)
    ]

    @wraps(original_create_gui)
    def create_gui_with_zero_range_support(*args, _term=term, **kwargs):
      original_ranges = _term.cfg.ranges
      zero_axes = [
        axis
        for axis in ("lin_vel_x", "lin_vel_y", "ang_vel_z")
        if getattr(original_ranges, axis) == (0.0, 0.0)
      ]
      for axis in zero_axes:
        setattr(original_ranges, axis, (-0.1, 0.1))
      try:
        return original_create_gui(*args, **kwargs)
      finally:
        for axis in zero_axes:
          setattr(original_ranges, axis, (0.0, 0.0))

    @wraps(original_compute)
    def compute_with_fixed_zero_axes(
      dt: float,
      _term=term,
      _original_compute=original_compute,
      _zero_axis_indices=zero_axis_indices,
    ) -> None:
      _original_compute(dt)
      if _zero_axis_indices:
        _term.vel_command_b[:, _zero_axis_indices] = 0.0

    term.create_gui = create_gui_with_zero_range_support
    term.compute = compute_with_fixed_zero_axes


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Check for local motion file first (works for both dummy and trained modes).
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion.npz (local file)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )
  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (
      Path("logs") / "rsl_rl" / _agent_get(agent_cfg, "experiment_name")
    ).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      wandb_run_path = getattr(cfg, "wandb_run_path", None)
      if wandb_run_path is None:
        raise ValueError(
          "`--checkpoint-file` is required when playing a trained policy."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(wandb_run_path)
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=_agent_get(agent_cfg, "clip_actions"))
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner_cfg = _agent_to_dict(agent_cfg)
    if _is_distillation_cfg(agent_cfg):
      print("[INFO]: Using DistillationRunner student policy for play")
    runner = runner_cls(env, runner_cfg, device=device)
    runner.load(
      str(resume_path),
      load_cfg=_checkpoint_load_cfg(agent_cfg),
      strict=True,
      map_location=device,
    )
    policy = runner.get_inference_policy(device=device)

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    NativeMujocoViewer(env, policy).run()
  elif resolved_viewer == "viser":
    _patch_zero_velocity_viser_gui(env.unwrapped)
    ViserPlayViewer(env, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401
  import omni_gs_playground.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
