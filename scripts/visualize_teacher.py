"""Visualise the EMP teacher model controlling the robot in a 3D viewer.

Creates the distillation environment, loads the frozen ``ActorCriticCNN``
teacher (``doc/model_48350.pt``), wraps it as an inference policy, and
launches the Viser interactive viewer — identical in experience to
``scripts/play.py`` but using the teacher's height-scan policy instead of a
trained student checkpoint.

Usage (default — Viser viewer)::

    uv run python scripts/visualize_teacher.py Kuavo-S45-Rough-Distill

Usage (static analysis — PNG plots + diagnostics)::

    uv run python scripts/visualize_teacher.py Kuavo-S45-Rough-Distill --static \\
        --steps 200 --output-dir outputs/teacher_viz
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommand
from mjlab.utils.torch import configure_torch_backends
from rsl_rl.modules.emp_modules import (
    S45_LAB_TO_MJCF,
    build_teacher,
    joint_order_mjcf_to_lab_term_major,
)


# ===================================================================
# Teacher policy wrapper
# ===================================================================


class TeacherInferencePolicy:
    """Thin wrapper that runs the frozen ``ActorCriticCNN`` teacher inside the
    standard viewer inference loop.

    The viewer/environment call chain is::

        viewer calls env.get_observations() → TensorDict
        viewer calls policy(obs_tensordict) → actions
        viewer calls env.step(actions)

    The TensorDict returned by ``RslRlVecEnvWrapper.get_observations()``
    preserves all named observation groups, including the teacher-specific
    ones (``teacher_proprio``, ``teacher_cmd``, ``teacher_height``).  This
    policy extracts them and passes them through the teacher network.
    """

    def __init__(self, teacher: torch.nn.Module) -> None:
        self._teacher = teacher

    def __call__(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run teacher forward from a TensorDict / dict of observation groups.

        Args:
            obs: Dict with keys ``teacher_proprio`` [B, 420],
                ``teacher_cmd`` [B, 3], ``teacher_height`` [B, 63].

        Returns:
            Teacher actions [B, 26] (pre-scale).
        """
        # The env returns teacher_proprio in term-major order (same as Isaac Lab
        # during training).  Only reorder joint dims from MJCF (grouped) → Lab
        # (interleaved) to match the checkpoint's training distribution.
        # Do NOT convert to frame-major — the normalizer was trained on term-major.
        proprio = joint_order_mjcf_to_lab_term_major(obs["teacher_proprio"])
        teacher_input = torch.cat(
            [obs["teacher_cmd"], proprio, obs["teacher_height"]],
            dim=-1,
        )  # [B, 486] = [cmd(3), proprio(420), height(63)] — Lab order

        if teacher_input.isnan().any():
            teacher_input = torch.nan_to_num(teacher_input, nan=0.0)
            print("[WARN] NaN in teacher_input — sanitised")

        with torch.no_grad():
            actions = self._teacher(teacher_input)  # [B, 26] — Lab order

        # Convert actions from Lab order → MJCF (env) order.
        idx = torch.tensor(S45_LAB_TO_MJCF, device=actions.device, dtype=torch.long)
        actions = actions[..., idx]

        if actions.isnan().any():
            actions = torch.nan_to_num(actions, nan=0.0)
            print("[WARN] NaN in teacher actions — sanitised")

        return actions


# ===================================================================
# Configuration
# ===================================================================


@dataclass(frozen=True)
class VisualizeTeacherConfig:
    """Configuration for the teacher model visualisation script."""

    checkpoint_path: str = "doc/model_48350.pt"
    """Path to the EMP teacher checkpoint."""

    num_envs: int = 1
    """Number of parallel environments (viewer shows env index 0)."""

    device: str | None = None
    """Device to run on (auto-detected when ``None``)."""

    # Viewer options.
    headless: bool = False
    """When True, skip the viewer and exit after loading."""
    viewer: Literal["auto", "native", "viser"] = "auto"
    """Viewer backend."""

    # Static-analysis options (only used when ``--static`` is passed).
    static: bool = False
    """Instead of launching the viewer, run a fixed number of rollout steps and
    save diagnostic PNG plots + text log."""
    steps: int = 500
    """Number of simulation steps for static mode."""
    output_dir: str = "outputs/teacher_viz"
    """Directory to save static-mode outputs."""


# ===================================================================
# Static-analysis mode helpers  (matplotlib PNGs)
# ===================================================================


def _import_plt() -> any:
    """Lazy-import matplotlib pyplot with Agg backend (head-safe)."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _run_static_analysis(
    task_id: str,
    cfg: VisualizeTeacherConfig,
    env: ManagerBasedRlEnv,
    teacher: torch.nn.Module,
) -> None:
    """Roll out the teacher for ``cfg.steps`` steps and produce PNG plots."""
    from mjlab.envs.mdp.observations import height_scan as raw_height_scan

    plt = _import_plt()

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_actions = env.action_manager.total_action_dim  # 26
    num_envs = env.num_envs
    steps = cfg.steps

    # Buffers.
    actions_buf = torch.zeros(steps, num_envs, num_actions)
    commands_buf = torch.zeros(steps, num_envs, 3)
    height_scan_raw_buf = torch.zeros(steps, num_envs, 187)
    height_scan_cropped_buf = torch.zeros(steps, num_envs, 63)
    action_mean_buf = torch.zeros(steps, num_envs)
    action_min_buf = torch.zeros(steps, num_envs)
    action_max_buf = torch.zeros(steps, num_envs)

    print(f"\n[Static] Running {steps} steps ...")
    num_nan_in, num_nan_out = 0, 0

    # Populate obs_buf with a dummy step (it is empty after __init__).
    dummy = torch.zeros(num_envs, num_actions, device=env.device)
    obs, _, _, _, _ = env.step(dummy)

    for s in range(steps):
        # Term-major proprio (same as Isaac Lab training).  Reorder joint dims
        # from MJCF (grouped) → Lab (interleaved) — no frame-major conversion.
        proprio = joint_order_mjcf_to_lab_term_major(obs["teacher_proprio"])
        teacher_in = torch.cat(
            [obs["teacher_cmd"], proprio, obs["teacher_height"]],
            dim=-1,
        )  # [B, 486] = [cmd(3), proprio(420), height(63)] — Lab order
        if teacher_in.isnan().any():
            num_nan_in += 1
            teacher_in = torch.nan_to_num(teacher_in, nan=0.0)
        with torch.no_grad():
            teacher_act = teacher(teacher_in)  # [B, 26] — Lab order
        # Convert actions from Lab order → MJCF (env) order.
        idx = torch.tensor(S45_LAB_TO_MJCF, device=teacher_act.device, dtype=torch.long)
        teacher_act = teacher_act[..., idx]
        if teacher_act.isnan().any():
            num_nan_out += 1
            teacher_act = torch.nan_to_num(teacher_act, nan=0.0)

        actions_buf[s] = teacher_act.cpu()
        commands_buf[s] = obs["teacher_cmd"].cpu()
        height_scan_cropped_buf[s] = obs["teacher_height"].cpu()
        full_scan = raw_height_scan(env, "terrain_scan", offset=0.0)
        height_scan_raw_buf[s] = full_scan.cpu()
        action_mean_buf[s] = teacher_act.mean(dim=-1).cpu()
        action_min_buf[s] = teacher_act.min(dim=-1).values.cpu()
        action_max_buf[s] = teacher_act.max(dim=-1).values.cpu()

        obs, _, _, _, _ = env.step(teacher_act)

        if (s + 1) % 100 == 0:
            print(
                f"  step {s + 1:>5d}/{steps}  |  "
                f"action mean: {teacher_act.mean().item():+.4f}  "
                f"range: [{teacher_act.min().item():+.4f}, {teacher_act.max().item():+.4f}]"
            )

    if num_nan_in or num_nan_out:
        print(f"  [WARN] NaN inputs: {num_nan_in}/{steps}  actions: {num_nan_out}/{steps}")

    # ── Generate plots ──────────────────────────────────────────
    print(f"\n[Static] Generating visualisations ...")

    # Height scan heatmap.
    raw_last = height_scan_raw_buf.mean(dim=0).reshape(1, 187)
    crop_last = height_scan_cropped_buf.mean(dim=0).reshape(1, 63)
    raw_2d = raw_last.reshape(-1, 11, 17)[-1]
    crop_2d = crop_last.reshape(-1, 7, 9)[-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    im1 = ax1.imshow(raw_2d.cpu().numpy(), cmap="terrain", aspect="auto")
    ax1.set_title("Raw Height Scan [11×17]")
    ax1.add_patch(
        plt.Rectangle((7.5, 1.5), 9, 7, fill=False, edgecolor="red", lw=2, ls="--")
    )
    plt.colorbar(im1, ax=ax1, label="Height (m)")
    im2 = ax2.imshow(crop_2d.cpu().numpy(), cmap="terrain", aspect="auto")
    ax2.set_title("Teacher Height Scan [7×9 cropped]")
    plt.colorbar(im2, ax=ax2, label="Height (m)")
    plt.tight_layout()
    plt.savefig(str(output_dir / "height_scan.png"), dpi=150)
    plt.close()
    print(f"  height_scan.png")

    # Actions time series.
    acts = actions_buf.mean(dim=1).cpu().numpy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    colors = plt.cm.tab20.colors
    for i in range(12):
        ax1.plot(acts[:, i], color=colors[i % 20], lw=0.7)
    ax1.set_title("Teacher Actions — Legs")
    ax1.grid(True, alpha=0.3)
    for i in range(12, 26):
        ax2.plot(acts[:, i], color=colors[(i - 12) % 20], lw=0.7)
    ax2.set_title("Teacher Actions — Arms")
    ax2.set_xlabel("Step")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_dir / "action_timeseries.png"), dpi=150)
    plt.close()
    print(f"  action_timeseries.png")

    # Commands.
    cmds = commands_buf.mean(dim=1).cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, lbl in enumerate(["vx", "vy", "wz"]):
        ax.plot(cmds[:, i], label=lbl, lw=0.9)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Velocity Commands")
    plt.tight_layout()
    plt.savefig(str(output_dir / "commands.png"), dpi=150)
    plt.close()
    print(f"  commands.png")

    # Action stats.
    mean_ = action_mean_buf.mean(dim=1).cpu().numpy()
    min_ = action_min_buf.mean(dim=1).cpu().numpy()
    max_ = action_max_buf.mean(dim=1).cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(mean_, label="mean", color="b", lw=0.9)
    ax.plot(min_, label="min", color="r", lw=0.7, alpha=0.7)
    ax.plot(max_, label="max", color="g", lw=0.7, alpha=0.7)
    ax.fill_between(range(len(mean_)), min_, max_, alpha=0.1, color="gray")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Action Statistics")
    plt.tight_layout()
    plt.savefig(str(output_dir / "action_stats.png"), dpi=150)
    plt.close()
    print(f"  action_stats.png")

    # Diagnostics text.
    lines = [
        "# Teacher model diagnostic log",
        f"# Steps: {steps}, Envs: {num_envs}",
        "",
        f"{'step':>5}  {'act_mean':>9}  {'act_min':>9}  {'act_max':>9}  "
        f"{'cmd_vx':>9}  {'cmd_vy':>9}  {'cmd_wz':>9}",
        "-" * 75,
    ]
    for s in range(steps):
        lines.append(
            f"{s:>5d}  {action_mean_buf[s].mean().item():>9.5f}  "
            f"{action_min_buf[s].mean().item():>9.5f}  "
            f"{action_max_buf[s].mean().item():>9.5f}  "
            f"{commands_buf[s, 0, 0].item():>9.3f}  "
            f"{commands_buf[s, 0, 1].item():>9.3f}  "
            f"{commands_buf[s, 0, 2].item():>9.3f}"
        )
    lines += [
        "",
        f"Actions — global mean: {action_mean_buf.mean().item():.5f}  "
        f"min: {action_min_buf.min().item():.5f}  max: {action_max_buf.max().item():.5f}",
    ]
    diag_path = output_dir / "diagnostics.txt"
    diag_path.write_text("\n".join(lines))
    print(f"  diagnostics.txt")

    print(f"\n[Static] All outputs → {output_dir.resolve()}")


# ===================================================================
# Viewer mode
# ===================================================================


def _patch_zero_velocity_viser_gui(env: ManagerBasedRlEnv) -> None:
    """Allow Viser to build a slider for an intentionally fixed command axis.

    Viser requires every command-axis maximum to be at least 0.1. The EMP
    teacher was trained with lin_vel_y=(0, 0), so changing the environment range
    permanently would alter its inference distribution. Temporarily widen the
    range only while the GUI controls are constructed, then restore it before
    simulation starts.
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


def _run_viewer(
    task_id: str,
    cfg: VisualizeTeacherConfig,
    env: ManagerBasedRlEnv,
    teacher: torch.nn.Module,
) -> None:
    """Wrap env, create teacher policy, and launch the 3D viewer."""
    # Wrap in RslRlVecEnvWrapper (required by the viewer infrastructure).
    wrapped_env = RslRlVecEnvWrapper(env)

    # Teacher inference policy — reads teacher groups from the TensorDict.
    policy = TeacherInferencePolicy(teacher)

    # Resolve viewer backend.
    resolved_viewer: str = cfg.viewer
    if resolved_viewer == "auto":
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        resolved_viewer = "native" if has_display else "viser"

    print(f"  Launching viewer: {resolved_viewer}")

    if resolved_viewer == "native":
        from mjlab.viewer import NativeMujocoViewer

        NativeMujocoViewer(wrapped_env, policy).run()
    elif resolved_viewer == "viser":
        from mjlab.viewer import ViserPlayViewer

        _patch_zero_velocity_viser_gui(env)
        ViserPlayViewer(wrapped_env, policy).run()
    else:
        raise RuntimeError(f"Unknown viewer: {resolved_viewer}")


# ===================================================================
# Entry point
# ===================================================================


def run_teacher_visualization(task_id: str, cfg: VisualizeTeacherConfig) -> None:
    """Shared setup, then dispatch to viewer or static-analysis mode."""
    configure_torch_backends()

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # ── Create environment ─────────────────────────────────────────
    print(f"Creating env (task={task_id}, num_envs={cfg.num_envs}) ...")
    env_cfg = load_env_cfg(task_id, play=True)
    env_cfg.scene.num_envs = cfg.num_envs
    render_mode = None if (cfg.headless or cfg.static) else "rgb_array"
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

    # Log observation groups.
    obs_keys = list(env_cfg.observations.keys())
    teacher_grps = [k for k in obs_keys if k.startswith("teacher")]
    print(f"  Total observation groups: {len(obs_keys)}")
    print(f"  Teacher groups: {teacher_grps}")

    # ── Load teacher model ─────────────────────────────────────────
    print(f"Loading teacher from {cfg.checkpoint_path} ...")
    teacher = build_teacher(cfg.checkpoint_path, device=device)
    teacher.eval()
    total_params = sum(p.numel() for p in teacher.parameters())
    print(f"  Parameters: {total_params:,}")

    # ── Dispatch ───────────────────────────────────────────────────
    if cfg.static:
        _run_static_analysis(task_id, cfg, env, teacher)
    else:
        _run_viewer(task_id, cfg, env, teacher)

    env.close()
    print("Done.")


def main():
    import mjlab.tasks  # noqa: F401
    import omni_gs_playground.tasks  # noqa: F401

    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
    )

    args = tyro.cli(
        VisualizeTeacherConfig,
        args=remaining_args,
        default=VisualizeTeacherConfig(),
        prog=sys.argv[0] + f" {chosen_task}",
    )

    run_teacher_visualization(chosen_task, args)


if __name__ == "__main__":
    main()
