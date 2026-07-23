#!/usr/bin/env python
"""Re-estimate teacher normalizer statistics in the MJLab environment.

Rolls out the frozen EMP teacher for N steps, collects the 423-dim normalizer
input (cmd + term-major Lab-ordered proprio), and computes running mean / std
with a minimum ``--min-std`` floor to avoid division by near-zero.

Usage::

    uv run python scripts/estimate_teacher_norm.py \\
        --task Kuavo-S45-Rough-Distill \\
        --steps 5000 --num-envs 2 \\
        --output doc/teacher_norm_reestimated.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends
from rsl_rl.modules.emp_modules import (
    build_teacher,
    joint_order_mjcf_to_lab_term_major,
    S45_LAB_TO_MJCF,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Kuavo-S45-Rough-Distill")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--output", type=str, default="doc/teacher_norm_reestimated.pt")
    parser.add_argument("--min-std", type=float, default=0.01,
                        help="Floor for estimated standard deviation")
    parser.add_argument("--checkpoint", type=str, default="doc/model_48350.pt")
    args = parser.parse_args()

    configure_torch_backends()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Env ---
    import omni_gs_playground.tasks  # noqa: register tasks
    env_cfg = load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = args.num_envs
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)

    # --- Teacher (with _std=1.0 as in build_teacher) ---
    teacher = build_teacher(args.checkpoint, device=device)
    teacher.eval()

    lab_to_mjcf = torch.tensor(S45_LAB_TO_MJCF, device=device, dtype=torch.long)

    # --- Collect observations ---
    # Populate history buffers with a dummy step.
    dummy = torch.zeros(args.num_envs, env.action_manager.total_action_dim, device=device)
    obs, _, _, _, _ = env.step(dummy)

    sum_obs = None    # running sum of norm inputs (cmd + proprio)
    sum_sq = None     # running sum of squares
    count = 0

    for s in range(args.steps):
        # Build normalizer input: [cmd(3), term_major_proprio(420)] — Lab joints
        cmd = obs["teacher_cmd"]                                    # [B, 3]
        proprio = joint_order_mjcf_to_lab_term_major(obs["teacher_proprio"])  # [B, 420]
        norm_in = torch.cat([cmd, proprio], dim=-1)                 # [B, 423]

        # Accumulate
        if sum_obs is None:
            sum_obs = norm_in.sum(dim=0)
            sum_sq = (norm_in ** 2).sum(dim=0)
        else:
            sum_obs += norm_in.sum(dim=0)
            sum_sq += (norm_in ** 2).sum(dim=0)
        count += args.num_envs

        # Teacher step
        teacher_in = torch.cat([cmd, proprio, obs["teacher_height"]], dim=-1)  # [B, 486]
        with torch.no_grad():
            actions = teacher(teacher_in)  # [B, 26] Lab order
        actions = actions[..., lab_to_mjcf]  # -> MJCF order

        obs, _, _, _, _ = env.step(actions)

        if (s + 1) % 500 == 0:
            print(
                f"  step {s+1:>5d}/{args.steps}  |  "
                f"action mean={actions.mean().item():+.4f}  "
                f"range=[{actions.min().item():+.2f}, {actions.max().item():+.2f}]  "
                f"samples={count}"
            )

    # Compute statistics
    total_mean = sum_obs / count
    total_var = (sum_sq / count - total_mean ** 2).clamp(min=0)
    total_std = torch.sqrt(total_var).clamp(min=args.min_std)

    print(f"\nEstimated normalizer ({count} samples, min_std={args.min_std}):")
    print(f"  mean in [{total_mean.min().item():.4f}, {total_mean.max().item():.4f}]")
    print(f"  std  in [{total_std.min().item():.4f}, {total_std.max().item():.4f}]")
    print(f"  count = {count}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "mean": total_mean.cpu(),
        "std": total_std.cpu(),
        "count": count,
    }, str(output_path))
    print(f"\nSaved to {output_path}")

    # --- Quick verification on last observation ---
    teacher2 = build_teacher(args.checkpoint, device=device)
    teacher2.actor_obs_normalizer._mean[:] = total_mean.unsqueeze(0).to(device)
    teacher2.actor_obs_normalizer._std[:] = total_std.unsqueeze(0).to(device)
    teacher2.eval()

    cmd = obs["teacher_cmd"]
    proprio = joint_order_mjcf_to_lab_term_major(obs["teacher_proprio"])
    teacher_in = torch.cat([cmd, proprio, obs["teacher_height"]], dim=-1)
    with torch.no_grad():
        actions2 = teacher2(teacher_in)
    actions2 = actions2[..., lab_to_mjcf]
    print(f"\nVerification (re-estimated normalizer):")
    print(f"  action mean={actions2.mean().item():+.4f}  "
          f"range=[{actions2.min().item():+.2f}, {actions2.max().item():+.2f}]")

    # --- Also evaluate with clean checkpoint normalizer for comparison ---
    teacher3 = build_teacher(args.checkpoint, device=device)
    teacher3.eval()
    cmd = obs["teacher_cmd"]
    proprio = joint_order_mjcf_to_lab_term_major(obs["teacher_proprio"])
    teacher_in = torch.cat([cmd, proprio, obs["teacher_height"]], dim=-1)
    with torch.no_grad():
        actions3 = teacher3(teacher_in)
    actions3 = actions3[..., lab_to_mjcf]
    print(f"\nComparison (checkpoint normalizer, _std=1.0):")
    print(f"  action mean={actions3.mean().item():+.4f}  "
          f"range=[{actions3.min().item():+.2f}, {actions3.max().item():+.2f}]")

    env.close()
    print("Done.")


if __name__ == "__main__":
    main()
