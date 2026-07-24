"""Re-export ONNX policy from a saved checkpoint.

Usage:
  # Auto-detect task ID from experiment directory name (standard naming):
  uv run python scripts/reexport_onnx.py logs/rsl_rl/kuavo_s45_flat_blind_velocity/2026-07-07_17-19-44/model_19000.pt

  # Explicit task ID for non-standard experiment directory names:
  uv run python scripts/reexport_onnx.py logs/rsl_rl/kuavo_s45_velocity/2026-07-06_16-30-20/model_39999.pt Kuavo-S45-Rough
"""

import argparse
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch

# Import tasks to populate the registry.
import mjlab.tasks  # noqa: F401
import omni_gs_playground.tasks  # noqa: F401

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from rsl_rl.utils.onnx_export import build_history_interleave_perm


def _infer_task_id(exp_name: str) -> str | None:
    """Map experiment directory name to task ID."""
    all_tasks = list_tasks()
    task_map: dict[str, str] = {}
    for t in all_tasks:
        key = t.lower().replace("-", "_")
        task_map[key] = t
    if exp_name in task_map:
        return task_map[exp_name]
    base = exp_name.replace("_velocity", "")
    return task_map.get(base)


def main():
    parser = argparse.ArgumentParser(description="Re-export ONNX policy from a checkpoint.")
    parser.add_argument("checkpoint", help="Path to the .pt checkpoint file.")
    parser.add_argument(
        "task",
        nargs="?",
        default=None,
        help="Task ID (e.g. Kuavo-S45-Rough). Auto-detected from experiment directory "
        "name when the name follows the standard convention.",
    )
    args = parser.parse_args()

    checkpoint_path = os.path.abspath(args.checkpoint)
    if not os.path.isfile(checkpoint_path):
        print(f"ERROR: checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    exp_dir = Path(checkpoint_path).parent.parent
    exp_name = exp_dir.name

    task_id = args.task or _infer_task_id(exp_name)
    if task_id is None:
        print(f"ERROR: could not infer task ID from experiment name '{exp_name}'")
        all_tasks = list_tasks()
        print(f"Available tasks: {all_tasks}")
        print(f"Please specify the task ID explicitly as the second argument.")
        sys.exit(1)

    print(f"Task ID: {task_id}")
    print(f"Checkpoint: {checkpoint_path}")

    configure_torch_backends()
    device = "cpu"

    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)

    env_cfg.seed = 42
    agent_cfg.seed = 42

    print("Creating environment...")
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env)

    agent_dict = asdict(agent_cfg)
    print("Creating runner...")
    runner_cls = load_runner_cls(task_id)
    runner = runner_cls(env, agent_dict, device=device)

    print("Loading checkpoint...")
    runner.load(checkpoint_path)
    print("Checkpoint loaded successfully.")

    # ---- Verify observation terms & permutation ----
    print()
    print("=" * 72)
    print("Observation term mapping (frame-major → term-major)")
    print("=" * 72)

    obs_manager = env.unwrapped.observation_manager
    actor = runner.alg.get_policy()

    # Build the same perm the runner uses.
    perm = build_history_interleave_perm(obs_manager, actor)

    offset = 0
    for group in actor.obs_groups:
        term_names = obs_manager.active_terms[group]
        term_dims = obs_manager.group_obs_term_dim[group]

        print(f"\nGroup: {group}")
        for name, dim in zip(term_names, term_dims):
            term_cfg = obs_manager.get_term_cfg(group, name)
            total = int(math.prod(dim))
            if term_cfg.history_length > 0 and term_cfg.flatten_history_dim:
                hist = term_cfg.history_length
            else:
                hist = 1
            base = total // hist
            print(f"  {name:24s}  dim={str(dim):12s}  total={total:4d}  "
                  f"base={base:3d}  hist={hist}")

        # If perm exists, verify the mapping for this group.
        if perm is not None:
            group_dim = sum(
                (int(math.prod(d)) // (
                    obs_manager.get_term_cfg(group, n).history_length
                    if obs_manager.get_term_cfg(group, n).history_length > 0
                    and obs_manager.get_term_cfg(group, n).flatten_history_dim
                    else 1
                ))
                * (obs_manager.get_term_cfg(group, n).history_length
                   if obs_manager.get_term_cfg(group, n).history_length > 0
                   and obs_manager.get_term_cfg(group, n).flatten_history_dim
                   else 1)
                for n, d in zip(term_names, term_dims)
            )
            perm_slice = perm[offset : offset + group_dim]
            identity = torch.arange(offset, offset + group_dim)
            if not torch.equal(perm_slice, identity):
                print(f"  [REORDERED] — perm[{offset}:{offset + group_dim}] ≠ identity")
            else:
                print(f"  [identity]")
        offset += group_dim

    if perm is not None:
        print(f"\nTotal perm length: {perm.numel()}")
        # Show a sample: first 2 terms × first 3 frames of the first group.
        first_group = actor.obs_groups[0]
        group_offset = 0  # actor group starts at offset 0
        bases = []
        for name, dim in zip(
            obs_manager.active_terms[first_group],
            obs_manager.group_obs_term_dim[first_group],
        ):
            term_cfg = obs_manager.get_term_cfg(first_group, name)
            total_elems = int(math.prod(dim))
            if term_cfg.history_length > 0 and term_cfg.flatten_history_dim:
                bases.append(total_elems // term_cfg.history_length)
            else:
                bases.append(total_elems)
        if bases:
            per_frame = sum(bases)
            print(f"per_frame (sum of bases) = {per_frame}")
            print(f"Sample: first 2 terms, first 3 frames of group '{first_group}':")
            for ti in range(min(2, len(bases))):
                term_off = sum(bases[:ti])
                name = obs_manager.active_terms[first_group][ti]
                print(f"  {name} (base={bases[ti]}, hist=5):")
                for f in range(3):
                    f_start = group_offset + term_off * 5 + f * bases[ti]
                    f_end = f_start + bases[ti]
                    vals = perm[f_start:f_end].tolist()
                    print(f"    frame {f}: perm[{f_start:>3}:{f_end:<3}] = {vals}")
    print("=" * 72)
    print()

    # Export ONNX to the same directory as the checkpoint.
    export_dir = str(Path(checkpoint_path).parent)
    print(f"Exporting ONNX to: {export_dir}")
    runner.export_policy_to_onnx(export_dir, filename="policy.onnx", verbose=False)

    onnx_path = os.path.join(export_dir, "policy.onnx")
    file_size = os.path.getsize(onnx_path)
    print(f"Done! ONNX exported to: {onnx_path} ({file_size} bytes)")

    # Also attach metadata.
    from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
    metadata = get_base_metadata(env.unwrapped, exp_name)
    attach_metadata_to_onnx(onnx_path, metadata)
    print("Metadata attached.")

    env.close()


if __name__ == "__main__":
    main()
