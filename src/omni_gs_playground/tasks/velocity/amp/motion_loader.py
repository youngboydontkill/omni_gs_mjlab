"""Load retargeted Kuavo S45 motion data and build AMP reference sequences.

CSV format (35 columns, comma-separated)::

    root_pos_x, root_pos_y, root_pos_z,
    root_rot_x, root_rot_y, root_rot_z, root_rot_w,   # xyzw quaternion
    dof_pos[0..27]                                     # 28 joint angles

The last two dof_pos columns (indices 33–34, head joints) are discarded;
S45 only controls 26 joints (legs + arms).

Each frame is converted to the AMP observation state defined in the paper
(Hiking in the Wild §III-E, eq. 6)::

    s_t = [v_t(3), omega_t(3), g_t(3), q_t(26), qdot_t(26)]   -> dim 61

and then sliding-window sequences are built for the discriminator.
"""

from __future__ import annotations

import glob
import os

import torch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STATE_DIM = 3 + 3 + 3 + 26 + 26  # v + omega + g + q + qdot = 61
"""Per-frame AMP state dimension."""

# Columns in the CSV (0-indexed).
_COL_ROOT_POS = slice(0, 3)       # x, y, z
_COL_ROOT_ROT = slice(3, 7)       # x, y, z, w  (xyzw)
_COL_DOF_POS = slice(7, 33)       # first 26 of 28 dof_pos (drop head joints)


# ---------------------------------------------------------------------------
# Quaternion helpers (operate on xyzw convention as stored in CSV)
# ---------------------------------------------------------------------------

def _quat_xyzw_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Convert xyzw unit quaternions to 3×3 rotation matrices.

    Args:
        q: ``[..., 4]`` quaternion in xyzw order.

    Returns:
        ``[..., 3, 3]`` rotation matrix R such that ``v_world = R @ v_body``.
    """
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    r00 = 1.0 - 2.0 * (yy + zz)
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)
    r10 = 2.0 * (xy + wz)
    r11 = 1.0 - 2.0 * (xx + zz)
    r12 = 2.0 * (yz - wx)
    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (xx + yy)

    R = torch.stack([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=-1)
    return R.view(*q.shape[:-1], 3, 3)


def _quat_xyzw_inv(q: torch.Tensor) -> torch.Tensor:
    """Inverse (conjugate) of unit xyzw quaternions."""
    inv = q.clone()
    inv[..., :3] = -inv[..., :3]
    return inv


def _quat_xyzw_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two xyzw quaternions: q1 * q2."""
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([x, y, z, w], dim=-1)


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def _csv_to_states(path: str, dt: float = 1.0 / 30.0) -> torch.Tensor:
    """Load one CSV clip and convert every frame to an AMP state.

    Returns:
        ``[T, 61]`` tensor.  The first frame has ``v`` and ``omega`` set to
        zero (they require a backward difference).  The last frame is included
        so that sliding-window builds complete sequences; its ``v``/``omega``
        duplicate the penultimate frame.
    """
    import numpy as np

    data = torch.from_numpy(
        np.loadtxt(path, delimiter=",", dtype="float64")
    ).float()

    # Drop rows with NaN (sparse; mostly the very last frame on some clips).
    valid = ~torch.isnan(data).any(dim=-1)
    data = data[valid]

    T = data.shape[0]
    if T < 2:
        return torch.empty(0, _STATE_DIM)

    root_pos = data[:, _COL_ROOT_POS]                    # [T, 3]
    root_rot_xyzw = data[:, _COL_ROOT_ROT]               # [T, 4]
    dof_pos = data[:, _COL_DOF_POS]                      # [T, 26]

    # --- projected gravity: R^T @ [0, 0, -1] ---
    R = _quat_xyzw_to_rotmat(root_rot_xyzw)              # [T, 3, 3]
    gravity_world = torch.tensor([0.0, 0.0, -1.0])
    g = torch.einsum("tij,j->ti", R, gravity_world)      # [T, 3]

    # --- body-frame linear velocity: R^T @ (dp / dt) ---
    dp = root_pos[1:] - root_pos[:-1]                    # [T-1, 3]
    v_world = dp / dt
    R_mid = R[:-1]
    v = torch.einsum("tij,tj->ti", R_mid, v_world)       # [T-1, 3]
    v = torch.cat([v, v[-1:]], dim=0)                    # [T, 3] pad last
    v = torch.cat([torch.zeros(1, 3), v[1:]], dim=0)     # [T, 3] pad first

    # --- body-frame angular velocity from quaternion difference ---
    q_xyzw = root_rot_xyzw                                # [T, 4]
    q_inv = _quat_xyzw_inv(q_xyzw[:-1])                   # [T-1, 4]
    dq = _quat_xyzw_mul(q_inv, q_xyzw[1:])                # [T-1, 4]
    dq_vec = dq[..., :3]
    dq_w = dq[..., 3:4]
    omega = (2.0 / dt) * dq_vec * torch.sign(dq_w)       # [T-1, 3]
    omega = torch.cat([omega, omega[-1:]], dim=0)         # [T, 3]
    omega = torch.cat([torch.zeros(1, 3), omega[1:]], dim=0)  # [T, 3]

    # --- joint velocities by finite difference ---
    dq_dof = dof_pos[1:] - dof_pos[:-1]                  # [T-1, 26]
    qdot = dq_dof / dt
    qdot = torch.cat([qdot, qdot[-1:]], dim=0)           # [T, 26]
    qdot = torch.cat([torch.zeros(1, 26), qdot[1:]], dim=0)  # [T, 26]

    # --- assemble s_t ---
    states = torch.cat([v, omega, g, dof_pos, qdot], dim=-1)
    return states


# ---------------------------------------------------------------------------
# MotionDataset
# ---------------------------------------------------------------------------

class MotionDataset:
    """All AMP reference sequences obtained by sliding a window over CSV clips.

    The dataset is built eagerly at construction time; typical size is well
    under 100 MB (135 clips × ~150 frames/clip × 244 floats × 4 bytes ≈ 19 MB
    for seq_len=4).
    """

    def __init__(
        self,
        csv_dir: str,
        seq_len: int = 4,
        dt: float = 1.0 / 30.0,
    ) -> None:
        self.seq_len = seq_len
        self.state_dim = _STATE_DIM
        self.input_dim = seq_len * _STATE_DIM

        all_seqs: list[torch.Tensor] = []
        pattern = os.path.join(csv_dir, "*.csv")
        for path in sorted(glob.glob(pattern)):
            states = _csv_to_states(path, dt=dt)
            if states.shape[0] < seq_len:
                continue
            # Sliding window with stride 2 for slightly more diversity.
            windows = states.unfold(0, seq_len, 2)        # [num_win, 61, seq_len]
            if windows.shape[0] == 0:
                continue
            windows = windows.permute(0, 2, 1)            # [num_win, seq_len, 61]
            windows = windows.reshape(windows.shape[0], self.input_dim)
            all_seqs.append(windows)

        if not all_seqs:
            raise FileNotFoundError(
                f"No valid motion clips (≥{seq_len} frames) found in {csv_dir}"
            )

        self._sequences = torch.cat(all_seqs, dim=0)
        self._num_sequences = self._sequences.shape[0]

    def sample(self, n: int) -> torch.Tensor:
        """Uniform random sample of *n* reference sequences.

        Returns:
            ``[n, input_dim]`` tensor sampled with replacement.
        """
        idx = torch.randint(0, self._num_sequences, (n,))
        return self._sequences[idx].clone()

    def __len__(self) -> int:
        return self._num_sequences


# ---------------------------------------------------------------------------
# AMPSampler – callable interface for PPO.set_amp_reference_sampler
# ---------------------------------------------------------------------------

class AMPSampler:
    """Callable reference-motion sampler matching the PPO hook signature.

    ``PPO.set_amp_reference_sampler(sampler)`` expects ``sampler(n)`` to
    return a ``[n, seq_dim]`` batch of reference state sequences.
    """

    def __init__(self, dataset: MotionDataset) -> None:
        self._dataset = dataset

    def __call__(self, n: int) -> torch.Tensor:
        return self._dataset.sample(n)
