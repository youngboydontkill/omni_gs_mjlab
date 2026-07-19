"""Compute AMP observation ``s_t`` from the live MuJoCo simulation state.

The AMP state defined in Hiking in the Wild §III-E eq. 6 is::

    s_t = [v_t(3), omega_t(3), g_t(3), q_t(N), qdot_t(N)]

where *N* is the number of controlled joints (26 for Kuavo S45) and all
quantities are read in the robot body frame from ``EntityData``.
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv


_STATE_DIM = 3 + 3 + 3 + 26 + 26  # 61


class AMPStateComputer:
    """Reads the MuJoCo entity data every step to produce ``s_t``.

    Parameters
    ----------
    env:
        The **unwrapped** ``ManagerBasedRlEnv`` — i.e. ``env.unwrapped`` from
        the wrapper.  Provides ``env.scene["robot"].data.*`` at each step.
    joint_ids:
        1-D ``LongTensor`` of length 26 indexing into ``data.joint_pos`` and
        ``data.joint_vel`` in ``preserve_order`` (legs first, then arms).
    """

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        joint_ids: torch.Tensor,
    ) -> None:
        self._env = env
        self._joint_ids = joint_ids.to(env.device)

    def compute(self) -> torch.Tensor:
        """Return the AMP state for all parallel environments.

        Returns:
            ``[B, 61]`` tensor on the env device.
        """
        data = self._env.scene["robot"].data
        v = data.root_link_lin_vel_b                           # [B, 3]
        omega = data.root_link_ang_vel_b                        # [B, 3]
        g = data.projected_gravity_b                            # [B, 3]
        q = data.joint_pos[:, self._joint_ids]                  # [B, 26]
        qdot = data.joint_vel[:, self._joint_ids]              # [B, 26]
        return torch.cat([v, omega, g, q, qdot], dim=-1)
