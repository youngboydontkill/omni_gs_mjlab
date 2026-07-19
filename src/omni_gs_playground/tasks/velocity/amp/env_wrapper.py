"""VecEnv wrapper that injects ``extras["amp_obs"]`` after every step.

The wrapper maintains a per-environment sliding history of recent AMP states
and concatenates them into a flat sequence expected by the discriminator.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from mjlab.rl import RslRlVecEnvWrapper

from .amp_obs import AMPStateComputer, _STATE_DIM


class AMPVecEnvWrapper(RslRlVecEnvWrapper):
    """Extends ``RslRlVecEnvWrapper`` to compute AMP observations on the fly.

    On each ``step()`` the current AMP state ``s_t`` is appended to a
    per-environment FIFO buffer of length ``seq_len``.  The buffer is
    flattened and placed into ``extras["amp_obs"]`` (``[B, seq_len *
    STATE_DIM]``) so that ``PPO.process_env_step`` can pick it up.

    When an environment resets (``dones == 1``), its entire history is
    overwritten with the post-reset frame so the discriminator does not
    receive a stale pre-reset / post-reset mixture.

    Parameters
    ----------
    env:
        Underlying ``ManagerBasedRlEnv``.
    state_computer:
        Callable ``() -> [B, STATE_DIM]`` that reads current MuJoCo state.
    seq_len:
        Number of frames in each AMP observation sequence.
    clip_actions:
        Forwarded to ``RslRlVecEnvWrapper``.
    """

    def __init__(
        self,
        env,
        state_computer: AMPStateComputer,
        seq_len: int = 4,
        clip_actions: float | None = None,
    ) -> None:
        super().__init__(env, clip_actions=clip_actions)
        self._state_computer = state_computer
        self._seq_len = seq_len
        self._state_dim = _STATE_DIM

        # Per-env history: [B, seq_len, STATE_DIM], initialised with the
        # first frame.
        cur = self._state_computer.compute().detach()
        self._amp_history = cur.unsqueeze(1).expand(-1, seq_len, -1).clone()

    def step(
        self, actions: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        obs, rew, dones, extras = super().step(actions)

        # ---- update per-env history ---------------------------------------
        cur = self._state_computer.compute().detach()

        # Shift left, insert cur at the rightmost position.
        self._amp_history = torch.roll(self._amp_history, -1, dims=1)
        self._amp_history[:, -1, :] = cur

        # For environments that just reset (dones == 1), overwrite the whole
        # history with the current frame so the discriminator sees a clean
        # post-reset sequence.
        done_mask = dones.bool().view(-1, 1, 1)
        self._amp_history = torch.where(
            done_mask,
            cur.unsqueeze(1).expand(-1, self._seq_len, -1),
            self._amp_history,
        )

        # ---- inject into extras -------------------------------------------
        extras["amp_obs"] = self._amp_history.flatten(1)

        return obs, rew, dones, extras

    def get_observations(self) -> TensorDict:
        """Return observations and re-initialise the history buffer.

        Called by the runner at the start of learning (after
        ``env.reset()`` in ``__init__``), so the history is always fresh.
        """
        result = super().get_observations()
        cur = self._state_computer.compute().detach()
        self._amp_history = cur.unsqueeze(1).expand(-1, self._seq_len, -1).clone()
        return result
