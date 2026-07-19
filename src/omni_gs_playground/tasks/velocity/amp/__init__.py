"""AMP (Adversarial Motion Priors) style-reward training support.

See `Hiking in the Wild <https://arxiv.org/abs/2506.12345>`_ §III-E and
``doc/amp_scaffold.md`` for the algorithm; ``doc/amp_scaffold.md`` also
documents the discriminator architecture and PPO integration hooks.
"""

from .amp_obs import AMPStateComputer
from .env_wrapper import AMPVecEnvWrapper
from .motion_loader import AMPSampler, MotionDataset

__all__ = [
    "AMPSampler",
    "AMPStateComputer",
    "AMPVecEnvWrapper",
    "MotionDataset",
]
