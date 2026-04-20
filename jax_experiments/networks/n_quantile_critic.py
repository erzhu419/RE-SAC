"""N-Quantile Critic for TQC — N independent IQN critics, Flax NNX."""
import jax
import jax.numpy as jnp
from flax import nnx

from jax_experiments.networks.quantile_critic import QuantileCritic


class NQuantileCritic(nnx.Module):
    """N independent IQN quantile critics Z_1..Z_N(s, a, tau).

    Each critic outputs (batch, T) quantile values.
    Forward returns (N, batch, T).
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256,
                 n_critics: int = 5, n_layers: int = 2, embedding_size: int = 32,
                 num_quantiles: int = 25, *, rngs: nnx.Rngs):
        self.n_critics = n_critics
        self.num_quantiles = num_quantiles
        self.critics = nnx.List([
            QuantileCritic(obs_dim, act_dim, hidden_dim, n_layers,
                           embedding_size, num_quantiles,
                           rngs=nnx.Rngs(params=rngs.params()))
            for _ in range(n_critics)
        ])

    def __call__(self, obs, act, tau):
        """Returns (N, batch, T) quantile values."""
        return jnp.stack([c(obs, act, tau) for c in self.critics], axis=0)
