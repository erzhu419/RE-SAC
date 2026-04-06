"""IQN Quantile Critic for DSAC — Flax NNX (optimized).

Optimizations vs v1:
  - No LayerNorm (expensive in lax.scan context)
  - Smaller default embedding_size (32 vs 64)
  - Plain ReLU MLP, matching SAC critic efficiency
"""
import jax
import jax.numpy as jnp
from flax import nnx


class QuantileCritic(nnx.Module):
    """IQN-style quantile value network Z(s, a, τ).

    Architecture (simplified for scan efficiency):
      base_fc:  (s,a) → h   via MLP with ReLU (no LayerNorm)
      tau_fc:   cos(τ·π·[1..E]) → sigmoid embedding
      merge_fc: h ⊙ τ_embed → hidden → scalar per quantile
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256,
                 n_layers: int = 2, embedding_size: int = 32,
                 num_quantiles: int = 8, *, rngs: nnx.Rngs):
        self.num_quantiles = num_quantiles
        self.embedding_size = embedding_size

        # Base MLP: (s,a) → hidden representation (no LayerNorm)
        input_dim = obs_dim + act_dim
        base_layers = []
        in_d = input_dim
        for _ in range(n_layers):
            base_layers.append(nnx.Linear(in_d, hidden_dim, rngs=rngs))
            in_d = hidden_dim
        self.base_layers = nnx.List(base_layers)
        self.n_base = n_layers

        # Tau embedding: cos features → sigmoid (no LayerNorm)
        self.tau_linear = nnx.Linear(embedding_size, hidden_dim, rngs=rngs)

        # Merge → output (no LayerNorm)
        self.merge_linear = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.last_linear = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(self, obs, act, tau):
        """
        Args:
            obs: (N, obs_dim)
            act: (N, act_dim)
            tau: (N, T)  — quantile fractions in [0, 1]
        Returns:
            (N, T) quantile values
        """
        # Base features
        h = jnp.concatenate([obs, act], axis=-1)
        for i in range(self.n_base):
            h = self.base_layers[i](h)
            h = nnx.relu(h)
        # h: (N, hidden_dim)

        # Tau embedding: cos(τ·π·[1..E]) — const_vec computed inline
        const_vec = jnp.arange(1, 1 + self.embedding_size, dtype=jnp.float32)
        cos_features = jnp.cos(
            tau[..., None] * const_vec[None, None, :] * jnp.pi
        )  # (N, T, E)
        tau_embed = nnx.sigmoid(self.tau_linear(cos_features))  # (N, T, hidden_dim)

        # Merge: h ⊙ tau_embed
        merged = h[:, None, :] * tau_embed  # (N, T, hidden_dim)
        merged = nnx.relu(self.merge_linear(merged))

        # Output: scalar per quantile
        return self.last_linear(merged).squeeze(-1)  # (N, T)


class TwinQuantileCritic(nnx.Module):
    """Twin IQN critics (Z1, Z2) for DSAC."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256,
                 n_layers: int = 2, embedding_size: int = 32,
                 num_quantiles: int = 8, *, rngs: nnx.Rngs):
        self.zf1 = QuantileCritic(
            obs_dim, act_dim, hidden_dim, n_layers, embedding_size,
            num_quantiles, rngs=nnx.Rngs(params=rngs.params()))
        self.zf2 = QuantileCritic(
            obs_dim, act_dim, hidden_dim, n_layers, embedding_size,
            num_quantiles, rngs=nnx.Rngs(params=rngs.params()))
        self.num_quantiles = num_quantiles
