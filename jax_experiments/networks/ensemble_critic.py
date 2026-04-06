"""Vectorized ensemble Q-network — single batched matmul for all K critics.

Key optimization: instead of K independent SingleCritic modules with a Python
for-loop + jnp.stack, all K critics share a [K, in, out] weight tensor and
forward is a single batched matmul: x @ W + b where x is [K, B, in].

This reduces the XLA graph size by ~K× and lets the GPU execute all critics
in one fused kernel instead of K sequential ones.
"""
import jax
import jax.numpy as jnp
from flax import nnx


class VectorizedLinear(nnx.Module):
    """K parallel linear layers as one batched matmul.

    kernel: [K, in_features, out_features]
    bias:   [K, 1, out_features]
    forward: [K, B, in] @ [K, in, out] + [K, 1, out] → [K, B, out]
    """

    def __init__(self, in_features: int, out_features: int,
                 ensemble_size: int, *, rngs: nnx.Rngs):
        # lecun_normal: truncated_normal(stddev=1/sqrt(fan_in)), truncated at ±2σ
        stddev = 1.0 / jnp.sqrt(float(in_features))
        self.kernel = nnx.Param(
            jax.random.truncated_normal(
                rngs.params(), -2.0, 2.0,
                (ensemble_size, in_features, out_features)) * stddev)
        self.bias = nnx.Param(
            jnp.zeros((ensemble_size, 1, out_features)))

    def __call__(self, x):
        """x: [K, B, in] → [K, B, out]"""
        return x @ self.kernel.value + self.bias.value


class EnsembleCritic(nnx.Module):
    """Vectorized ensemble of K Q-networks.

    All K critics are stored as stacked [K, in, out] weight tensors.
    Forward pass is a single chain of batched matmuls — no Python loop over K.

    API is identical to the old per-critic version:
      __call__(obs, act) → [K, batch]
      compute_reg_norm() → [K]
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256,
                 ensemble_size: int = 10, n_layers: int = 3, *, rngs: nnx.Rngs):
        self.ensemble_size = ensemble_size
        self.n_hidden = n_layers

        layers = []
        in_d = obs_dim + act_dim
        for _ in range(n_layers):
            layers.append(VectorizedLinear(in_d, hidden_dim, ensemble_size,
                                          rngs=rngs))
            in_d = hidden_dim
        # Output layer: → 1
        layers.append(VectorizedLinear(in_d, 1, ensemble_size, rngs=rngs))
        self.layers = nnx.List(layers)

    def __call__(self, obs, act):
        """Returns [ensemble_size, batch] Q-values."""
        x = jnp.concatenate([obs, act], axis=-1)          # [B, in_dim]
        # Broadcast to [K, B, in_dim] — zero-copy on GPU
        x = jnp.broadcast_to(x[None], (self.ensemble_size,) + x.shape)
        for i, layer in enumerate(self.layers):
            x = layer(x)                                   # [K, B, out]
            if i < self.n_hidden:
                x = jax.nn.relu(x)
        return x.squeeze(-1)                               # [K, B]

    def compute_reg_norm(self):
        """L1 regularization norm per critic head. Returns [ensemble_size].

        Vectorized: one jnp.abs + jnp.sum per layer (not per critic).
        """
        total = jnp.zeros(self.ensemble_size)
        for layer in self.layers:
            # kernel: [K, in, out] → sum over (in, out) → [K]
            total = total + jnp.sum(jnp.abs(layer.kernel.value), axis=(1, 2))
            # bias: [K, 1, out] → sum over (1, out) → [K]
            total = total + jnp.sum(jnp.abs(layer.bias.value), axis=(1, 2))
        return total
