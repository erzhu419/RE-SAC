"""Deterministic Policy (Tanh MLP) for TD3."""
import jax
import jax.numpy as jnp
from flax import nnx


class DeterministicPolicy(nnx.Module):
    """TD3-style deterministic policy: obs → tanh(MLP(obs))."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256,
                 n_layers: int = 2, *, rngs: nnx.Rngs):
        layers = []
        in_d = obs_dim
        for _ in range(n_layers):
            layers.append(nnx.Linear(in_d, hidden_dim, rngs=rngs))
            in_d = hidden_dim
        self.layers = nnx.List(layers)
        self.n_hidden = n_layers
        self.head = nnx.Linear(hidden_dim, act_dim, rngs=rngs)
        self.act_dim = act_dim

    def __call__(self, obs):
        """Returns deterministic action ∈ [-1, 1]."""
        x = obs
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.n_hidden:
                x = nnx.relu(x)
        return jnp.tanh(self.head(x))

    def noisy_action(self, obs, key, noise_std=0.1):
        """Add Gaussian exploration noise, clipped to [-1, 1]."""
        action = self(obs)
        noise = jax.random.normal(key, action.shape) * noise_std
        return jnp.clip(action + noise, -1.0, 1.0)

    # Alias for compatibility with evaluate() which calls .deterministic()
    def deterministic(self, obs):
        return self(obs)

    # Dummy sample() for compatibility with rollout code
    def sample(self, obs, key):
        """For API compat: returns (action, dummy_log_prob=0)."""
        action = self.noisy_action(obs, key, noise_std=0.1)
        log_prob = jnp.zeros(obs.shape[0])
        return action, log_prob
