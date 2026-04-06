"""MLP building blocks using Flax NNX."""
import jax
import jax.numpy as jnp
from flax import nnx


class MLP(nnx.Module):
    """Simple feedforward MLP with ReLU activations."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 n_layers: int = 2, *, rngs: nnx.Rngs):
        layers = []
        in_d = input_dim
        for _ in range(n_layers):
            layers.append(nnx.Linear(in_d, hidden_dim, rngs=rngs))
            in_d = hidden_dim
        layers.append(nnx.Linear(in_d, output_dim, rngs=rngs))
        self.layers = nnx.List(layers)
        self.n_hidden = n_layers

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.n_hidden:
                x = nnx.relu(x)
        return x
