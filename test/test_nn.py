import pytest
import flax.nnx as nnx
import optax
import jax
import jax.numpy as jnp
from alci.nn import MLP, train_step

def test_nn_init():
    rngs = nnx.Rngs(params=jax.random.PRNGKey(42))
    mlp = MLP([2, 4, 1], rngs)
    assert len(mlp.layers) == 2
    assert mlp.layers[0].in_features == 2
    assert mlp.layers[0].out_features == 4
    assert mlp.layers[1].in_features == 4
    assert mlp.layers[1].out_features == 1

def test_nn_call():
    rng = jax.random.PRNGKey(42)
    mlp = MLP([2, 4, 1], nnx.Rngs(params=rng))
    batched_input = jax.random.normal(rng, (100, 2))
    batched_output = mlp(batched_input)
    assert batched_output.shape == (100, 1)

def test_training():
    rng = jax.random.PRNGKey(42)
    mlp = MLP([2, 4, 1], nnx.Rngs(params=rng))
    xs = jax.random.normal(rng, (100, 2))
    ys = jax.random.normal(rng, (100, 1))
    optimizer = nnx.Optimizer(mlp, optax.adam(1e-3), wrt=nnx.Param)
    loss1 = train_step(mlp, optimizer, xs, ys)
    loss2 = train_step(mlp, optimizer, xs, ys)
    assert loss1.shape == ()
    assert loss2 < loss1
