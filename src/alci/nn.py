import jax
import jax.numpy as jnp
import flax.nnx as nnx

class MLP(nnx.Module):
    def __init__(self, dims: list[int], rngs: nnx.Rngs, activation: callable=nnx.relu):
        self.layers = nnx.List()
        self.activation = activation
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layer = nnx.Linear(in_dim, out_dim, rngs=rngs)
            self.layers.append(layer)
    
    def __call__(self, x: jax.Array) -> jax.Array:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x

@nnx.jit
def train_step(model: MLP, optimizer, x: jax.Array, y: jax.Array):
    def error_loss(model):
        y_pred = model(x)
        return jnp.mean((y - y_pred) ** 2)
    loss, grad = nnx.value_and_grad(error_loss)(model)
    optimizer.update(model, grad)
    return loss
