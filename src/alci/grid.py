import jax
import jax.numpy as jnp

class UniformGrid:
    def __init__(self, dim: int, start: jax.Array, end: jax.Array, num: jax.Array):
        self.dim = dim
        self.start = start
        self.end = end
        self.num = num
    
    def grid2space(self, grid_point: jax.Array) -> jax.Array:
        return self.start + (self.end - self.start) * grid_point / (self.num - 1)
