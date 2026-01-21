import jax
import jax.numpy as jnp

class UniformGrid:
    def __init__(self, length: int, ngrids: jax.Array, dim: int, start: jax.Array, end: jax.Array):
        self.length = length
        self.ngrids = ngrids
        self.dim = dim
        self.start = start
        self.end = end
    
    def grid2space(self, grid_point: jax.Array) -> jax.Array:
        return self.start + (self.end - self.start) * grid_point / (self.ngrids - 1)
    
    def all_grid_pts(self) -> jax.Array:
        return jnp.stack(jnp.meshgrid(*[jnp.arange(n) for n in self.ngrids], indexing='ij'), axis=-1).reshape(-1, self.length)
    
    def grid_and_space(self):
        grid_pts = self.all_grid_pts()
        space_pts = jax.vmap(self.grid2space)(grid_pts)
        return grid_pts, space_pts
