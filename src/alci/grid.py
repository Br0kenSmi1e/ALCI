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

class QuanticsGrid:
    def __init__(self, base: int, nbit: int, dim: int, start: jax.Array, end: jax.Array):
        base
        self.base = base
        self.nbit = nbit
        self.ngrids = jnp.array([base] * (nbit * dim))
        self.dim = dim
        self.start = start
        self.end = end
    
    def grid2uniform(self, grid_point: jax.Array) -> jax.Array:
        weight = jnp.array([[n for n in range(self.nbit)] for d in range(self.dim)])
        bits = jnp.array([[grid_point[self.dim*n+d] for n in range(self.nbit)] for d in range(self.dim)])
        return jnp.sum(bits * (self.base ** weight), axis=1)
    
    def uniform2grid(self, u_point: jax.Array) -> jax.Array:
        def int2quantics(n, b, w):
            digits = []
            for _ in range(w):
                digits.append(n % b)
                n //= b
            return digits
        weight = [int2quantics(n, self.base, self.nbit) for n in u_point]
        return jnp.array([weight[d][n] for n in range(self.nbit) for d in range(self.dim)])
    
    def grid2space(self, grid_point: jax.Array) -> jax.Array:
        return self.start + (self.end - self.start) * self.grid2uniform(grid_point) / (self.base ** self.nbit - 1)
    
    def all_grid_pts(self) -> jax.Array:
        return jnp.stack(jnp.meshgrid(*[jnp.arange(self.base) for _ in range(self.nbit * self.dim)], indexing='ij'), axis=-1).reshape(-1, self.nbit * self.dim)
    
    def grid_and_space(self):
        grid_pts = self.all_grid_pts()
        space_pts = jax.vmap(self.grid2space)(grid_pts)
        return grid_pts, space_pts
