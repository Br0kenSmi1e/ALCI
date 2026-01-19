import jax
import jax.numpy as jnp

class DataSet:
    def __init__(self, grid):
        self.grid = grid
        self.points = jnp.zeros((0, grid.dim), dtype=int)
        self.vals = jnp.zeros((0, ), dtype=float)
        self.pivot_pos = jnp.zeros((grid.dim, 0), dtype=int)
    
    def isnewpivot(self, point: jax.Array):
        return ~jnp.any(jax.vmap(jnp.isin, (0, 0), 0)(point, self.pivot_pos))
    
    def add_pivot(self, pivot: jax.Array):
        return jax.vmap(jnp.append, (0, 0), 0)(self.pivot_pos, pivot)
    
    def update_pivot(self, pivot: jax.Array):
        if self.isnewpivot(pivot):
            self.pivot_pos = self.add_pivot(pivot)
    
    def new_points(self, pivot: jax.Array):
        new_points = [pivot.at[i].set(j) \
            for i in range(self.grid.dim) \
            for j in range(self.grid.num[i]) \
            if not j in self.pivot_pos]
        new_points.append(pivot)
        return jnp.array(new_points)
    
    def query(self, f: callable, new_points: jax.Array):
        space_points = jax.vmap(self.grid.grid2space)(new_points)
        vals = jax.vmap(f)(space_points)
        return vals
    
    def add_data(self, points: jax.Array, vals: jax.Array):
        self.points = jnp.append(self.points, points)
        self.vals = jnp.append(self.vals, vals)
