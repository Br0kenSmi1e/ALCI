import jax
import jax.numpy as jnp

class PivotList():
    def __init__(self, length: int, ngrids: jax.Array):
        self.length = length
        self.ngrids = ngrids
        self.row_list = [[] for _ in range(length - 1)] # list[list[jax.Array]]
        self.col_list = [[] for _ in range(length - 1)] # list[list[jax.Array]]
    
    def isnewpivot(self, pivot: jax.Array):
        def isin(idx: jax.Array, idx_set: list[jax.Array]):
            for i in idx_set:
                if jnp.all(i == idx):
                    return True
                else:
                    continue
            return False
        
        isnewrow = [not isin(pivot[:l], self.row_list[l-1]) for l in range(1, self.length)]
        isnewcol = [not isin(pivot[l:], self.col_list[l-1]) for l in range(1, self.length)]
        return all(isnewrow) and all(isnewcol)

    def _add_pivot(self, pivot: jax.Array):
        def pushunique(idx: jax.Array, idx_set: list[jax.Array]):
            for i in idx_set:
                if jnp.all(i == idx):
                    return None
                else:
                    continue
            idx_set.append(idx)
            return None
        for l in range(1, self.length):
            pushunique(pivot[:l], self.row_list[l-1])
            pushunique(pivot[l:], self.col_list[l-1])
        return None
    
    def add_pivot(self, pivot: jax.Array):
        # if self.isnewpivot(pivot):
        #     self._add_pivot(pivot)
        self._add_pivot(pivot)
    
    def dim0slice(self):
        return [jnp.array([[jnp.hstack([r, c]) for c in cs] for r in rs]) \
                for rs, cs in zip(self.row_list, self.col_list)]
    
    def dim1slice(self):
        _row_list = [[jnp.array([], dtype=int)], *self.row_list]
        _col_list = [*self.col_list, [jnp.array([], dtype=int)]]
        dim1idx = [jnp.array([[[jnp.hstack([r, jnp.array([s]), c]) \
                for c in _col_list[l]] \
                for s in range(self.ngrids[l])] \
                for r in _row_list[l]]) \
            for l in range(self.length)]
        return dim1idx
    
    def data_points(self):
        dim1idx = self.dim1slice()
        return jnp.unique(jnp.vstack([jnp.reshape(i, (-1, self.length)) for i in dim1idx]), axis=0)
    
class TCI():
    def __init__(self, f: callable, g, pl: PivotList):
        self.pivot_list = pl
        self.func = f
        self.grid = g
        self.slices0 = []
        self.slices1 = [] # list[jax.Array]
        self.sitetensors = [] # list[jax.Array]

    def update_slices(self):
        f = lambda i: self.func(self.grid.grid2space(i))
        dim1idx = self.pivot_list.dim1slice()
        self.slices1 = [jax.vmap(jax.vmap(jax.vmap(f)))(i) for i in dim1idx]
        dim0idx = self.pivot_list.dim0slice()
        self.slices0 = [jax.vmap(jax.vmap(f))(i) for i in dim0idx]
    
    def update_sitetensors(self, tol=1e-8):

        # @jax.jit
        def single_site_contract(t: jax.Array, p: jax.Array, tol: float):
            mat_t = jnp.reshape(t, (-1, t.shape[2]))
            # new_t = jnp.linalg.solve(p.T, mat_t.T).T
            u, s, vt = jnp.linalg.svd(p, full_matrices=False)
            rank = jnp.sum(s / s[0] > tol)
            new_t = mat_t @ vt[:rank, :].T @ jnp.diag(1 / s[:rank]) @ u[:, :rank].T
            return jnp.reshape(new_t, (t.shape[0], t.shape[1], -1))
        
        self.sitetensors = [single_site_contract(t, p, tol) for t, p in zip(self.slices1[:-1], self.slices0)]
        self.sitetensors.append(self.slices1[-1])

    def __call__(self, gp: jax.Array) -> jax.Array:
        mats = [t[:, l, :] for l, t in zip(gp, self.sitetensors)]
        return jnp.linalg.multi_dot(mats)[0, 0]
