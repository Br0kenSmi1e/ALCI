import pytest
import jax
import jax.numpy as jnp
from alci.tci import PivotList, TCI
from alci.grid import UniformGrid

def test_pivot_list_construct():
    pl = PivotList(2, jnp.array([10, 10]))
    assert pl.length == 2
    assert jnp.all(pl.ngrids == jnp.array([10, 10]))
    assert len(pl.row_list) == 1
    assert len(pl.col_list) == 1

def test_add_pivot():
    pl = PivotList(2, jnp.array([10, 10]))
    pl.add_pivot(jnp.array([0, 0]))
    assert [len(r) for r in pl.row_list] == [1]
    assert [len(l) for l in pl.col_list] == [1]

    pl.add_pivot(jnp.array([5, 5]))
    assert [len(r) for r in pl.row_list] == [2]
    assert [len(l) for l in pl.col_list] == [2]

def test_dim0slice():
    pl = PivotList(2, jnp.array([10, 10]))
    pl.add_pivot(jnp.array([0, 0]))
    pl.add_pivot(jnp.array([5, 5]))
    idx = pl.dim0slice()
    assert len(idx) == 1
    assert idx[0].shape == (2, 2, 2)

def test_dim1slice():
    pl = PivotList(2, jnp.array([10, 10]))
    pl.add_pivot(jnp.array([0, 0]))
    pl.add_pivot(jnp.array([5, 5]))
    idx = pl.dim1slice()
    assert len(idx) == 2
    for i, r, c in zip(idx, [1, 2], [2, 1]):
        assert i.shape == (r, 10, c, 2)

def test_data_points():
    pl = PivotList(2, jnp.array([10, 10]))
    pl.add_pivot(jnp.array([0, 0]))
    pts = pl.data_points()
    assert pts.shape == (19, 2)

    pl.add_pivot(jnp.array([5, 5]))
    pts = pl.data_points()
    assert pts.shape == (36, 2)

def test_tci_construct():
    f = lambda x: jnp.linalg.norm(x) ** 2
    g = UniformGrid(2, jnp.array([10, 10]), 2, jnp.array([-1.0, -1.0]), jnp.array([1.0, 1.0]))
    pl = PivotList(2, jnp.array([10, 10]))
    tci = TCI(f, g, pl)
    assert len(tci.sitetensors) == 0

def test_tci_update_slice():
    f = lambda x: jnp.linalg.norm(x) ** 2
    g = UniformGrid(2, jnp.array([10, 10]), 2, jnp.array([-1.0, -1.0]), jnp.array([1.0, 1.0]))
    pl = PivotList(2, jnp.array([10, 10]))
    tci = TCI(f, g, pl)
    tci.pivot_list.add_pivot(jnp.array([5, 5]))
    tci.update_slices()
    idx = tci.pivot_list.dim1slice()
    for i, t in zip(idx, tci.slices1):
        for a in range(t.shape[0]):
            for b in range(t.shape[1]):
                for c in range(t.shape[2]):
                    assert jnp.allclose(t[a, b, c], f(g.grid2space(i[a, b, c])))
                    assert len(t.shape) == 3

def test_tci_update_site():
    f = lambda x: jnp.linalg.norm(x) ** 2
    g = UniformGrid(2, jnp.array([10, 10]), 2, jnp.array([-1.0, -1.0]), jnp.array([1.0, 1.0]))
    pl = PivotList(2, jnp.array([10, 10]))
    tci = TCI(f, g, pl)
    tci.pivot_list.add_pivot(jnp.array([5, 5]))
    tci.pivot_list.add_pivot(jnp.array([3, 3]))
    tci.update_slices()
    tci.update_sitetensors()
    shapes = [t.shape for t in tci.sitetensors]
    for l in range(tci.grid.length):
        assert shapes[l][1] == tci.grid.ngrids[l]
    for l in range(tci.grid.length - 1):
        assert shapes[l][2] == shapes[l + 1][0]
    assert shapes[0][0] == 1
    assert shapes[-1][2] == 1

def test_tci_call():
    f = lambda x: jnp.linalg.norm(x) ** 2
    g = UniformGrid(2, jnp.array([10, 10]), 2, jnp.array([-1.0, -1.0]), jnp.array([1.0, 1.0]))
    pl = PivotList(2, jnp.array([10, 10]))
    tci = TCI(f, g, pl)
    tci.pivot_list.add_pivot(jnp.array([5, 5]))
    tci.pivot_list.add_pivot(jnp.array([3, 3]))
    tci.update_slices()
    tci.update_sitetensors()

    dp = tci.pivot_list.data_points()
    for p in dp:
        assert jnp.allclose(tci(p), tci.func(tci.grid.grid2space(p)))
