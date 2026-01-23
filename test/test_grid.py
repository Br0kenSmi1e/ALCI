import pytest
from alci.grid import UniformGrid, QuanticsGrid
import jax
import jax.numpy as jnp

def test_gridconstruction():
    g = UniformGrid(2, jnp.array([10, 10]), 2, jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0]))
    assert g.length == 2
    assert jnp.all(g.ngrids == jnp.array([10, 10]))
    assert g.dim == 2
    assert jnp.allclose(g.start, jnp.array([0.0, 0.0]))
    assert jnp.allclose(g.end, jnp.array([1.0, 1.0]))

def test_grid2space():
    g = UniformGrid(2, jnp.array([10, 10]), 2, jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0]))
    assert jnp.allclose(g.grid2space(jnp.array([0, 0])), g.start)
    assert jnp.allclose(g.grid2space(g.ngrids - 1), g.end)

def test_all_grid_pts():
    g = UniformGrid(2, jnp.array([10, 10]), 2, jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0]))
    pts = g.all_grid_pts()
    assert pts.shape == (100, 2)

def test_grid_and_space():
    g = UniformGrid(2, jnp.array([10, 10]), 2, jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0]))
    gp, sp = g.grid_and_space()
    assert gp.shape == (100, 2)
    assert sp.shape == (100, 2)

def test_quantics_construct():
    g = QuanticsGrid(2, 4, 2, jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0]))
    assert g.base == 2
    assert g.nbit == 4
    assert g.dim == 2
    assert jnp.allclose(g.start, jnp.array([0.0, 0.0]))
    assert jnp.allclose(g.end, jnp.array([1.0, 1.0]))

def test_quantics_grid2space():
    g = QuanticsGrid(2, 4, 2, jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0]))
    assert jnp.allclose(g.grid2space(jnp.array([0, 0, 0, 0, 0, 0, 0, 0])), g.start)
    assert jnp.allclose(g.grid2space(jnp.array([1, 1, 1, 1, 1, 1, 1, 1])), g.end)
    assert jnp.allclose(g.grid2space(jnp.array([0, 1, 0, 1, 0, 1, 0, 1])), jnp.array([0.0, 1.0]))
    assert jnp.allclose(g.grid2space(jnp.array([1, 0, 1, 0, 1, 0, 1, 0])), jnp.array([1.0, 0.0]))
