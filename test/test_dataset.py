import pytest
from alci.dataset import DataSet
from alci.grid import UniformGrid
import jax
import jax.numpy as jnp

def test_construct():
    g = UniformGrid(2, jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0]), jnp.array([10, 10]))
    d = DataSet(g)
    assert d.points.shape == (0, 2)
    assert d.vals.shape == (0, )
    assert d.pivot_pos.shape == (2, 0)

def test_add_data():
    g = UniformGrid(2, jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0]), jnp.array([2, 2]))
    d = DataSet(g)
    d.add_data(jnp.array([[0, 0], [0, 1], [1, 0]]), jnp.array([1.0, 0.0, 0.0]))
    assert jnp.all(d.points[0] == jnp.array([0, 0]))
    assert jnp.isclose(d.vals[0], jnp.array(1.0))

def test_isnewpivot():
    g = UniformGrid(2, jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0]), jnp.array([2, 2]))
    d = DataSet(g)
    assert d.isnewpivot(jnp.array([0, 0]))
    assert d.isnewpivot(jnp.array([0, 1]))
    assert d.isnewpivot(jnp.array([1, 0]))
    assert d.isnewpivot(jnp.array([1, 1]))

    d.update_pivot(jnp.array([0, 0]))
    assert ~d.isnewpivot(jnp.array([0, 0]))
    assert ~d.isnewpivot(jnp.array([0, 1]))
    assert ~d.isnewpivot(jnp.array([1, 0]))
    assert d.isnewpivot(jnp.array([1, 1]))

def test_new_points():
    g = UniformGrid(2, jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0]), jnp.array([2, 2]))
    d = DataSet(g)
    d.update_pivot(jnp.array([0, 0]))
    np = d.new_points(jnp.array([0, 0]))
    assert jnp.all(np == jnp.array([[1, 0], [0, 1], [0, 0]]))

def test_query():
    f = lambda x: jnp.linalg.norm(x) ** 2
    g = UniformGrid(2, jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0]), jnp.array([2, 2]))
    d = DataSet(g)
    assert jnp.allclose(d.query(f, jnp.array([[0, 0], [1, 1]])), jnp.array([0.0, 2.0]))
