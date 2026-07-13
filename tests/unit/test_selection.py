import jax.numpy as jnp

from cboed.core.selection import selection_operator


def test_selection_extracts():
    u = jnp.array([10.0, 20.0, 30.0, 40.0])
    H = selection_operator(jnp.array([0, 2]), n=4)
    assert jnp.array_equal(H.matvec(u), jnp.array([10.0, 30.0]))


def test_selection_adjoint():
    """⟨Hu, y⟩ = ⟨u, Hᵀy⟩ pour tous u, y."""
    H = selection_operator(jnp.array([0, 2]), n=4)
    u = jnp.arange(4.0)
    y = jnp.array([1.0, 2.0])
    lhs = jnp.dot(H.matvec(u), y)
    rhs = jnp.dot(u, H.rmatvec(y))
    assert jnp.allclose(lhs, rhs)


def test_selection_shape():
    H = selection_operator(jnp.array([1, 3]), n=5)
    assert H.shape == (2, 5)
