import jax.numpy as jnp  # type: ignore

from cboed.core.linear_operator import LinearizedOperator


def test_operator_matches_matrix():
    A = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    op = LinearizedOperator(
        matvec=lambda x: A @ x,
        rmatvec=lambda y: A.T @ y,
        shape=A.shape,
    )

    x = jnp.array([7.0, 8.0])
    y = jnp.array([1.0, 2.0, 3.0])

    jnp.allclose(op.matvec(x), A @ x)
    jnp.allclose(op.rmatvec(y), A.T @ y)
    jnp.allclose(op.T.matvec(y), A.T @ y)
    jnp.allclose(op.T.rmatvec(x), A @ x)
