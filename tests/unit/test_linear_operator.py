import numpy as np

from cboed.core.linear_operator import LinearizedOperator


def test_operator_matches_matrix():
    A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    op = LinearizedOperator(
        matvec=lambda x: A @ x,
        rmatvec=lambda y: A.T @ y,
        shape=A.shape,
    )

    x = np.array([7.0, 8.0])
    y = np.array([1.0, 2.0, 3.0])

    np.testing.assert_allclose(op.matvec(x), A @ x)
    np.testing.assert_allclose(op.rmatvec(y), A.T @ y)
    np.testing.assert_allclose(op.T.matvec(y), A.T @ y)
    np.testing.assert_allclose(op.T.rmatvec(x), A @ x)
