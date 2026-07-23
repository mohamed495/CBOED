r"""Selection (restriction) operator ``H(xi)`` for a sensor design."""

import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from cboed.core.linear_operator import LinearizedOperator


def selection_operator(indices: Int[Array, " n_obs"], n: int) -> LinearizedOperator:
    r"""Build the restriction operator ``H(xi) : R^n -> R^{n_obs}``.

    Extracts the observed components of the full state at the design
    ``indices``. Used to compose with a forward model's tangent operator
    (:func:`cboed.core.linear_operator.compose`) so that the Jacobian of the
    restricted map ``W^T u`` never needs to materialize ``W``.

    Parameters
    ----------
    indices : Int[Array, " n_obs"]
        Indices of the observed components (the design ``W``, in canonical
        selection form).
    n : int
        Dimension of the full state ``u`` the operator acts on.

    Returns
    -------
    LinearizedOperator
        Matrix-free operator of shape ``(n_obs, n)`` with:

        - ``matvec(u) = u[indices]`` -- full state to observations.
        - ``rmatvec(y)`` -- scatters ``y`` back into a length-``n`` vector of
          zeros at ``indices`` (the adjoint of the restriction).
    """

    def matvec(u: Float[Array, " n"]) -> Float[Array, " n_obs"]:
        return u[indices]

    def rmatvec(y: Float[Array, " n_obs"]) -> Float[Array, " n"]:
        return jnp.zeros(n).at[indices].set(y)

    return LinearizedOperator(matvec, rmatvec, (indices.shape[0], n))
