# core/selection.py  (or in linear_operator.py)
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from cboed.core.linear_operator import LinearizedOperator


def selection_operator(indices: Int[Array, " n_obs"], n: int) -> LinearizedOperator:
    """H(ξ) : Rⁿ → R^{n_obs}, extracts the observed components.

    matvec  : full state → observations (u ↦ u[indices])
    rmatvec : observations → full state (scatter, adjoint)
    """

    def matvec(u: Float[Array, " n"]) -> Float[Array, " n_obs"]:
        return u[indices]

    def rmatvec(y: Float[Array, " n_obs"]) -> Float[Array, " n"]:
        return jnp.zeros(n).at[indices].set(y)

    return LinearizedOperator(matvec, rmatvec, (indices.shape[0], n))
