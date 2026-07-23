r"""Matrix-free linear operators: application, adjoint, and composition.

Forward-model Jacobians in :mod:`cboed.core` are linear maps that are cheap
to apply (via ``jax.linearize``/``jax.vjp``) but expensive or wasteful to
materialize as dense matrices. :class:`LinearizedOperator` wraps a pair
``(matvec, rmatvec)`` so that composition (e.g. with a selection operator,
see :mod:`cboed.core.selection`) stays matrix-free all the way through.
"""

from collections.abc import Callable

from beartype import beartype  # type: ignore
from jaxtyping import Array, Float, jaxtyped  # type: ignore


class LinearizedOperator:
    r"""A matrix-free linear operator ``G : R^{n_in} -> R^{n_out}``.

    Parameters
    ----------
    matvec : Callable[[Float[Array, " n_in"]], Float[Array, " n_out"]]
        Forward application, ``v -> G v``.
    rmatvec : Callable[[Float[Array, " n_out"]], Float[Array, " n_in"]]
        Adjoint application, ``w -> G^T w``.
    shape : tuple[int, int]
        ``(n_out, n_in)``, the shape of the (never materialized) matrix
        ``G`` would represent.

    Attributes
    ----------
    shape : tuple[int, int]
        ``(n_out, n_in)``.
    """

    def __init__(
        self,
        matvec: Callable[[Float[Array, " n_in"]], Float[Array, " n_out"]],  # type: ignore
        rmatvec: Callable[[Float[Array, " n_out"]], Float[Array, " n_in"]],  # type: ignore
        shape: tuple[int, int],
    ) -> None:
        self._mv = matvec
        self._rmv = rmatvec
        self.shape = shape

    @jaxtyped(typechecker=beartype)
    def matvec(self, v: Float[Array, " n_in"]) -> Float[Array, " n_out"]:  # type: ignore
        """Apply ``G`` to ``v``."""
        return self._mv(v)

    @jaxtyped(typechecker=beartype)
    def rmatvec(self, w: Float[Array, " n_out"]) -> Float[Array, " n_in"]:  # type: ignore
        """Apply the adjoint ``G^T`` to ``w``."""
        return self._rmv(w)

    @property
    def T(self) -> "LinearizedOperator":
        """The adjoint operator ``G^T``, obtained by swapping matvec/rmatvec.

        Returns
        -------
        LinearizedOperator
            Operator of shape ``(n_in, n_out)`` with ``matvec = rmatvec`` and
            ``rmatvec = matvec`` of ``self``.
        """
        n_out, n_in = self.shape
        return LinearizedOperator(self._rmv, self._mv, (n_in, n_out))


def compose(outer, inner):
    r"""Compose two matrix-free operators, ``outer \circ inner``.

    Builds the operator representing ``H G`` (apply ``inner`` = ``G`` first,
    then ``outer`` = ``H``) without ever forming either matrix.

    Parameters
    ----------
    outer : LinearizedOperator
        Left operator ``H``, applied second in ``matvec``.
    inner : LinearizedOperator
        Right operator ``G``, applied first in ``matvec``.

    Returns
    -------
    LinearizedOperator
        Operator of shape ``(outer.shape[0], inner.shape[1])`` whose
        ``matvec`` is ``v -> H(G(v))`` and whose ``rmatvec`` is
        ``w -> G^T(H^T(w))``, using the identity ``(H G)^T = G^T H^T``.
    """

    def matvec(v):
        return outer.matvec(inner.matvec(v))

    def rmatvec(w):
        return inner.rmatvec(outer.rmatvec(w))  # (H∘G)ᵀ = Gᵀ∘Hᵀ

    return LinearizedOperator(matvec, rmatvec, (outer.shape[0], inner.shape[1]))
