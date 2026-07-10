from collections.abc import Callable

from beartype import beartype  # type: ignore
from jaxtyping import Array, Float, jaxtyped  # type: ignore


class LinearizedOperator:
    """Linear operator matrix-free.

    - matvec(v)  : v -> G v
    - rmatvec(w) : w -> G^T w
    - T          : opérateur adjoint
    - shape      : (n_out, n_in)
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
        return self._mv(v)

    @jaxtyped(typechecker=beartype)
    def rmatvec(self, w: Float[Array, " n_out"]) -> Float[Array, " n_in"]:  # type: ignore
        return self._rmv(w)

    @property
    def T(self) -> "LinearizedOperator":
        n_out, n_in = self.shape
        return LinearizedOperator(self._rmv, self._mv, (n_in, n_out))
