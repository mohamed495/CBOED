# estimators/base.py
from abc import ABC, abstractmethod

import jax
from jaxtyping import Array, Float, Int


def chunked_vmap(f, *xs, chunk_size: int | None = None):
    """``vmap(f)(*xs)``, or ``lax.map`` in batches of ``chunk_size`` if given.

    NMC estimators vectorize an outer loop (``n_outer``) whose every element
    already launches a vectorized inner loop (``n_inner``) -- XLA fuses the
    two into an effective ``n_outer x n_inner`` batch, and at real scale (full
    field, large ``n_outer``/``n_inner``) this batch far exceeds available GPU
    memory. ``chunk_size`` bounds peak memory to ``chunk_size x n_inner`` by
    processing the outer axis in sequential batches (``lax.map``) instead of a
    single ``vmap`` -- slower, but necessary as soon as ``n_outer x n_inner``
    no longer fits in memory. ``None`` (default): a single ``vmap``, as
    before -- no overhead at small scales (tests, dev).
    """
    if chunk_size is None:
        return jax.vmap(f)(*xs)
    if len(xs) == 1:
        return jax.lax.map(f, xs[0], batch_size=chunk_size)
    return jax.lax.map(lambda args: f(*args), xs, batch_size=chunk_size)


class EIGEstimator(ABC):
    """EIG estimator. Subclasses decide *how* to approximate it."""

    def __init__(self, **hyperparameters):
        self._hyperparameters = hyperparameters

    @property
    def inference(self):
        return self._hyperparameters["inference"]

    @abstractmethod
    def estimate(
        self,
        design: Int[Array, " n_obs"] | None = None,
    ) -> Float[Array, ""]: ...
