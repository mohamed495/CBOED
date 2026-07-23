# estimators/base.py
from abc import ABC, abstractmethod

import jax
from jaxtyping import Array, Float, Int


def chunked_vmap(f, *xs, chunk_size: int | None = None):
    """Apply ``f`` over the leading axis of ``xs``, vectorized or chunked.

    Equivalent to ``jax.vmap(f)(*xs)`` when ``chunk_size`` is ``None``, or to
    ``jax.lax.map(f, xs, batch_size=chunk_size)`` otherwise.

    Parameters
    ----------
    f : Callable
        Function applied to one slice along the leading axis of each array
        in ``xs``.
    *xs : Array
        One or more arrays sharing the same leading (batch) dimension.
    chunk_size : int or None, optional
        If ``None`` (default), a single ``vmap`` processes the whole batch
        at once. Otherwise, ``jax.lax.map`` processes the batch in
        sequential chunks of size ``chunk_size``.

    Returns
    -------
    Array or pytree of Array
        The stacked outputs of ``f`` applied to every element of the batch,
        as if returned by ``vmap(f)(*xs)``.

    Notes
    -----
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
    """Base contract for EIG estimators.

    Subclasses decide *how* the Expected Information Gain is approximated
    (nested Monte Carlo, variational NMC, Laplace linearization, prior
    contrastive estimation, ...).

    Parameters
    ----------
    **hyperparameters : dict
        Keyword hyperparameters stored verbatim (e.g. ``inference``,
        ``prior``, ``likelihood`` depending on the subclass).
    """

    def __init__(self, **hyperparameters):
        self._hyperparameters = hyperparameters

    @property
    def inference(self):
        """The :class:`~cboed.inference.base.InferenceModel`, if used by the subclass."""
        return self._hyperparameters["inference"]

    @abstractmethod
    def estimate(
        self,
        design: Int[Array, " n_obs"] | None = None,
    ) -> Float[Array, ""]:
        """Estimate the EIG for a given design.

        Parameters
        ----------
        design : Int[Array, " n_obs"] or None, optional
            Indices of the observed sensors. ``None`` means the full field
            is observed.

        Returns
        -------
        Float[Array, ""]
            Estimate of the Expected Information Gain for ``design``.
        """
        ...
