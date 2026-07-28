from dataclasses import dataclass, field

import jax
from jax import Array
from jaxtyping import Float


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
    Any computation that vectorizes an outer loop whose every element
    already launches its own vectorized inner work (nested Monte Carlo
    estimators, or a batch of per-sample Jacobians followed by a per-sample
    quadratic form) gets fused by XLA into one large effective batch -- at
    real scale (full field, large sample counts) this can far exceed
    available GPU memory. ``chunk_size`` bounds peak memory to
    ``chunk_size`` times the cost of one element, by processing the batch
    axis in sequential chunks (``lax.map``) instead of a single ``vmap`` --
    slower, but necessary as soon as the full batch no longer fits in
    memory. ``None`` (default): a single ``vmap``, as before -- no overhead
    at small scales (tests, dev).
    """
    if chunk_size is None:
        return jax.vmap(f)(*xs)
    if len(xs) == 1:
        return jax.lax.map(f, xs[0], batch_size=chunk_size)
    return jax.lax.map(lambda args: f(*args), xs, batch_size=chunk_size)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DiagnosticMatrices:
    r"""Container for the four diagnostic matrices of Theorem 2.1.

    Bundles ``Sigma_Y``, ``Sigma_Y_given_theta``, ``Sigma_signal`` and
    ``Sigma_noise`` -- the matrices consumed by :func:`cboed.bounds.bounds.incremental_bounds`
    and :func:`cboed.bounds.bounds.conservative_bounds` to certify an
    enclosure of ``EIG(design)``. They come from **two distinct sources**,
    never a single one:

    ===================  ==========================  ====================
    Matrices             Route                       Alternative
    ===================  ==========================  ====================
    ``Sigma_Y``,         §3.1 sample-based (26)(27)  none
    ``Sigma_Y_given_theta``
    ``Sigma_signal``,    §3.3 gradient, Prop. 4      §3.2 approximation
    ``Sigma_noise``
    ===================  ==========================  ====================

    Attributes
    ----------
    Sigma_Y : Float[Array, "n_obs n_obs"]
        ``Sigma_obs + Cov(u(eta))``.
    Sigma_Y_given_theta : Float[Array, "n_obs n_obs"]
        ``Sigma_obs + E[Cov(u(eta)|theta)]``.
    Sigma_signal : Float[Array, "n_obs n_obs"]
        Bound on the Fisher information: ``Sigma_signal^{-1} ⪰ I_Y``.
    Sigma_noise : Float[Array, "n_obs n_obs"]
        ``Sigma_noise^{-1} ⪰ E[I_{Y|theta}]``.
    certified : bool
        Is the Loewner order required by Theorem 2.1 **guaranteed**?

        ``True`` for the gradient route (Prop. 4). ``False`` for the
        approximation route (§3.2), whose inequality points the **wrong way**
        (``(Sigma^{(N,F)}_signal)^{-1} ⪯ I_Y``): consistent as ``N → ∞`` and
        ``F → L²``, but with no guarantee at finite ``N``. The paper is
        explicit -- it *cannot be safely used in Theorem 2.1*.

    Notes
    -----
    ``certified`` is not decorative. A certified bound must **refuse** an
    uncertified diagnostic, or degrade its own result -- never stay silent.
    That is the only honest way to let two routes coexist when only one of
    them certifies.

    Not to be confused with ``BoundResult.is_certified`` ("is the gap below
    the tolerance?"). Two useful, distinct notions.
    """

    Sigma_Y: Float[Array, "n_obs n_obs"]
    Sigma_Y_given_theta: Float[Array, "n_obs n_obs"]
    Sigma_signal: Float[Array, "n_obs n_obs"]
    Sigma_noise: Float[Array, "n_obs n_obs"]
    certified: bool = field(metadata=dict(static=True))
