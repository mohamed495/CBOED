r"""Compute Schur complements and their rank-1 updates.

Equations (11)-(14) of Theorem 2.1: after selecting design ``W_m``, the four
diagnostic matrices are conditioned via

.. math::
    \Sigma(W_m) = \Sigma - \Sigma W_m (W_m^T \Sigma W_m)^{-1} W_m^T \Sigma

This is **the** computational core of ``bounds/``. The theorem is not merely
stated in this form: it is this form that makes the greedy algorithm
tractable, because Schur complements **compose** -- conditioning by
``S U {j}`` is the same as conditioning by ``{j}`` the matrix already
conditioned by ``S``. Hence a rank-1 update in ``O(p²)`` where direct
recomputation costs ``O(p² m + m³)``.

This module knows nothing about the paper or about BOED: it is linear
algebra on an SDP matrix. It is the first piece of what will become
``linalg/``.

Notes
-----
``W_m`` is never materialized. For a canonical selection (columns = basis
vectors), ``W_m^T \Sigma W_m`` is the submatrix ``Sigma[ix_(design, design)]``
-- see ``core/selection.py``, which already is that ``W_m^T``.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from beartype import beartype
from jax import Array
from jaxtyping import Float, Int, jaxtyped


@jax.jit
@jaxtyped(typechecker=beartype)
def schur_complement(
    Sigma: Float[Array, "n_obs n_obs"],
    design: Int[Array, " n_sensors"] | None = None,
) -> Float[Array, "n_obs n_obs"]:
    r"""Compute ``Sigma(W_m)`` from scratch -- equations (11)-(14).

    Cost ``O(p² m + m³)``. This is the **oracle** for :func:`schur_update`:
    an independent numerical path, no recurrence.

    Parameters
    ----------
    Sigma : Float[Array, "n_obs n_obs"]
        SDP matrix to condition (one of the four diagnostic matrices).
    design : Int[Array, " n_sensors"] | None
        Selected indices. ``None`` or empty: no conditioning, returns ``Sigma``.

    Returns
    -------
    Float[Array, "n_obs n_obs"]
        Schur complement, symmetrized.

    Notes
    -----
    The rows and columns of ``design`` are **exactly zero** in the output (up
    to rounding): conditioning on an observation removes all the information
    it carried. See :func:`schur_gain_diagonal`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> Sigma = jnp.array([[2.0, 1.0], [1.0, 2.0]])
    >>> jnp.round(schur_complement(Sigma, jnp.array([0])), 4)
    Array([[0. , 0. ],
           [0. , 1.5]], dtype=float64)
    """
    if design is None or design.shape[0] == 0:
        return Sigma

    Sigma_S = Sigma[:, design]  # (p, m)
    Sigma_SS = Sigma[jnp.ix_(design, design)]  # (m, m)
    chol = jsp.linalg.cho_factor(Sigma_SS, lower=True)
    out = Sigma - Sigma_S @ jsp.linalg.cho_solve(chol, Sigma_S.T)
    return 0.5 * (out + out.T)


@jax.jit
@jaxtyped(typechecker=beartype)
def schur_update(
    Sigma_cond: Float[Array, "n_obs n_obs"],
    j: Int[Array, ""] | int,
) -> Float[Array, "n_obs n_obs"]:
    r"""Add sensor ``j`` to a Schur complement -- **rank-1** update.

    .. math::
        \Sigma(S \cup \{j\}) = \Sigma(S)
        - \frac{\Sigma(S)_{:,j}\, \Sigma(S)_{j,:}}{\Sigma(S)_{j,j}}

    Cost ``O(p²)``: this is what takes the greedy algorithm from
    ``O(n_sensors * p² m)`` to ``O(n_sensors * p²)``.

    Parameters
    ----------
    Sigma_cond : Float[Array, "n_obs n_obs"]
        Complement already conditioned by ``S`` (or raw ``Sigma`` if ``S`` is
        empty).
    j : int or Int[Array, ""]
        Index of the added sensor (plain Python ``int`` or 0-d integer
        array). **Must not already be in ``S``**: the pivot
        ``Sigma_cond[j, j]`` would be numerically zero there and the
        division would blow up.

    Returns
    -------
    Float[Array, "n_obs n_obs"]
        ``Sigma(S U {j})``, symmetrized.

    Notes
    -----
    The symmetrization at each step is not cosmetic: over a chain of
    ``n_sensors`` updates, rounding asymmetry accumulates and eventually
    makes ``cho_factor`` fail downstream.
    """
    col = Sigma_cond[:, j]
    pivot = Sigma_cond[j, j]
    out = Sigma_cond - jnp.outer(col, col) / pivot
    return 0.5 * (out + out.T)


@jax.jit
@jaxtyped(typechecker=beartype)
def schur_gain_diagonal(
    Sigma_num_cond: Float[Array, "n_obs n_obs"],
    Sigma_den_cond: Float[Array, "n_obs n_obs"],
    selected: Int[Array, " n_sensors"] | None = None,
) -> Float[Array, " n_obs"]:
    r"""Compute the marginal gain of **every** candidate sensor, in one shot.

    For ``W_new = e_j``, the incremental term of Theorem 2.1 reduces to a
    ratio of two diagonal entries:

    .. math::
        \mathrm{gain}(j) = \tfrac12 \ln
        \frac{\Sigma_{\rm num}(W_m)_{j,j}}{\Sigma_{\rm den}(W_m)_{j,j}}

    This is **the whole** trick: ``O(1)`` per candidate, ``O(p)`` for the
    full pass, no factorization, no call to the forward model.

    Parameters
    ----------
    Sigma_num_cond, Sigma_den_cond : Float[Array, "n_obs n_obs"]
        Numerator and denominator **already conditioned** by the current
        design.
        Incremental: ``(Sigma_signal, Sigma_Y_given_theta)``.
        Conservative: ``(Sigma_Y, Sigma_noise)``.
    selected : Int[Array, " n_sensors"] | None
        Sensors already selected. Masked to ``-inf``.

    Returns
    -------
    Float[Array, " n_obs"]
        Gain per candidate; ``-inf`` at indices already selected.

    Notes
    -----
    The masking is **mandatory**, not defensive. For ``j`` already
    selected, both diagonal entries are ~1e-16: the ratio becomes a quotient
    of rounding noise, finite and arbitrary. Without the mask, the greedy
    algorithm silently reselects a sensor already taken.
    """
    num = jnp.diagonal(Sigma_num_cond)
    den = jnp.diagonal(Sigma_den_cond)
    gain = 0.5 * (jnp.log(num) - jnp.log(den))

    if selected is not None and selected.shape[0] > 0:
        gain = gain.at[selected].set(-jnp.inf)
    return gain


@jax.jit
@jaxtyped(typechecker=beartype)
def log_ratio(
    Sigma_num: Float[Array, "n_obs n_obs"],
    Sigma_den: Float[Array, "n_obs n_obs"],
    design: Int[Array, " n_sensors"] | None = None,
) -> Float[Array, ""]:
    r"""Compute ``½ ln |W^T A W| / |W^T B W|`` -- generalized Rayleigh quotient, flat computation.

    Parameters
    ----------
    Sigma_num : Float[Array, "n_obs n_obs"]
        Numerator matrix ``A`` (e.g. ``Sigma_signal`` or ``Sigma_Y``).
    Sigma_den : Float[Array, "n_obs n_obs"]
        Denominator matrix ``B`` (e.g. ``Sigma_Y_given_theta`` or ``Sigma_noise``).
    design : Int[Array, " n_sensors"] | None
        Selected indices ``W``. ``None`` -> full design ``I_p``, used by the
        conservative bounds for their reference term.

    Returns
    -------
    Float[Array, ""]
        ``½ ln(|W^T A W| / |W^T B W|)``.

    Notes
    -----
    Submatrices and ``slogdet``, no Schur complement: a path independent of
    :func:`schur_gain_diagonal` and of ``greedy_schur``, hence their oracle.
    """
    if design is None:
        num, den = Sigma_num, Sigma_den
    else:
        ix = jnp.ix_(design, design)
        num, den = Sigma_num[ix], Sigma_den[ix]
    _, ln = jnp.linalg.slogdet(num)
    _, ld = jnp.linalg.slogdet(den)
    return 0.5 * (ln - ld)
