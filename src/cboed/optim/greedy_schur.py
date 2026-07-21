r"""Greedy via Schur complements -- ``O(n_sensors * p²)``.

Greedy selection of the design maximizing a generalized Rayleigh quotient

.. math::
    \tfrac12 \ln \frac{|W_m^T A\, W_m|}{|W_m^T B\, W_m|}

where ``(A, B)`` is a pair of SDP matrices in observation space. The module
does not know where they come from: ``(Sigma_signal, Sigma_Y_given_theta)``
gives the incremental lower bound (19), ``(Sigma_Y, Sigma_noise)`` the
conservative lower bound (20).

Why this is tractable
----------------------
The generic greedy (``optim/greedy.py``) evaluates the criterion as a black
box: ``n_candidates * n_sensors`` calls, each refactorizing
``Gamma_post^{-1}``. Here:

* the marginal gain of a candidate is a **ratio of two diagonal entries** of
  the conditioned matrices -- ``O(1)`` per candidate;
* adding a sensor is a **rank-1 update** of the two matrices -- ``O(p²)``.

No factorization, no call to the forward model: the whole cost of the model
is already paid, once, when building ``A`` and ``B``.

Warning: ``optim/greedy.py`` remains the **oracle** for this module. Do not remove it.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from jax import Array
from jaxtyping import Float, jaxtyped

from cboed.bounds.schur import schur_gain_diagonal, schur_update
from cboed.optim.base import Result


@jaxtyped(typechecker=beartype)
def greedy_schur(
    Sigma_num: Float[Array, "n_obs n_obs"],
    Sigma_den: Float[Array, "n_obs n_obs"],
    n_sensors: int,
) -> Result:
    r"""Greedy design maximizing ``½ ln |W^T A W| / |W^T B W|``.

    Parameters
    ----------
    Sigma_num, Sigma_den : Float[Array, "n_obs n_obs"]
        Numerator ``A`` and denominator ``B``, SDP, **unconditioned**.
        Incremental: ``(Sigma_signal, Sigma_Y_given_theta)``.
        Conservative: ``(Sigma_Y, Sigma_noise)``.
    n_sensors : int
        Budget. Must be at most ``n_obs``.

    Returns
    -------
    Result
        ``design``: indices in the order added.
        ``scores``: **cumulative** value of the quotient after each addition -- see Notes.

    Notes
    -----
    **The scores telescope.** The marginal gain equals
    ``log_ratio(S U {j}) - log_ratio(S)``, and ``log_ratio(∅) = 0``; the
    cumulative sum of the gains is therefore **exactly** ``log_ratio(S_k)`` at
    each step. This is what makes ``scores`` comparable to that of
    ``GreedyOptimizer``, whose contract is "criterion score after each
    addition", not a marginal gain.

    This is not an implementation detail: the telescoping **is** the
    incremental decomposition of Theorem 2.1, and the lower bound of
    Corollary 1 can be read directly off ``scores[-1]``.

    Warning: no ``jit`` around the loop: ``argmax`` returns an index that
    drives a ``schur_update``, hence a static value. The discrete greedy is
    not differentiable anyway -- the design parameterizes the operator, it
    never enters ``jax.grad``.
    """
    p = Sigma_num.shape[0]
    if not 0 < n_sensors <= p:
        raise ValueError(f"n_sensors must be in (0, {p}], got {n_sensors}")

    num_cond, den_cond = Sigma_num, Sigma_den
    selected: list[int] = []
    scores: list[float] = []
    cumulative = 0.0

    for _ in range(n_sensors):
        gain = schur_gain_diagonal(num_cond, den_cond, jnp.asarray(selected, dtype=int))
        j = int(jnp.argmax(gain))

        cumulative += float(gain[j])
        selected.append(j)
        scores.append(cumulative)

        num_cond = schur_update(num_cond, j)
        den_cond = schur_update(den_cond, j)

    return Result(design=jnp.asarray(selected, dtype=int), scores=scores)


@jax.jit
@jaxtyped(typechecker=beartype)
def log_ratio(
    Sigma_num: Float[Array, "n_obs n_obs"],
    Sigma_den: Float[Array, "n_obs n_obs"],
    design: Array,
) -> Float[Array, ""]:
    r"""``½ ln |W^T A W| / |W^T B W|`` evaluated directly.

    Path independent of :func:`greedy_schur`: submatrices and ``slogdet``,
    no Schur complement. This is the oracle for ``scores``.
    """
    ix = jnp.ix_(design, design)
    _, ln = jnp.linalg.slogdet(Sigma_num[ix])
    _, ld = jnp.linalg.slogdet(Sigma_den[ix])
    return 0.5 * (ln - ld)
