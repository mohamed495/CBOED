r"""Compute sample-based diagnostic matrices -- §3.1.

Provides ``Sigma_Y`` and ``Sigma_{Y|theta}``. **No alternative**: unlike
``Sigma_signal``/``Sigma_noise``, which have two routes (§3.2 and §3.3),
these two can only be obtained by sampling.

Paired-difference estimator
----------------------------
Rather than the usual empirical covariance, we exploit
``Cov(u(eta)) = ½ E[(u(eta) - u(eta'))^⊗2]`` for ``eta'`` an independent
copy:

.. math::
    \Sigma_Y^{(N)} = \Sigma_{\rm obs}
    + \frac{1}{2N} \sum_{i=1}^N (u(\eta^{(i)}) - u(\eta'^{(i)}))^{\otimes 2}

Unbiased, and with no empirical mean to subtract -- hence no cancellation
error of the kind that threatens ``E[X X^T] - \bar{X}\bar{X}^T``. Cost:
``2N`` forward-model evaluations.

``Sigma_{Y|theta}`` follows the same form with ``eta, eta'`` **conditionally
independent given theta** (27): ``eta ~ pi_eta``, ``theta ~ pi_{theta|eta}``,
``eta' ~ pi_{eta|theta}``. The last draw is the paper's bottleneck -- it
generally requires MCMC. Rem. 3.1 gives it in closed form when ``h`` is
linear.

Notes
-----
⚠️ **Pick-freeze (Rem. 3.2) does not apply to this test bench.** It requires
``eta_1 ⊥ eta_2``; the two halves of the field come from the same GP, so
``Sigma_{theta eta} ≠ 0``. It is Rem. 3.1 that applies here.

⚠️ Unlike Prop. 4, this route is **regular at ``Sigma_xi = 0``**: Rem. 3.1
stays valid without noise (``eta|theta`` degenerates cleanly into a
``delta``). Where the gradient route has a singular limit, the sample route
does not.
"""

from functools import partial

import jax
import jax.numpy as jnp
from beartype import beartype
from jax import Array
from jaxtyping import Float, PRNGKeyArray, jaxtyped

from cboed.priors.base import Prior


@jax.jit
@jaxtyped(typechecker=beartype)
def _paired_covariance(
    diffs: Float[Array, "n_samples n_obs"],
) -> Float[Array, "n_obs n_obs"]:
    r"""Compute ``(1/2N) sum_i d_i d_i^T`` -- the estimator (26)/(27), given the paired diffs."""
    out = 0.5 * diffs.T @ diffs / diffs.shape[0]
    return 0.5 * (out + out.T)


@jax.jit
@jaxtyped(typechecker=beartype)
def _psd_sqrt(A: Float[Array, "n n"]) -> Float[Array, "n n"]:
    r"""Compute a PSD square root of ``A`` via ``eigh``, eigenvalues clipped to zero.

    Notes
    -----
    Not ``cholesky``: the posterior covariance of Rem. 3.1 **degenerates**
    to zero when ``Sigma_xi -> 0`` and ``B = I`` (``eta|theta`` becomes a
    Dirac). This is a nominal case, not an accident -- and LAPACK returns
    ``nan`` on a singular PSD matrix.
    """
    ev, P = jnp.linalg.eigh(A)
    return P @ jnp.diag(jnp.sqrt(jnp.clip(ev, 0.0, None)))


@partial(jax.jit, static_argnums=(0, 1, 4))
@jaxtyped(typechecker=beartype)
def sample_Sigma_Y(
    u,
    prior_eta: Prior,
    Sigma_obs: Float[Array, "n_obs n_obs"],
    key: PRNGKeyArray,
    n_samples: int,
) -> Float[Array, "n_obs n_obs"]:
    r"""Compute ``Sigma_Y = Sigma_obs + Cov(u(eta))`` -- equation (26).

    Uses the paired-difference estimator
    ``Cov(u(eta)) = (1/2N) sum (u(eta) - u(eta'))^{⊗2}`` (see module docstring).

    Parameters
    ----------
    u : Callable
        Forward model ``eta -> observations``, without a design.
    prior_eta : Prior
        Prior on ``eta``.
    Sigma_obs : Float[Array, "n_obs n_obs"]
        Observation noise covariance.
    key : PRNGKeyArray
        Random key, split into two independent draws ``eta``, ``eta'``.
    n_samples : int
        Number of **pairs**. Cost: ``2 * n_samples`` evaluations of ``u``.

    Returns
    -------
    Float[Array, "n_obs n_obs"]
        ``Sigma_Y``.
    """
    k1, k2 = jax.random.split(key)
    eta = prior_eta.sample(k1, n_samples)
    eta_prime = prior_eta.sample(k2, n_samples)
    diffs = jax.vmap(u)(eta) - jax.vmap(u)(eta_prime)
    return Sigma_obs + _paired_covariance(diffs)


@partial(jax.jit, static_argnums=(0, 1, 6))
@jaxtyped(typechecker=beartype)
def sample_Sigma_Y_given_theta(
    u,
    prior_eta: Prior,
    B: Float[Array, "n_param n_eta"],
    Sigma_obs: Float[Array, "n_obs n_obs"],
    Sigma_xi: Float[Array, "n_param n_param"],
    key: PRNGKeyArray,
    n_samples: int,
) -> Float[Array, "n_obs n_obs"]:
    r"""Compute ``Sigma_{Y|theta} = Sigma_obs + E[Cov(u(eta)|theta)]`` -- (27) via Rem. 3.1.

    Parameters
    ----------
    u : Callable
        Forward model ``eta -> observations``, without a design.
    prior_eta : Prior
        Prior on ``eta``, assumed Gaussian (mean ``prior_eta.mu``, covariance
        ``prior_eta.Sigma()``).
    B : Float[Array, "n_param n_eta"]
        Jacobian of ``h``, **assumed constant**: the closed form of Rem.
        3.1 only exists for linear ``h``. For non-linear ``h``, MCMC
        targeting ``pi_{eta|theta}`` is required -- out of scope here.
    Sigma_obs : Float[Array, "n_obs n_obs"]
        Observation noise covariance.
    Sigma_xi : Float[Array, "n_param n_param"]
        Covariance of ``xi``. **Can be zero**: see Rem. 3.1.
    key : PRNGKeyArray
        Random key, split into three independent draws (``theta`` noise,
        ``xi`` noise, posterior noise).
    n_samples : int
        Number of **pairs**. Cost: ``2 * n_samples`` evaluations of ``u``.

    Returns
    -------
    Float[Array, "n_obs n_obs"]
        ``Sigma_{Y|theta}``.

    Notes
    -----
    The Kalman gain ``K = Sigma_eta B^T (B Sigma_eta B^T + Sigma_xi)^{-1}``
    and the posterior covariance ``Sigma_pos = Sigma_eta - K B Sigma_eta``
    **do not depend on theta**: factorized once, outside the loop. Only the
    mean depends on it.
    """
    k_eta, k_xi, k_pos = jax.random.split(key, 3)

    Sigma_eta = prior_eta.Sigma()
    m_eta = prior_eta.mu

    # -- Rem. 3.1: eta|theta Gaussian, covariance independent of theta ------
    S = B @ Sigma_eta @ B.T + Sigma_xi
    K = jnp.linalg.solve(S, B @ Sigma_eta).T  # (q, d)
    Sigma_pos = Sigma_eta - K @ B @ Sigma_eta
    L_pos = _psd_sqrt(0.5 * (Sigma_pos + Sigma_pos.T))

    # -- eta ~ pi_eta, theta ~ pi_{theta|eta}, eta' ~ pi_{eta|theta} --------
    eta = prior_eta.sample(k_eta, n_samples)
    L_xi = _psd_sqrt(Sigma_xi)
    z_xi = jax.random.normal(k_xi, (n_samples, Sigma_xi.shape[0]))
    theta = eta @ B.T + z_xi @ L_xi.T

    z_pos = jax.random.normal(k_pos, (n_samples, m_eta.shape[0]))
    eta_prime = m_eta + (theta - m_eta @ B.T) @ K.T + z_pos @ L_pos.T

    diffs = jax.vmap(u)(eta) - jax.vmap(u)(eta_prime)
    return Sigma_obs + _paired_covariance(diffs)


@partial(jax.jit, static_argnums=(0, 1, 4))
@jaxtyped(typechecker=beartype)
def sample_diagnostics_standard(
    u,
    prior_theta: Prior,
    Sigma_obs: Float[Array, "n_obs n_obs"],
    key: PRNGKeyArray,
    n_samples: int,
) -> tuple[Float[Array, "n_obs n_obs"], Float[Array, "n_obs n_obs"]]:
    r"""Compute ``(Sigma_Y, Sigma_Y_given_theta)`` in the standard case ``Y = u(theta) + eps``.

    Parameters
    ----------
    u : Callable
        Forward model ``theta -> observations``, without a design.
    prior_theta : Prior
        Prior on ``theta``.
    Sigma_obs : Float[Array, "n_obs n_obs"]
        Observation noise covariance.
    key : PRNGKeyArray
        Random key for sampling ``theta``.
    n_samples : int
        Number of **pairs**. Cost: ``2 * n_samples`` evaluations of ``u``.

    Returns
    -------
    Sigma_Y : Float[Array, "n_obs n_obs"]
        From :func:`sample_Sigma_Y`.
    Sigma_Y_given_theta : Float[Array, "n_obs n_obs"]
        ``Sigma_obs``, exactly (see Notes).

    Notes
    -----
    Prop. 2 with ``h = id`` and ``xi = 0`` gives ``E[Cov(u(theta)|theta)] = 0``,
    hence ``Sigma_Y_given_theta = Sigma_obs`` **exactly**. No sampling
    needed: this is an equality to posit, not a limit to approximate.

    Symmetric to :func:`cboed.bounds.diagnostics.gradient_based.gradient_diagnostics_standard`,
    and for the same reason.
    """
    return sample_Sigma_Y(u, prior_theta, Sigma_obs, key, n_samples), Sigma_obs
