r"""Gradient-based diagnostic matrices -- Proposition 4, §3.3.

The module separates **the moments** (the expensive part: `N` Jacobians of
the forward model) from **the assembly** (linear algebra on `q x q`
matrices). Two assembly routes exist for the same quantity -- they act as
oracles for one another.

Moments (31)-(34)
-----------------
    L(u)  = E[Jac u(eta)]^T                                              (q, p)
    H(u)  = E[(Jac u - E[Jac u])^T Sigma_obs^{-1} (Jac u - E[Jac u])]    (q, q)
    J(h)  = E[Jac h(eta)^T Sigma_xi^{-1} Jac h(eta)]                     (q, q)
    I_eta = Cov(grad_eta ln pi(eta))                                     (q, q)

Assembly (35)-(36)
------------------
    Sigma_signal = Sigma_obs + L^T (H + I_eta)^{-1} L
    Sigma_noise  = Sigma_obs + L^T (H + I_eta + J(h))^{-1} L

`J(h)` is the only difference between the two.
"""

from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from beartype import beartype
from jax import Array
from jaxtyping import Float, PRNGKeyArray, jaxtyped

from cboed.priors.base import Prior


@jax.jit
@jaxtyped(typechecker=beartype)
def psd_sqrt(A: Float[Array, "n n"]) -> Float[Array, "n n"]:
    """PSD square root via ``eigh``, eigenvalues clipped to zero.

    Not ``cholesky``: the matrices involved can be singular (degenerate
    posterior covariance, ``Sigma_xi = 0``), and LAPACK returns ``nan`` there.
    """
    ev, P = jnp.linalg.eigh(0.5 * (A + A.T))
    return (P * jnp.sqrt(jnp.clip(ev, 0.0))) @ P.T


@jaxtyped(typechecker=beartype)
def _expected_quadratic(
    Jacs: Float[Array, "n_samples a b"],
    chol: tuple[Float[Array, "a a"], bool],
) -> Float[Array, "b b"]:
    """``E[J^T M^{-1} J]`` over the sample, ``M`` given by its factorization."""

    def quad(J: Float[Array, "a b"]) -> Float[Array, "b b"]:
        return J.T @ jsp.linalg.cho_solve(chol, J)

    out = jnp.mean(jax.vmap(quad)(Jacs), axis=0)
    return 0.5 * (out + out.T)


# =============================================================================
# Moments -- the expensive part: N Jacobians of the forward model
# =============================================================================


@partial(jax.jit, static_argnums=(0,))
@jaxtyped(typechecker=beartype)
def expected_jacobian_moments(
    u,
    etas: Float[Array, "n_samples n_eta"],
    Sigma_obs: Float[Array, "n_obs n_obs"],
) -> tuple[Float[Array, "n_eta n_obs"], Float[Array, "n_eta n_eta"]]:
    r"""``(L(u), H(u))`` -- equations (31)-(32).

    Parameters
    ----------
    u : Callable
        Forward model ``eta -> observations``, without a design.
    etas : Float[Array, "n_samples n_eta"]
        Draws from the prior.
    Sigma_obs : Float[Array, "n_obs n_obs"]

    Notes
    -----
    ``H(u)`` is computed in **two passes** (mean, then centered quadratic)
    rather than via ``E[J^T S^{-1} J] - Jbar^T S^{-1} Jbar``. The two are
    algebraically equal, but the latter loses all precision when the mean
    dominates the variance -- in particular when the Jacobian is constant,
    where ``H(u)`` must be **exactly** zero.

    ``O(N p q)`` memory: the Jacobians are materialized.
    """
    Jacs = jax.vmap(jax.jacfwd(u))(etas)
    J_bar = jnp.mean(Jacs, axis=0)
    chol_obs = jsp.linalg.cho_factor(Sigma_obs, lower=True)
    return J_bar.T, _expected_quadratic(Jacs - J_bar, chol_obs)


@partial(jax.jit, static_argnums=(0,))
@jaxtyped(typechecker=beartype)
def qoi_fisher_moment(
    h,
    etas: Float[Array, "n_samples n_eta"],
    Sigma_xi: Float[Array, "n_param n_param"],
) -> Float[Array, "n_eta n_eta"]:
    r"""``J(h) = E[Jac h^T Sigma_xi^{-1} Jac h]`` -- equation (33).

    Cheap: ``h`` is explicit, no forward model to evaluate.

    ``Sigma_xi`` must be strictly positive definite. As ``Sigma_xi -> 0``,
    ``J(h) -> inf``: the case ``xi = 0`` is not reachable here, it belongs to
    :func:`gradient_diagnostics_standard`.
    """
    Jacs_h = jax.vmap(jax.jacfwd(h))(etas)
    chol_xi = jsp.linalg.cho_factor(Sigma_xi, lower=True)
    return _expected_quadratic(Jacs_h, chol_xi)


@partial(jax.jit, static_argnums=(0,))
@jaxtyped(typechecker=beartype)
def fisher_information_prior(prior_eta: Prior) -> Float[Array, "n_eta n_eta"]:
    r"""``I_eta = Cov(grad log pi(eta))`` -- equation (34), Gaussian case.

    For a Gaussian prior, ``grad log pi = -Gamma^{-1}(eta - m)``, hence
    ``Cov(grad log pi) = Gamma^{-1} Gamma Gamma^{-1} = Gamma^{-1}``. Exact, no
    sampling needed.

    See :func:`fisher_information_prior_mc` for an arbitrary prior.
    """
    q = prior_eta.mu.shape[0]
    return prior_eta.prior_precision_matmul(jnp.eye(q, dtype=prior_eta.mu.dtype))


@partial(jax.jit, static_argnums=(0, 2))
@jaxtyped(typechecker=beartype)
def fisher_information_prior_mc(
    prior_eta: Prior, key: PRNGKeyArray, n_samples: int
) -> Float[Array, "n_eta n_eta"]:
    """``I_eta`` via the empirical covariance of the scores. Valid for any prior.

    Oracle of :func:`fisher_information_prior`: their agreement proves that
    the Gaussian assumption holds.
    """
    scores = jax.vmap(prior_eta.grad_log_prior)(prior_eta.sample(key, n_samples))
    centered = scores - jnp.mean(scores, axis=0)
    out = centered.T @ centered / (n_samples - 1)
    return 0.5 * (out + out.T)


# =============================================================================
# Assembly -- cheap: q x q linear algebra
# =============================================================================


@jax.jit
@jaxtyped(typechecker=beartype)
def assemble(
    L: Float[Array, "n_eta n_obs"],
    A: Float[Array, "n_eta n_eta"],
    Sigma_obs: Float[Array, "n_obs n_obs"],
) -> Float[Array, "n_obs n_obs"]:
    r"""``Sigma_obs + L^T A^{-1} L``, via ``cho_solve``. ``A`` SDP.

    Common form for (35) and (36): only ``A`` changes.
    ``A = H + I_eta`` -> ``Sigma_signal``. ``A = H + I_eta + J(h)`` -> ``Sigma_noise``.
    """
    chol = jsp.linalg.cho_factor(A, lower=True)
    out = Sigma_obs + L.T @ jsp.linalg.cho_solve(chol, L)
    return 0.5 * (out + out.T)


@jax.jit
@jaxtyped(typechecker=beartype)
def assemble_misfit(
    L: Float[Array, "n_eta n_obs"],
    H: Float[Array, "n_eta n_eta"],
    Sigma_eta: Float[Array, "n_eta n_eta"],
    Sigma_obs: Float[Array, "n_obs n_obs"],
    extra: Float[Array, "n_eta n_eta"] | None = None,
) -> Float[Array, "n_obs n_obs"]:
    r"""Same, **preconditioned by the prior** -- without forming ``Sigma_eta^{-1}``.

    .. math::
        (H + \Sigma_\eta^{-1})^{-1}
        = \Sigma_\eta^{1/2}(I + \Sigma_\eta^{1/2} H \Sigma_\eta^{1/2})^{-1}
          \Sigma_\eta^{1/2}

    Path **independent** of :func:`assemble` with ``A = H + I_eta``: an oracle.

    Parameters
    ----------
    extra : Float[Array, "n_eta n_eta"] | None
        Additional term in ``A`` (``J(h)`` for ``Sigma_noise``). The
        preconditioning identity only applies to ``Sigma_eta^{-1}``: ``extra``
        is therefore absorbed into the Hessian, ``H + extra`` playing the role
        of ``H``.

    Notes
    -----
    Only uses ``Sigma_eta^{1/2}``, never ``Sigma_eta^{-1}``. This is what will
    survive once the prior precision can no longer be formed.

    ⚠️ ``A_mis`` **is already the inverse**: the term is ``A_mis @ L``, not
    ``solve(A_mis, L)``. The NumPy prototype did the latter -- a 248% relative
    error, and the result stayed SDP, so it went unnoticed.
    """
    Hx = H if extra is None else H + extra
    S_sqrt = psd_sqrt(Sigma_eta)
    H_mis = S_sqrt @ Hx @ S_sqrt
    n = H_mis.shape[0]
    A_inv = S_sqrt @ jnp.linalg.solve(jnp.eye(n, dtype=H_mis.dtype) + H_mis, S_sqrt)
    out = Sigma_obs + L.T @ (A_inv @ L)
    return 0.5 * (out + out.T)


# =============================================================================
# Orchestration
# =============================================================================


@partial(jax.jit, static_argnums=(0, 1, 2, 6))
@jaxtyped(typechecker=beartype)
def gradient_diagnostics(
    u,
    h,
    prior_eta: Prior,
    Sigma_obs: Float[Array, "n_obs n_obs"],
    Sigma_xi: Float[Array, "n_param n_param"],
    key: PRNGKeyArray,
    n_samples: int,
) -> tuple[Float[Array, "n_obs n_obs"], Float[Array, "n_obs n_obs"]]:
    r"""``(Sigma_signal, Sigma_noise)`` in the goal-oriented setting -- Prop. 4.

    Satisfy ``Sigma_signal^{-1} ⪰ I_Y`` and ``Sigma_noise^{-1} ⪰ E[I_{Y|theta}]``.

    Returns **two of the four** matrices: ``Sigma_Y`` and
    ``Sigma_Y_given_theta`` come from §3.1, with no alternative.
    """
    etas = prior_eta.sample(key, n_samples)
    L, H = expected_jacobian_moments(u, etas, Sigma_obs)
    J_h = qoi_fisher_moment(h, etas, Sigma_xi)
    I_eta = fisher_information_prior(prior_eta)
    return assemble(L, H + I_eta, Sigma_obs), assemble(L, H + I_eta + J_h, Sigma_obs)


@partial(jax.jit, static_argnums=(0, 1, 4))
@jaxtyped(typechecker=beartype)
def gradient_diagnostics_standard(
    u,
    prior_theta: Prior,
    Sigma_obs: Float[Array, "n_obs n_obs"],
    key: PRNGKeyArray,
    n_samples: int,
) -> tuple[Float[Array, "n_obs n_obs"], Float[Array, "n_obs n_obs"]]:
    r"""``(Sigma_signal, Sigma_noise)`` in the standard setting ``Y = u(theta) + eps``.

    Prop. 2 gives ``E[I_{Y|theta}] = Sigma_obs^{-1}``, hence
    ``Sigma_noise = Sigma_obs`` **exactly** -- posited, not computed.

    This is **not** :func:`gradient_diagnostics` with ``h = id`` and a tiny
    ``Sigma_xi``: that limit is singular (``J(h) = Sigma_xi^{-1} -> inf``),
    there remains an ``O(Sigma_xi)`` term, and the enclosure flips.
    """
    thetas = prior_theta.sample(key, n_samples)
    L, H = expected_jacobian_moments(u, thetas, Sigma_obs)
    I_theta = fisher_information_prior(prior_theta)
    return assemble(L, H + I_theta, Sigma_obs), Sigma_obs
