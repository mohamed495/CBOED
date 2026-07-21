"""EIG via variational NMC with a Laplace proposal."""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array
from jaxtyping import Float, Int, PRNGKeyArray

from cboed.estimators.base import EIGEstimator


class LaplaceProposal:
    r"""``q(. | y) = N(mu_post(y), Gamma_post)`` -- importance proposal.

    What VNMC needs from a proposal: to **sample** it and to **evaluate its
    density**. Neither requires ``Gamma_post`` itself: these are actions,
    hence a separate object from the :class:`InferenceModel` contract.

    The point is to hoist the factorization. ``Gamma_post^{-1}`` depends
    neither on ``y`` nor on anything that varies in the outer loop
    (``theta_lin`` and ``design`` are fixed): it is factored **once** at
    construction. Otherwise every call to ``_mu`` would redo the same
    Cholesky -- ``n_outer`` times for the same matrix, and ``vmap`` does not
    factor identical computations across vmapped instances.

    Parameters
    ----------
    inference : InferenceModel
        Must expose ``prior``, ``likelihood`` and ``_posterior_chol``.
    theta_lin : Float[Array, " n_param"]
        Linearization point. Exact in LG; at ``lambda > 0`` the MAP would be
        better -- that is an argument, not a decision made by this object.
    design : Int[Array, " n_sensors"] | None
    """

    def __init__(self, inference, theta_lin, design=None) -> None:
        self._inference = inference
        self._theta_lin = theta_lin
        self._design = design
        # ONE factorization, reused across the whole outer loop.
        self._chol = inference._posterior_chol(theta_lin, design)
        cov = jsp.linalg.cho_solve(self._chol, jnp.eye(theta_lin.shape[0], dtype=theta_lin.dtype))
        self._L = jnp.linalg.cholesky(cov)

    def mean(self, y: Float[Array, " n_sensors"]) -> Float[Array, " n_param"]:
        """``mu_post(y)`` -- a single solve, using the stored factorization."""
        grad = self._inference.likelihood.grad_log_likelihood(
            y=y, theta=self._theta_lin, design=self._design
        )
        return self._inference.prior.mu + jsp.linalg.cho_solve(self._chol, grad)

    def sample(
        self, key: PRNGKeyArray, y: Float[Array, " n_sensors"], n_samples: int
    ) -> Float[Array, "n_samples n_param"]:
        mu_q = self.mean(y)
        z = jax.random.normal(key, (n_samples, mu_q.shape[0]))
        return mu_q + z @ self._L.T

    def log_pdf(
        self, theta_s: Float[Array, " n_param"], y: Float[Array, " n_sensors"]
    ) -> Float[Array, ""]:
        """``log q(theta_s | y)``."""
        return self._log_gaussian(theta_s, self.mean(y), self._L)

    @staticmethod
    def _log_gaussian(theta, mu, L) -> Float[Array, ""]:
        """``log N(theta; mu, L L^T)``."""
        d = mu.shape[0]
        r = jsp.linalg.solve_triangular(L, theta - mu, lower=True)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        return -0.5 * (d * jnp.log(2 * jnp.pi) + logdet + r @ r)


class VariationalNMCEIG(EIGEstimator):
    r"""EIG via variational NMC.

    An **upper** bound, like NMC, but tighter when ``q`` approximates the
    posterior well (the bias equals ``E[log p(y) - log p_hat]``, controlled
    by the variance of the importance weights).

    In LG, the Laplace proposal is the **exact** posterior -> constant
    weights -> zero variance -> ``VNMC = EIG``.

    For a bracketing bound, pair with PCE (**lower** bound):
    ``PCE <= EIG <= VNMC``. NMC and VNMC are on the **same side** -- their
    difference does not measure a bracket.

    Notes
    -----
    ``VNMC <= NMC`` is not a theorem: it holds when ``q`` beats the prior
    (the LG case), not universally.
    """

    @property
    def likelihood(self):
        return self._hyperparameters["likelihood"]

    @property
    def prior(self):
        return self._hyperparameters["prior"]

    @property
    def inference(self):
        return self._hyperparameters["inference"]

    def estimate(
        self,
        key: PRNGKeyArray,
        design: Int[Array, " n_sensors"] | None = None,
        n_outer: int = 1000,
        n_inner: int = 1000,
    ) -> Float[Array, ""]:
        k_theta, k_y, k_prop = jax.random.split(key, 3)

        # -- outer loop: (theta_i, y_i) ~ p(theta) p(y | theta) ------------
        thetas = self.prior.sample(k_theta, n_outer)
        keys_y = jax.random.split(k_y, n_outer)
        ys = jax.vmap(lambda th, k: self.likelihood.sample(k, th, design, n_samples=1)[0])(
            thetas, keys_y
        )

        log_lik_matched = jax.vmap(lambda y, th: self.likelihood.log_likelihood(y, th, design))(
            ys, thetas
        )

        # -- proposal: built ONCE ------------------------------------------
        proposal = LaplaceProposal(self.inference, self.prior.mu, design)
        keys_prop = jax.random.split(k_prop, n_outer)

        def log_marginal(y, k):
            theta_prop = proposal.sample(k, y, n_inner)
            log_lik = jax.vmap(lambda th: self.likelihood.log_likelihood(y, th, design))(theta_prop)
            log_prior = jax.vmap(self.prior.log_prior)(theta_prop)
            log_q = jax.vmap(lambda th: proposal.log_pdf(th, y))(theta_prop)

            # logsumexp, never log(mean(exp)): guaranteed overflow otherwise.
            log_weights = log_lik + log_prior - log_q
            return jsp.special.logsumexp(log_weights) - jnp.log(n_inner)

        log_marg = jax.vmap(log_marginal)(ys, keys_prop)
        return jnp.mean(log_lik_matched - log_marg)
