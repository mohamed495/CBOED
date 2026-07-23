"""EIG via variational NMC with a Laplace proposal."""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array
from jaxtyping import Float, Int, PRNGKeyArray

from cboed.estimators.base import EIGEstimator


class LaplaceProposal:
    r"""Gaussian importance proposal ``q(. | y) = N(mu_post(y), Gamma_post)``.

    What VNMC needs from a proposal: to **sample** it and to **evaluate its
    density**. Neither requires ``Gamma_post`` itself: these are actions,
    hence a separate object from the
    :class:`~cboed.inference.base.InferenceModel` contract.

    Parameters
    ----------
    inference : InferenceModel
        Must expose ``prior``, ``likelihood`` and ``_posterior_chol``.
    theta_lin : Float[Array, " n_param"]
        Linearization point. Exact in the linear-Gaussian case; at
        ``lambda > 0`` the MAP would be better -- that is an argument, not a
        decision made by this object. The covariance factorization is valid
        at any ``theta_lin``, but :meth:`mean` additionally assumes
        ``theta_lin == inference.prior.mu`` (see that method) -- the only
        way this class is currently instantiated.
    design : Int[Array, " n_sensors"] or None, optional
        Indices of the observed sensors. ``None`` means the full field is
        observed.

    Notes
    -----
    The point is to hoist the factorization. ``Gamma_post^{-1}`` depends
    neither on ``y`` nor on anything that varies in the outer loop
    (``theta_lin`` and ``design`` are fixed): it is factored **once** at
    construction. Otherwise every call to :meth:`mean` would redo the same
    Cholesky -- ``n_outer`` times for the same matrix, and ``vmap`` does not
    factor identical computations across vmapped instances.
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
        """Compute ``mu_post(y)`` with a single solve against the stored factorization.

        Parameters
        ----------
        y : Float[Array, " n_sensors"]
            Observation.

        Returns
        -------
        Float[Array, " n_param"]
            Posterior mean at ``y``, under the Laplace/linear-Gaussian
            approximation.

        Notes
        -----
        Computed as ``prior.mu + Gamma_post @ grad_log_likelihood(theta_lin)``,
        which omits the ``grad_log_prior(theta_lin)`` term present in the
        general one-step correction (cf. :meth:`~cboed.inference.linear_model.LinearModel._mu`).
        That term vanishes only at ``theta_lin = prior.mu``, so this
        shortcut is the exact posterior mean precisely when the proposal was
        built at the prior mean -- true of every current call site -- and
        would need the omitted term reinstated for a general ``theta_lin``
        (e.g. a MAP point).
        """
        grad = self._inference.likelihood.grad_log_likelihood(
            y=y, theta=self._theta_lin, design=self._design
        )
        return self._inference.prior.mu + jsp.linalg.cho_solve(self._chol, grad)

    def sample(
        self, key: PRNGKeyArray, y: Float[Array, " n_sensors"], n_samples: int
    ) -> Float[Array, "n_samples n_param"]:
        """Draw samples from the proposal ``q(. | y)``.

        Parameters
        ----------
        key : PRNGKeyArray
            Source of randomness.
        y : Float[Array, " n_sensors"]
            Observation.
        n_samples : int
            Number of samples to draw.

        Returns
        -------
        Float[Array, "n_samples n_param"]
            Samples ``mu_q(y) + L z``, ``z ~ N(0, I)``, ``L`` the stored
            Cholesky factor of the proposal covariance.
        """
        mu_q = self.mean(y)
        z = jax.random.normal(key, (n_samples, mu_q.shape[0]))
        return mu_q + z @ self._L.T

    def log_pdf(
        self, theta_s: Float[Array, " n_param"], y: Float[Array, " n_sensors"]
    ) -> Float[Array, ""]:
        """Evaluate ``log q(theta_s | y)``.

        Parameters
        ----------
        theta_s : Float[Array, " n_param"]
            Point at which to evaluate the proposal density.
        y : Float[Array, " n_sensors"]
            Observation.

        Returns
        -------
        Float[Array, ""]
            Log-density of the Gaussian proposal at ``theta_s``.
        """
        return self._log_gaussian(theta_s, self.mean(y), self._L)

    @staticmethod
    def _log_gaussian(theta, mu, L) -> Float[Array, ""]:
        """Evaluate ``log N(theta; mu, L L^T)`` from a Cholesky factor ``L``."""
        d = mu.shape[0]
        r = jsp.linalg.solve_triangular(L, theta - mu, lower=True)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        return -0.5 * (d * jnp.log(2 * jnp.pi) + logdet + r @ r)


class VariationalNMCEIG(EIGEstimator):
    r"""Estimate EIG via variational nested Monte Carlo (VNMC).

    Notes
    -----
    An **upper** bound, like NMC, but tighter when ``q`` approximates the
    posterior well (the bias equals ``E[log p(y) - log p_hat]``, controlled
    by the variance of the importance weights).

    In the linear-Gaussian case, the Laplace proposal is the **exact**
    posterior -> constant weights -> zero variance -> ``VNMC = EIG``.

    For a bracketing bound, pair with PCE
    (:class:`~cboed.estimators.pce.PriorContrastiveEIG`, a **lower** bound):
    ``PCE <= EIG <= VNMC``. NMC and VNMC are on the **same side** -- their
    difference does not measure a bracket.

    ``VNMC <= NMC`` is not a theorem: it holds when ``q`` beats the prior
    (the linear-Gaussian case), not universally.

    Examples
    --------
    >>> vnmc = VariationalNMCEIG(
    ...     likelihood=likelihood, prior=prior, inference=inference
    ... )  # doctest: +SKIP
    >>> vnmc.estimate(jax.random.key(0), design, n_outer=1000, n_inner=1000)  # doctest: +SKIP
    """

    @property
    def likelihood(self):
        """The :class:`~cboed.likelihood.base.Likelihood`, ``p(y | theta, design)``."""
        return self._hyperparameters["likelihood"]

    @property
    def prior(self):
        """The :class:`~cboed.priors.base.Prior` on ``theta``."""
        return self._hyperparameters["prior"]

    @property
    def inference(self):
        """The :class:`~cboed.inference.base.InferenceModel` used to build the Laplace proposal."""
        return self._hyperparameters["inference"]

    def estimate(
        self,
        key: PRNGKeyArray,
        design: Int[Array, " n_sensors"] | None = None,
        n_outer: int = 1000,
        n_inner: int = 1000,
    ) -> Float[Array, ""]:
        """Estimate the EIG by variational nested Monte Carlo.

        Parameters
        ----------
        key : PRNGKeyArray
            Source of randomness, split into the outer draws
            ``theta_i ~ prior``, ``y_i ~ p(.|theta_i)`` and the proposal
            draws ``theta_j ~ q(. | y_i)``.
        design : Int[Array, " n_sensors"] or None, optional
            Indices of the observed sensors. ``None`` means the full field
            is observed.
        n_outer : int, optional
            Number of outer samples ``(theta_i, y_i)``. Default 1000.
        n_inner : int, optional
            Number of importance samples drawn from the Laplace proposal
            ``q(. | y_i)`` to estimate each marginal ``log p(y_i)``. Default
            1000.

        Returns
        -------
        Float[Array, ""]
            Mean over the ``n_outer`` outer samples of
            ``log p(y_i|theta_i) - log p_hat(y_i)``, where ``p_hat(y_i)`` is
            a self-normalized importance-sampling estimate of ``p(y_i)``
            under the Laplace proposal built once at ``mu_prior``: a
            biased-high Monte Carlo estimate of the EIG.

        Notes
        -----
        The proposal is built **once** (at ``mu_prior``, shared by every
        outer sample) rather than once per outer sample, since its
        factorization does not depend on ``y``. The importance weights are
        combined via ``logsumexp``, never ``log(mean(exp(.)))``, to avoid
        guaranteed overflow.
        """
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
