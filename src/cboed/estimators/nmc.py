# estimators/nmc.py
"""EIG via standard nested Monte Carlo."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from cboed.estimators.base import EIGEstimator, chunked_vmap


class NestedMonteCarloEIG(EIGEstimator):
    """Estimate EIG via nested Monte Carlo.

    Reference estimator for the nonlinear case: it makes no assumption on
    the forward model beyond being sampleable, with a tractable
    log-likelihood.

    Notes
    -----
    Biased **upper** bound of the true EIG, with bias in ``O(1/n_inner)``:
    ``log p(y)`` is estimated by ``logsumexp`` over ``n_inner`` prior draws,
    and by Jensen's inequality this nested log-mean-exp estimate
    underestimates ``log p(y)`` in expectation; since it enters the EIG with
    a minus sign, the resulting EIG estimate is biased high. The bias
    vanishes as ``n_inner -> infinity``.

    Cost: ``n_outer * n_inner`` likelihood evaluations.

    Examples
    --------
    >>> nmc = NestedMonteCarloEIG(likelihood=likelihood, prior=prior)  # doctest: +SKIP
    >>> nmc.estimate(jax.random.key(0), design, n_outer=1000, n_inner=1000)  # doctest: +SKIP
    """

    @property
    def likelihood(self):
        """The :class:`~cboed.likelihood.base.Likelihood`, ``p(y | theta, design)``."""
        return self._hyperparameters["likelihood"]

    @property
    def prior(self):
        """The :class:`~cboed.priors.base.Prior` on ``theta``."""
        return self._hyperparameters["prior"]

    def estimate(
        self,
        key: PRNGKeyArray,
        design: Int[Array, " n_obs"] | None = None,
        n_outer: int = 1000,
        n_inner: int = 1000,
        chunk_size: int | None = None,
    ) -> Float[Array, ""]:
        """Estimate the EIG by nested Monte Carlo.

        Parameters
        ----------
        key : PRNGKeyArray
            Source of randomness, split into the outer draws
            ``theta_i ~ prior``, ``y_i ~ p(.|theta_i)`` and the inner draws
            ``theta_j ~ prior`` used for the marginal estimate.
        design : Int[Array, " n_obs"] or None, optional
            Indices of the observed sensors. ``None`` means the full field
            is observed.
        n_outer : int, optional
            Number of outer samples ``(theta_i, y_i)``. Default 1000.
        n_inner : int, optional
            Number of inner samples ``theta_j`` used to estimate each
            marginal ``log p(y_i)`` by ``logsumexp``. Default 1000.
        chunk_size : int or None, optional
            Forwarded to :func:`~cboed.estimators.base.chunked_vmap` for the
            marginal loop, to bound peak memory. ``None`` (default) uses a
            single ``vmap``.

        Returns
        -------
        Float[Array, ""]
            Mean over the ``n_outer`` outer samples of
            ``log p(y_i|theta_i) - log p_hat(y_i)``: a biased-high Monte
            Carlo estimate of ``EIG = E[log p(y|theta) - log p(y)]``, with
            ``O(1/n_inner)`` bias.
        """
        k_theta, k_y, k_inner = jax.random.split(key, 3)

        # outer loop: θᵢ ~ prior,  yᵢ ~ p(·|θᵢ)
        thetas = self.prior.sample(k_theta, n_outer)  # (N, d)
        keys_y = jax.random.split(k_y, n_outer)
        ys = jax.vmap(lambda th, k: self.likelihood.sample(k, th, design, n_samples=1)[0])(
            thetas, keys_y
        )  # (N, m)

        # log p(yᵢ | θᵢ)
        log_lik_matched = jax.vmap(lambda y, th: self.likelihood.log_likelihood(y, th, design))(
            ys, thetas
        )  # (N,)

        # log p̂(yᵢ) = logmeanexp_j log p(yᵢ | θⱼ),  θⱼ ~ prior
        thetas_inner = self.prior.sample(k_inner, n_inner)  # (M, d)

        def log_marginal(y):
            lls = jax.vmap(lambda th: self.likelihood.log_likelihood(y, th, design))(
                thetas_inner
            )  # (M,)
            return jax.scipy.special.logsumexp(lls) - jnp.log(n_inner)

        log_marg = chunked_vmap(log_marginal, ys, chunk_size=chunk_size)  # (N,)

        return jnp.mean(log_lik_matched - log_marg)
