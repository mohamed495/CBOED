# estimators/pce.py
"""EIG via Prior Contrastive Estimation."""

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from cboed.estimators.base import EIGEstimator


class PriorContrastiveEIG(EIGEstimator):
    """Estimate EIG via Prior Contrastive Estimation (PCE, Foster et al. 2020).

    Notes
    -----
    Biased **lower** bound of the true EIG. Differs from
    :class:`~cboed.estimators.nmc.NestedMonteCarloEIG` only by including
    ``theta_i`` (the parameter that generated ``y_i``) in the inner
    contrastive sum used to estimate ``log p(y_i)``. Together with VNMC
    (:class:`~cboed.estimators.vnmc.VariationalNMCEIG`, an **upper** bound),
    it brackets the EIG: ``PCE <= EIG <= VNMC``.

    Examples
    --------
    >>> pce = PriorContrastiveEIG(likelihood=likelihood, prior=prior)  # doctest: +SKIP
    >>> pce.estimate(jax.random.key(0), design, n_outer=1000, n_inner=1000)  # doctest: +SKIP
    """

    @property
    def likelihood(self):
        """The :class:`~cboed.likelihood.base.Likelihood`, ``p(y | theta, design)``."""
        return self._hyperparameters["likelihood"]

    @property
    def prior(self):
        """The :class:`~cboed.priors.base.Prior` on ``theta``."""
        return self._hyperparameters["prior"]

    def estimate(self, key, design=None, n_outer=1000, n_inner=1000):
        """Estimate the EIG by Prior Contrastive Estimation.

        Parameters
        ----------
        key : PRNGKeyArray
            Source of randomness, split into the outer draws
            ``theta_i ~ prior``, ``y_i ~ p(.|theta_i)`` and the contrastive
            inner draws ``theta_j ~ prior``.
        design : Int[Array, " n_obs"] or None, optional
            Indices of the observed sensors. ``None`` means the full field
            is observed.
        n_outer : int, optional
            Number of outer samples ``(theta_i, y_i)``. Default 1000.
        n_inner : int, optional
            Number of contrastive samples ``theta_j`` added to ``theta_i``
            itself when estimating each marginal ``log p(y_i)``. Default
            1000.

        Returns
        -------
        Float[Array, ""]
            Mean over the ``n_outer`` outer samples of
            ``log p(y_i|theta_i) - log p_hat(y_i)``, where ``p_hat(y_i)`` is
            the ``(n_inner + 1)``-term contrastive estimate including
            ``theta_i``: a biased-low Monte Carlo estimate of the EIG.
        """
        k_theta, k_y, k_inner = jax.random.split(key, 3)

        thetas = self.prior.sample(k_theta, n_outer)
        keys_y = jax.random.split(k_y, n_outer)
        ys = jax.vmap(lambda th, k: self.likelihood.sample(k, th, design, n_samples=1)[0])(
            thetas, keys_y
        )

        log_lik_matched = jax.vmap(lambda y, th: self.likelihood.log_likelihood(y, th, design))(
            ys, thetas
        )

        thetas_inner = self.prior.sample(k_inner, n_inner)

        def log_marginal(y, theta_i, ll_i):
            lls = jax.vmap(lambda th: self.likelihood.log_likelihood(y, th, design))(thetas_inner)
            # <- THE DIFFERENCE: include θᵢ in the contrast
            all_lls = jnp.concatenate([jnp.array([ll_i]), lls])
            return jsp.special.logsumexp(all_lls) - jnp.log(n_inner + 1)

        log_marg = jax.vmap(log_marginal)(ys, thetas, log_lik_matched)
        return jnp.mean(log_lik_matched - log_marg)
