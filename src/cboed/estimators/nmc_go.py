r"""Estimate goal-oriented EIG via nested Monte Carlo, ``theta = B eta + xi``.

:class:`~cboed.estimators.nmc.NestedMonteCarloEIG` estimates ``EIG(eta)``: its
``log p(y|theta)`` is directly ``likelihood.log_likelihood(y, theta, ...)``
because ``theta`` there **is the forward-model parameter itself**. Here
``theta`` is a lower-dimensional linear projection ``B eta + xi`` (the QoI):
``p(y|theta)`` no longer has a closed form, it is an expectation over
``eta | theta`` -- a second level of nested Monte Carlo.

.. math::
    \mathrm{EIG}(\theta) = E_{\theta, y}\left[
        \log p(y|\theta) - \log p(y)
    \right], \qquad
    p(y|\theta) = E_{\eta|\theta}[p(y|\eta)]

``eta | theta`` is Gaussian in closed form (Rem. 3.1, the same computation as
:func:`cboed.bounds.diagnostics.sample_based.sample_Sigma_Y_given_theta`) --
valid only for linear (constant) ``B``, and without it, no closed form exists
for drawing ``eta | theta``.

``log p(y)`` remains the usual marginal (independent of ``theta``): the same
nested estimator as in :mod:`cboed.estimators.nmc`.

Notes
-----
Cost: ``n_outer * (n_inner_theta + n_inner_marginal)`` evaluations of
``log_likelihood`` (hence of the forward model) -- one notch more expensive
than standard NMC, due to the second level of nesting.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from cboed.bounds.diagnostics.gradient_based import psd_sqrt
from cboed.estimators.base import EIGEstimator, chunked_vmap


def _eta_given_theta_params(prior_eta, B: Array, Sigma_xi: Array):
    r"""Compute the closed-form conditional ``eta | theta ~ N(mean_fn(theta), Sigma_pos)``
      (Rem. 3.1, ``B`` constant).

    Parameters
    ----------
    prior_eta : Prior
        Prior on the latent field ``eta``.
    B : Array
        Constant Jacobian of the QoI projection ``theta = B eta + xi``.
    Sigma_xi : Array
        Covariance of the QoI noise ``xi``.

    Returns
    -------
    mean_fn : Callable
        Maps a value of ``theta`` to the conditional mean ``E[eta | theta]``,
        via the Kalman gain ``K``.
    L_pos : Array
        PSD square root (Cholesky-like factor) of the conditional covariance
        ``Sigma_pos``, which does not depend on ``theta``.

    Notes
    -----
    Covariance independent of ``theta`` (factored once); only the mean
    depends on it. Same computation as in ``sample_Sigma_Y_given_theta``,
    exposed here to be reused without duplicating the Kalman step.
    """
    Sigma_eta = prior_eta.Sigma()
    m_eta = prior_eta.mu
    S = B @ Sigma_eta @ B.T + Sigma_xi
    K = jnp.linalg.solve(S, B @ Sigma_eta).T  # (q, d)
    Sigma_pos = Sigma_eta - K @ B @ Sigma_eta
    L_pos = psd_sqrt(0.5 * (Sigma_pos + Sigma_pos.T))

    def mean_fn(theta):
        return m_eta + (theta - m_eta @ B.T) @ K.T

    return mean_fn, L_pos


class GoalOrientedNestedMonteCarloEIG(EIGEstimator):
    r"""Estimate ``EIG(theta)`` via nested Monte Carlo, ``theta = B eta + xi`` (Rem. 3.1).

    Parameters
    ----------
    likelihood : Likelihood
        Likelihood ``p(y | eta, design)`` on the **full** field.
    prior_eta : Prior
        Prior on ``eta`` (the full field, not the QoI).
    B : Float[Array, "n_qoi n_eta"]
        Jacobian of ``h`` (QoI projection), constant.
    Sigma_xi : Float[Array, "n_qoi n_qoi"]
        Covariance of the noise ``xi``. Cf. the ``bounds`` modules:
        ``Sigma_xi -> 0`` is a singular limit, do not go there.

    Notes
    -----
    Warning: bias in both nested terms (``log p(y|theta)`` **and**
    ``log p(y)``), each underestimated at finite ``n_inner`` (same mechanism
    as :class:`~cboed.estimators.nmc.NestedMonteCarloEIG` -- Jensen on the
    ``logsumexp``). The net bias on the EIG has no guaranteed sign a priori.

    Examples
    --------
    >>> go_nmc = GoalOrientedNestedMonteCarloEIG(
    ...     likelihood=likelihood, prior_eta=prior_eta, B=B, Sigma_xi=Sigma_xi
    ... )  # doctest: +SKIP
    >>> go_nmc.estimate(jax.random.key(0), design, n_outer=1000)  # doctest: +SKIP
    """

    @property
    def likelihood(self):
        """The :class:`~cboed.likelihood.base.Likelihood`, ``p(y | eta, design)``."""
        return self._hyperparameters["likelihood"]

    @property
    def prior_eta(self):
        """The :class:`~cboed.priors.base.Prior` on the latent field ``eta``."""
        return self._hyperparameters["prior_eta"]

    @property
    def B(self) -> Float[Array, "n_qoi n_eta"]:
        """Constant Jacobian of the QoI projection ``theta = B eta + xi``."""
        return self._hyperparameters["B"]

    @property
    def Sigma_xi(self) -> Float[Array, "n_qoi n_qoi"]:
        """Covariance of the QoI noise ``xi``."""
        return self._hyperparameters["Sigma_xi"]

    def estimate(
        self,
        key: PRNGKeyArray,
        design: Int[Array, " n_obs"] | None = None,
        n_outer: int = 1000,
        n_inner_theta: int = 500,
        n_inner_marginal: int = 1000,
        chunk_size: int | None = None,
    ) -> Float[Array, ""]:
        """Estimate ``EIG(theta)`` by two-level nested Monte Carlo.

        Parameters
        ----------
        key : PRNGKeyArray
            Source of randomness, split into the outer draws
            (``eta_i``, ``xi_i``, ``y_i``) and the two inner draws
            (``eta' | theta_i`` and ``eta ~ prior_eta``).
        design : Int[Array, " n_obs"] or None, optional
            Indices of the observed sensors. ``None`` means the full field
            is observed.
        n_outer : int, optional
            Number of outer samples ``(eta_i, theta_i, y_i)``. Default 1000.
        n_inner_theta : int, optional
            Number of inner samples ``eta' ~ eta | theta_i`` used to
            estimate ``log p(y_i | theta_i)`` (first nesting level,
            Rem. 3.1). Default 500.
        n_inner_marginal : int, optional
            Number of inner samples ``eta ~ prior_eta`` used to estimate the
            marginal ``log p(y_i)`` (second nesting level, as in standard
            NMC). Default 1000.
        chunk_size : int or None, optional
            Forwarded to :func:`~cboed.estimators.base.chunked_vmap` for both
            inner loops, to bound peak memory. ``None`` (default) uses a
            single ``vmap``.

        Returns
        -------
        Float[Array, ""]
            Mean over the ``n_outer`` outer samples of
            ``log p(y_i|theta_i) - log p(y_i)``, a Monte Carlo estimate of
            ``EIG(theta)`` whose bias has no guaranteed sign a priori (both
            nested terms are individually underestimated, see Notes).
        """
        k_eta, k_xi, k_y, k_theta_inner, k_marg_inner = jax.random.split(key, 5)

        mean_fn, L_pos = _eta_given_theta_params(self.prior_eta, self.B, self.Sigma_xi)
        n_eta = self.prior_eta.mu.shape[0]

        # -- outer loop: eta_i ~ prior, theta_i = B eta_i + xi_i, y_i ~ p(.|eta_i) --
        eta_outer = self.prior_eta.sample(k_eta, n_outer)
        L_xi = psd_sqrt(self.Sigma_xi)
        z_xi = jax.random.normal(k_xi, (n_outer, self.Sigma_xi.shape[0]))
        theta_outer = eta_outer @ self.B.T + z_xi @ L_xi.T
        keys_y = jax.random.split(k_y, n_outer)
        y_outer = jax.vmap(lambda eta, k: self.likelihood.sample(k, eta, design, n_samples=1)[0])(
            eta_outer, keys_y
        )

        # -- log p(y_i | theta_i): nested MC over eta' | theta_i (Rem. 3.1) --
        def log_lik_given_theta(y, theta, k):
            z = jax.random.normal(k, (n_inner_theta, n_eta))
            etas_cond = mean_fn(theta) + z @ L_pos.T
            lls = jax.vmap(lambda e: self.likelihood.log_likelihood(y, e, design))(etas_cond)
            return jsp.special.logsumexp(lls) - jnp.log(n_inner_theta)

        keys_theta_inner = jax.random.split(k_theta_inner, n_outer)
        log_lik_matched = chunked_vmap(
            log_lik_given_theta, y_outer, theta_outer, keys_theta_inner, chunk_size=chunk_size
        )

        # -- log p(y_i): usual marginal, nested MC over eta ~ prior ---------
        def log_marginal(y, k):
            etas_prior = self.prior_eta.sample(k, n_inner_marginal)
            lls = jax.vmap(lambda e: self.likelihood.log_likelihood(y, e, design))(etas_prior)
            return jsp.special.logsumexp(lls) - jnp.log(n_inner_marginal)

        keys_marg_inner = jax.random.split(k_marg_inner, n_outer)
        log_marg = chunked_vmap(log_marginal, y_outer, keys_marg_inner, chunk_size=chunk_size)

        return jnp.mean(log_lik_matched - log_marg)
