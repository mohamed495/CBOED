r"""EIG goal-oriented par Monte-Carlo imbriqué -- ``theta = B eta + xi``.

:class:`~cboed.estimators.nmc.NestedMonteCarloEIG` estime ``EIG(eta)`` : son
``log p(y|theta)`` est directement ``likelihood.log_likelihood(y, theta, ...)``
parce que ``theta`` y est **le paramètre du modèle direct lui-même**. Ici
``theta`` est une projection linéaire ``B eta + xi`` de dimension inférieure
(la QoI) : ``p(y|theta)`` n'a plus de forme fermée, c'est une espérance sur
``eta | theta`` -- un second niveau de Monte-Carlo imbriqué.

.. math::
    \mathrm{EIG}(\theta) = E_{\theta, y}\left[
        \log p(y|\theta) - \log p(y)
    \right], \qquad
    p(y|\theta) = E_{\eta|\theta}[p(y|\eta)]

``eta | theta`` est gaussienne en forme fermée (Rem. 3.1, même calcul que
:func:`cboed.bounds.diagnostics.sample_based.sample_Sigma_Y_given_theta`) --
valable seulement pour ``B`` linéaire (constant), et sans lui, aucune forme
fermée pour tirer ``eta | theta`` n'existe.

``log p(y)`` reste le marginal usuel (indépendant de ``theta``) : même
estimateur imbriqué que dans :mod:`cboed.estimators.nmc`.

Coût : ``n_outer * (n_inner_theta + n_inner_marginal)`` évaluations de
``log_likelihood`` (donc du modèle direct) -- un cran plus cher que le NMC
standard, à cause du second niveau d'imbrication.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from cboed.bounds.diagnostics.gradient_based import psd_sqrt
from cboed.estimators.base import EIGEstimator


def _eta_given_theta_params(prior_eta, B: Array, Sigma_xi: Array):
    r"""``eta | theta ~ N(mean_fn(theta), Sigma_pos)`` -- Rem. 3.1, ``B`` constant.

    Covariance indépendante de ``theta`` (factorisée une fois) ; seule la
    moyenne en dépend. Même calcul que dans ``sample_Sigma_Y_given_theta``,
    exposé ici pour être réutilisé sans dupliquer le Kalman.
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
    r"""EIG(theta) par MC imbriqué, ``theta = B eta + xi`` -- Rem. 3.1.

    Parameters
    ----------
    likelihood : Likelihood
        Vraisemblance ``p(y | eta, design)`` sur le champ **complet**.
    prior_eta : Prior
        Prior sur ``eta`` (le champ complet, pas la QoI).
    B : Float[Array, "n_qoi n_eta"]
        Jacobienne de ``h`` (projection QoI), constante.
    Sigma_xi : Float[Array, "n_qoi n_qoi"]
        Covariance du bruit ``xi``. Cf. les modules ``bounds`` : ``Sigma_xi -> 0``
        est une limite singulière, ne pas y aller.

    Notes
    -----
    ⚠️ Biais dans les deux termes imbriqués (``log p(y|theta)`` **et**
    ``log p(y)``), chacun sous-estimé à ``n_inner`` fini (même mécanisme que
    :class:`~cboed.estimators.nmc.NestedMonteCarloEIG` -- Jensen sur le
    ``logsumexp``). Le biais net sur l'EIG n'a pas de signe garanti a priori.
    """

    @property
    def likelihood(self):
        return self._hyperparameters["likelihood"]

    @property
    def prior_eta(self):
        return self._hyperparameters["prior_eta"]

    @property
    def B(self) -> Float[Array, "n_qoi n_eta"]:
        return self._hyperparameters["B"]

    @property
    def Sigma_xi(self) -> Float[Array, "n_qoi n_qoi"]:
        return self._hyperparameters["Sigma_xi"]

    def estimate(
        self,
        key: PRNGKeyArray,
        design: Int[Array, " n_obs"] | None = None,
        n_outer: int = 1000,
        n_inner_theta: int = 500,
        n_inner_marginal: int = 1000,
    ) -> Float[Array, ""]:
        k_eta, k_xi, k_y, k_theta_inner, k_marg_inner = jax.random.split(key, 5)

        mean_fn, L_pos = _eta_given_theta_params(self.prior_eta, self.B, self.Sigma_xi)
        n_eta = self.prior_eta.mu.shape[0]

        # -- boucle externe : eta_i ~ prior, theta_i = B eta_i + xi_i, y_i ~ p(.|eta_i) --
        eta_outer = self.prior_eta.sample(k_eta, n_outer)
        L_xi = psd_sqrt(self.Sigma_xi)
        z_xi = jax.random.normal(k_xi, (n_outer, self.Sigma_xi.shape[0]))
        theta_outer = eta_outer @ self.B.T + z_xi @ L_xi.T
        keys_y = jax.random.split(k_y, n_outer)
        y_outer = jax.vmap(lambda eta, k: self.likelihood.sample(k, eta, design, n_samples=1)[0])(
            eta_outer, keys_y
        )

        # -- log p(y_i | theta_i) : MC imbrique sur eta' | theta_i (Rem. 3.1) --
        def log_lik_given_theta(y, theta, k):
            z = jax.random.normal(k, (n_inner_theta, n_eta))
            etas_cond = mean_fn(theta) + z @ L_pos.T
            lls = jax.vmap(lambda e: self.likelihood.log_likelihood(y, e, design))(etas_cond)
            return jsp.special.logsumexp(lls) - jnp.log(n_inner_theta)

        keys_theta_inner = jax.random.split(k_theta_inner, n_outer)
        log_lik_matched = jax.vmap(log_lik_given_theta)(y_outer, theta_outer, keys_theta_inner)

        # -- log p(y_i) : marginal usuel, MC imbrique sur eta ~ prior --------
        def log_marginal(y, k):
            etas_prior = self.prior_eta.sample(k, n_inner_marginal)
            lls = jax.vmap(lambda e: self.likelihood.log_likelihood(y, e, design))(etas_prior)
            return jsp.special.logsumexp(lls) - jnp.log(n_inner_marginal)

        keys_marg_inner = jax.random.split(k_marg_inner, n_outer)
        log_marg = jax.vmap(log_marginal)(y_outer, keys_marg_inner)

        return jnp.mean(log_lik_matched - log_marg)
