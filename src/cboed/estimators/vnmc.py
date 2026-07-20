"""EIG par NMC variationnel avec proposition de Laplace."""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array
from jaxtyping import Float, Int, PRNGKeyArray

from cboed.estimators.base import EIGEstimator


class LaplaceProposal:
    r"""``q(. | y) = N(mu_post(y), Gamma_post)`` -- proposition d'importance.

    Ce dont VNMC a besoin d'une proposition : **tirer** et **évaluer sa
    densité**. Ni l'un ni l'autre ne réclame ``Gamma_post`` : ce sont des
    actions, d'où un objet séparé du contrat :class:`InferenceModel`.

    L'intérêt est de hisser la factorisation. ``Gamma_post^{-1}`` ne dépend ni
    de ``y`` ni de rien qui varie dans la boucle externe (``theta_lin`` et
    ``design`` sont fixes) : elle est factorisée **une fois** à la construction.
    Sinon chaque appel à ``_mu`` refait le même Cholesky -- ``n_outer`` fois la
    même matrice, et ``vmap`` ne factorise pas des calculs identiques à travers
    les instances vmappées.

    Parameters
    ----------
    inference : InferenceModel
        Doit exposer ``prior``, ``likelihood`` et ``_posterior_chol``.
    theta_lin : Float[Array, " n_param"]
        Point de linéarisation. Exact en LG ; à ``lambda > 0`` le MAP serait
        meilleur -- c'est un argument, pas une décision de cet objet.
    design : Int[Array, " n_sensors"] | None
    """

    def __init__(self, inference, theta_lin, design=None) -> None:
        self._inference = inference
        self._theta_lin = theta_lin
        self._design = design
        # UNE factorisation, réutilisée sur toute la boucle externe.
        self._chol = inference._posterior_chol(theta_lin, design)
        cov = jsp.linalg.cho_solve(self._chol, jnp.eye(theta_lin.shape[0], dtype=theta_lin.dtype))
        self._L = jnp.linalg.cholesky(cov)

    def mean(self, y: Float[Array, " n_sensors"]) -> Float[Array, " n_param"]:
        """``mu_post(y)`` -- un seul solve, sur la factorisation stockée."""
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
    r"""EIG par NMC variationnel.

    Borne **supérieure**, comme NMC, mais plus serrée quand ``q`` approxime bien
    la postérieure (le biais vaut ``E[log p(y) - log p_hat]``, contrôlé par la
    variance des poids d'importance).

    En LG, la proposition de Laplace est la postérieure **exacte** -> poids
    constants -> variance nulle -> ``VNMC = EIG``.

    Pour un encadrement, apparier avec PCE (borne **inférieure**) :
    ``PCE <= EIG <= VNMC``. NMC et VNMC sont du **même côté** -- leur différence
    ne mesure pas un encadrement.

    Notes
    -----
    ``VNMC <= NMC`` n'est pas un théorème : c'est vrai quand ``q`` bat le prior
    (cas LG), pas universellement.
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

        # -- boucle externe : (theta_i, y_i) ~ p(theta) p(y | theta) -------
        thetas = self.prior.sample(k_theta, n_outer)
        keys_y = jax.random.split(k_y, n_outer)
        ys = jax.vmap(lambda th, k: self.likelihood.sample(k, th, design, n_samples=1)[0])(
            thetas, keys_y
        )

        log_lik_matched = jax.vmap(lambda y, th: self.likelihood.log_likelihood(y, th, design))(
            ys, thetas
        )

        # -- proposition : construite UNE fois -----------------------------
        proposal = LaplaceProposal(self.inference, self.prior.mu, design)
        keys_prop = jax.random.split(k_prop, n_outer)

        def log_marginal(y, k):
            theta_prop = proposal.sample(k, y, n_inner)
            log_lik = jax.vmap(lambda th: self.likelihood.log_likelihood(y, th, design))(theta_prop)
            log_prior = jax.vmap(self.prior.log_prior)(theta_prop)
            log_q = jax.vmap(lambda th: proposal.log_pdf(th, y))(theta_prop)

            # logsumexp, jamais log(mean(exp)) : overflow garanti sinon.
            log_weights = log_lik + log_prior - log_q
            return jsp.special.logsumexp(log_weights) - jnp.log(n_inner)

        log_marg = jax.vmap(log_marginal)(ys, keys_prop)
        return jnp.mean(log_lik_matched - log_marg)
