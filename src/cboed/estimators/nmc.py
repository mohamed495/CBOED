# estimators/nmc.py
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from cboed.estimators.base import EIGEstimator


class NestedMonteCarloEIG(EIGEstimator):
    """EIG par Monte-Carlo imbriqué.

    Borne inférieure biaisée (biais en 1/M). Référence pour le non-linéaire.
    Coût : n_outer x n_inner évaluations de vraisemblance.
    """

    @property
    def likelihood(self):
        return self._hyperparameters["likelihood"]

    @property
    def prior(self):
        return self._hyperparameters["prior"]

    def estimate(
        self,
        key: PRNGKeyArray,
        design: Int[Array, " n_obs"] | None = None,
        n_outer: int = 1000,
        n_inner: int = 1000,
    ) -> Float[Array, ""]:
        k_theta, k_y, k_inner = jax.random.split(key, 3)

        # boucle externe : θᵢ ~ prior,  yᵢ ~ p(·|θᵢ)
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

        log_marg = jax.vmap(log_marginal)(ys)  # (N,)

        return jnp.mean(log_lik_matched - log_marg)
