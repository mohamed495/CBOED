# estimators/nmc.py
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from cboed.estimators.base import EIGEstimator, chunked_vmap


class NestedMonteCarloEIG(EIGEstimator):
    """EIG via nested Monte Carlo.

    Biased lower bound (bias in 1/M). Reference for the nonlinear case.
    Cost: n_outer x n_inner likelihood evaluations.
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
        chunk_size: int | None = None,
    ) -> Float[Array, ""]:
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
