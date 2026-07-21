# estimators/pce.py
# estimators/vnmc.py
import jax
import jax.numpy as jnp
import jax.scipy as jsp

from cboed.estimators.base import EIGEstimator


class PriorContrastiveEIG(EIGEstimator):
    """EIG via Prior Contrastive Estimation (Foster et al. 2020).

    LOWER bound. Differs from NMC by including θᵢ (the one that generated
    yᵢ) in the inner sum. Together with VNMC (upper bound), it brackets the EIG.
    """

    @property
    def likelihood(self):
        return self._hyperparameters["likelihood"]

    @property
    def prior(self):
        return self._hyperparameters["prior"]

    def estimate(self, key, design=None, n_outer=1000, n_inner=1000):
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
