import jax.numpy as jnp
import jax.scipy as jsp
from beartype import beartype
from jax import Array
from jaxtyping import Float, Int, jaxtyped

from cboed.inference.base import InferenceModel
from cboed.likelihood.base import Likelihood
from cboed.priors.base import Prior


class LinearModel(InferenceModel):
    def __init__(self, **hyperparameters):
        super().__init__(**hyperparameters)

    @property
    def prior(self) -> Prior:
        return self._hyperparameters["prior"]

    @property
    def likelihood(self) -> Likelihood:
        return self._hyperparameters["likelihood"]

    def posterior(
        self,
        y: Float[Array, " n_obs"],
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> tuple[Float[Array, " n_param"], Float[Array, "n_param n_param"]]:
        return (
            self._mu(y=y, theta=theta, design=design),
            self._cov(theta=theta, design=design),
        )

    @jaxtyped(typechecker=beartype)
    def _mu(
        self,
        y: Float[Array, " n_obs"],
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, " n_param"]:
        """posterior mean mu_post = mu_prior + Sigma_post J^T Σ⁻¹ (y - G(μ_prior))."""
        grad = self.likelihood.grad_log_likelihood(y=y, theta=theta, design=design)
        cov = self._cov(theta, design)
        return self.prior.mu + cov @ grad

    @jaxtyped(typechecker=beartype)
    def posterior_precision(
        self,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, "n_param n_param"]:
        r"""Précision postérieure Γ_post⁻¹ = Γ_prior⁻¹ + JᵀΣ⁻¹J."""
        return -(
            self.prior.hessian() + self.likelihood.hessian(theta=theta, design=design)
        )

    def _posterior_chol(self, theta, design=None):
        """Cholesky Factorisation of precision posterior"""
        return jsp.linalg.cho_factor(
            self.posterior_precision(theta, design), lower=True
        )

    def log_det_posterior_precision(self, theta, design=None):
        chol = self._posterior_chol(theta, design)
        return 2.0 * jnp.sum(jnp.log(jnp.diag(chol[0])))

    def log_det_prior_precision(self):
        return self.prior.log_det_precision()

    @jaxtyped(typechecker=beartype)
    def _cov(
        self,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, "n_param n_param"]:
        r"""Covariance postérieure Γ_post = (Γ_prior⁻¹ + JᵀΣ⁻¹J)⁻¹."""
        chol = self._posterior_chol(theta, design)
        n = self.prior.mu.shape[0]
        return jsp.linalg.cho_solve(chol, jnp.eye(n))
