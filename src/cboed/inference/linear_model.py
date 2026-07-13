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
        xi: Int[Array, " n_sensors"] | None = None,
    ) -> tuple[Float[Array, " n_param"], Float[Array, "n_param n_param"]]:
        return (self._mu(y=y, theta=theta, xi=xi), self._cov(theta=theta, xi=xi))

    @jaxtyped(typechecker=beartype)
    def _mu(
        self,
        y: Float[Array, " n_obs"],
        theta: Float[Array, " n_param"],
        xi: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, " n_param"]:
        """posterior mean mu_post = mu_prior + Sigma_post J^T Σ⁻¹ (y - G(μ_prior))."""
        grad = self.likelihood.grad_log_likelihood(y=y, theta=theta, xi=xi)
        cov = self._cov(theta, xi)
        return self.prior.mu + cov @ grad

    @jaxtyped(typechecker=beartype)
    def _cov(
        self,
        theta: Float[Array, " n_param"],
        xi: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, "n_param n_param"]:
        r"""Covariance postérieure.

        .. math::
            \Sigma_{post} = (\Sigma_{prior}^{-1} + J^\top \Sigma_{obs}^{-1} J)^{-1}
        """
        """Covariance postérieure Γ_post = (Γ_prior⁻¹ + JᵀΣ⁻¹J)⁻¹."""
        precision = -(
            self.prior.hessian() + self.likelihood.hessian(theta=theta, xi=xi)
        )
        chol = jsp.linalg.cho_factor(precision, lower=True)
        return jsp.linalg.cho_solve(chol, jnp.eye(precision.shape[0]))

    def expected_information_gain(self, xi=None):
        """EIG of design xi."""
        ...
