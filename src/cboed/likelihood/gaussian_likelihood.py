import jax
import jax.numpy as jnp
import jax.scipy as jsp
from beartype import beartype
from jax import Array
from jaxtyping import Float, Int, PRNGKeyArray, jaxtyped

from cboed.core.base import ForwardModel
from cboed.core.linear_operator import LinearizedOperator
from cboed.likelihood.base import Likelihood
from cboed.priors.gaussian_process import GaussianProcess


class GaussianLikelihood(Likelihood):
    def __init__(self, **hyperparameters):
        super().__init__(**hyperparameters)
        # inverse of the observation covariance matrix
        self._chol = jsp.linalg.cho_factor(self.Sigma_obs, lower=True)  # once

    @property
    def Sigma_obs(self) -> Float[Array, "n_obs n_obs"]:
        return self._hyperparameters["Sigma_obs"]

    @property
    def model(self) -> ForwardModel:
        return self._hyperparameters["model"]

    @property
    def prior(self) -> GaussianProcess:
        return self._hyperparameters["prior"]

    def _obs_chol(self, design=None):
        """Cholesky de Σ_obs restreint au design (Wₘᵀ Σ_obs Wₘ)."""
        if design is None:
            return self._chol
        Sigma_sub = self.Sigma_obs[jnp.ix_(design, design)]
        return jsp.linalg.cho_factor(Sigma_sub, lower=True)

    @jaxtyped(typechecker=beartype)
    def log_likelihood(
        self,
        y: Float[Array, " n_obs"],
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_obs"] | None = None,
    ) -> Float[Array, ""]:
        """log p(y | theta, design), bruit gaussien additif."""
        chol = self._obs_chol(design)  # ← restreint au design
        r = y - self.model(theta, design)
        n = y.shape[0]
        quad = r @ jsp.linalg.cho_solve(chol, r)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(chol[0])))
        return -0.5 * (n * jnp.log(2 * jnp.pi) + logdet + quad)

    def jacobian(
        self,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> LinearizedOperator:
        """d(mean)/dtheta = A(design), matrix-free operator. Independent of y."""
        return self.model.jacobian_operator(
            theta, design
        )  # composé avec H(design) si besoin

    @jaxtyped(typechecker=beartype)
    def precision_weighted_residual(self, y, theta, design=None):
        """Σ_obs⁻¹ (y - M(θ)), restreint au design."""
        r = y - self.model(theta=theta, design=design)
        return jsp.linalg.cho_solve(self._obs_chol(design), r)

    @jaxtyped(typechecker=beartype)
    def hessian(self, theta, design=None):
        A = self.model.jacobian(theta=theta, design=design)  # (m, n_param)
        chol = self._obs_chol(design)
        H = -A.T @ jsp.linalg.cho_solve(chol, A)
        return 0.5 * (H + H.T)

    @jaxtyped(typechecker=beartype)
    def grad_log_likelihood(
        self,
        y: Float[Array, " n_obs"],
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, " n_param"]:
        """d(log p)/dtheta = J^T Gamma_obs^{-1} (y - M(theta))."""
        op = self.jacobian(theta, design)
        return op.rmatvec(self.precision_weighted_residual(y, theta, design))

    def hessian_operator(
        self,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> LinearizedOperator:
        """Same, matrix-free. Nothing is materialized."""
        A = self.model.jacobian_operator(theta=theta, design=design)
        chol = self._obs_chol(design)

        def matvec(v):
            return -A.rmatvec(jsp.linalg.cho_solve(chol, A.matvec(v)))

        n = A.shape[1]
        return LinearizedOperator(matvec, matvec, (n, n))

    def sample(
        self,
        key: PRNGKeyArray,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
        n_samples: int = 1,
    ) -> Float[Array, "n_samples n_obs"]:
        """Draw y ~ p(· | theta, design)."""

        mean = self.model(theta, design)
        Sigma = (
            self.Sigma_obs
            if design is None
            else self.Sigma_obs[jnp.ix_(design, design)]
        )
        L = jnp.linalg.cholesky(Sigma)
        z = jax.random.normal(key, (n_samples, mean.shape[0]))
        return mean + z @ L.T
