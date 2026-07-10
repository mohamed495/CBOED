import jax
import jax.numpy as jnp
import jax.scipy as jsp
from beartype import beartype
from jax import Array
from jaxtyping import Float, PRNGKeyArray, jaxtyped

from cboed.core.linear_operator import LinearizedOperator
from cboed.likelihood.base import Likelihood


class gaussianLikelihood(Likelihood):
    def __init__(self, **hyperparameters):
        super().__init__(**hyperparameters)
        # inverse of the observation covariance matrix
        self._chol = jsp.linalg.cho_factor(self.Sigma_obs, lower=True)  # once

    @property
    def Sigma_obs(self):
        return self._hyperparameters["Sigma_obs"]

    @property
    def model(self):
        return self._hyperparameters["model"]

    @property
    def prior(self):
        return self._hyperparameters["prior"]

    @jaxtyped(typechecker=beartype)
    def log_likelihood(
        self,
        y: Float[Array, " n_obs"],
        theta: Float[Array, " n_param"],
        xi: Float[Array, " n_sensors"] | None = None,
    ) -> Float[Array, ""]:
        """log p(y | theta, xi) for additive Gaussian noise."""
        n = y.shape[0]
        A = y - self.model(theta)
        quad = A @ jsp.linalg.cho_solve(self._chol, A)  # A^T Sigma^{-1} A
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(self._chol[0])))  # log det Sigma
        return -0.5 * (n * jnp.log(2 * jnp.pi) + logdet + quad)

    def jacobian(
        self,
        theta: Float[Array, " n_param"],
        xi: Float[Array, " n_sensors"] | None = None,
    ) -> LinearizedOperator:
        """d(mean)/dtheta = A(xi), matrix-free operator. Independent of y."""
        return self.model.jacobian_operator(theta)  # composé avec H(xi) si besoin

    @jaxtyped(typechecker=beartype)
    def whitened_residual(self, y, theta, xi=None):
        """Gamma_obs^{-1} (y - M(theta)). Building block, not a derivative."""
        r = y - self.model(theta)
        return jsp.linalg.cho_solve(self._chol, r)

    @jaxtyped(typechecker=beartype)
    def grad_log_likelihood(
        self,
        y: Float[Array, " n_obs"],
        theta: Float[Array, " n_param"],
        xi: Float[Array, " n_sensors"] | None = None,
    ) -> Float[Array, " n_param"]:
        """d(log p)/dtheta = J^T Gamma_obs^{-1} (y - M(theta))."""
        op = self.jacobian(theta, xi)
        return op.rmatvec(self.whitened_residual(y, theta, xi))

    @jaxtyped(typechecker=beartype)
    def hessian(
        self,
        theta: Float[Array, " n_param"],
        xi: Float[Array, " n_sensors"] | None = None,
    ) -> Float[Array, "n_param n_param"]:
        """d^2(log p)/dtheta^2 = -A^T Gamma_obs^{-1} A, materialized.

        Exact here: the predictive mean is linear in theta, so the
        residual-weighted curvature term vanishes. Independent of y and theta.
        """
        A = self.model.jacobian(theta=theta, xi=xi)  # dense (n_obs, n)
        H = -A.T @ jsp.linalg.cho_solve(self._chol, A)  # never invert Sigma_obs
        return 0.5 * (H + H.T)  # symmetrize (round-off)

    def hessian_operator(
        self,
        theta: Float[Array, " n_param"],
        xi: Float[Array, " n_sensors"] | None = None,
    ) -> LinearizedOperator:
        """Same, matrix-free. Nothing is materialized."""
        A = self.model.jacobian_operator(theta=theta, xi=xi)

        def matvec(v):
            return -A.rmatvec(jsp.linalg.cho_solve(self._chol, A.matvec(v)))

        n = A.shape[1]
        return LinearizedOperator(matvec, matvec, (n, n))

    def sample(
        self,
        key: PRNGKeyArray,
        theta: Float[Array, " n_param"],
        xi: Float[Array, " n_sensors"] | None = None,
        n_samples: int = 1,
    ) -> Float[Array, "n_samples n_obs"]:
        """Draw y ~ p(· | theta, xi)."""

        mean = self.model(theta, xi)
        L = self._chol[0]  # lower=True
        z = jax.random.normal(key, (n_samples, mean.shape[0]))
        return mean + z @ L.T
