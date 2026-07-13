"""Gaussian Process priors for Bayesian inverse problems.

Provides Gaussian Process prior distributions with flexible kernel choices
for parameter estimation in PDE-based inverse problems.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from beartype import beartype
from jax import Array
from jaxtyping import Float, PRNGKeyArray, jaxtyped

from cboed.priors.base import KernelBase, Prior


class GaussianProcess:
    """Gaussian Process prior over parameter fields.

    Defines a  Gaussian Process with a specified kernel.
    Computes and stores the prior covariance matrix with numerical
    stabilization (jitter terms) to ensure positive definiteness.

    Parameters
    ----------
    kernel : KernelBase
        Covariance kernel (e.g., Gaussian, Matern32)
    nx : int
        Number of spatial grid points
    domain : tuple, default=(0, 1)
        Domain interval for spatial grid
    jitter : float, default=1e-10
        Small positive value added to diagonal for numerical stability

    Attributes
    ----------
    kernel : KernelBase
        The covariance kernel
    Sigma : np.ndarray
        Prior covariance matrix, shape (nx, nx)
    mu : np.ndarray
        Prior mean (zero), shape (nx,)

    Examples
    --------
    >>> # Using Gaussian (RBF) kernel
    >>> kernel = Gaussian(length_scale=0.2, sigma=1.0)
    >>> prior = GaussianProcessPrior(kernel, mu)
    >>>
    >>> # Using Matérn kernel
    >>> kernel = Matern32(length_scale=0.5, sigma=2.0)
    >>> prior = GaussianProcessPrior(kernel, mu, domain=(0, 2))
    >>>
    >>> # Access components
    >>> print(prior.Sigma.shape)  # (200, 200)
    """

    def __init__(
        self,
        kernel: KernelBase,
        mu: Float[Array, " nx"],
        domain: tuple[float, float] = (0.0, 1.0),
        jitter: float = 1e-10,
    ) -> None:
        self.kernel = kernel
        self.mu = mu
        self.domain = domain
        self.jitter = jitter
        x = jnp.linspace(domain[0], domain[1], len(mu))
        self.Sigma = self._build_covariance(x)

    def _build_covariance(self, x: Float[Array, " nx"]) -> Float[Array, "nx nx"]:
        K = self.kernel(x, x)
        return K + self.jitter * jnp.eye(len(x))


class GaussianPrior(Prior):
    def __init__(self, **hyperparameters):
        super().__init__(**hyperparameters)

        Sigma = self.prior.Sigma

        # Factorization used for linear solves (Sigma^{-1} x)
        self._chol = jsp.linalg.cho_factor(Sigma, lower=True)

        # Cholesky factor used for sampling
        self._L = jnp.linalg.cholesky(Sigma)

        self._H = -jsp.linalg.cho_solve(
            self._chol,
            jnp.eye(self.prior.mu.shape[0], dtype=Sigma.dtype),
        )

    @property
    def prior(self) -> GaussianProcess:
        return self._hyperparameters["prior"]

    # dans GaussianPrior
    @property
    def mu(self) -> Float[Array, " n_param"]:
        return self.prior.mu

    @jaxtyped(typechecker=beartype)
    def log_prior(
        self,
        theta: Float[Array, " n_param"],
    ) -> Float[Array, ""]:
        """Log-density of a multivariate Gaussian prior."""
        n = theta.shape[0]
        r = theta - self.prior.mu

        quad = r @ jsp.linalg.cho_solve(self._chol, r)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(self._chol[0])))

        return -0.5 * (n * jnp.log(2 * jnp.pi) + logdet + quad)

    @jaxtyped(typechecker=beartype)
    def grad_log_prior(
        self,
        theta: Float[Array, " n_param"],
    ) -> Float[Array, " n_param"]:
        """Gradient of the Gaussian log prior."""
        r = theta - self.prior.mu
        return -jsp.linalg.cho_solve(self._chol, r)

    def log_det_precision(self) -> Float:
        return -2.0 * jnp.sum(jnp.log(jnp.diag(self._chol[0])))

    @jaxtyped(typechecker=beartype)
    def hessian(
        self,
    ) -> Float[Array, "n_param n_param"]:
        """Hessian of the Gaussian log prior."""
        return self._H

    def sample(
        self,
        key: PRNGKeyArray,
        n_samples: int = 1,
    ) -> Float[Array, "n_samples n_parameters"]:
        """Draw samples from the Gaussian prior."""
        mean = self.prior.mu
        z = jax.random.normal(key, (n_samples, mean.shape[0]))
        return mean + z @ self._L.T
