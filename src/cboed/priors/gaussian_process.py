"""Gaussian Process priors for Bayesian inverse problems.

Provides Gaussian Process prior distributions with flexible kernel choices
for parameter estimation in PDE-based inverse problems.
"""

import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from cboed.priors.base import KernelBase


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
