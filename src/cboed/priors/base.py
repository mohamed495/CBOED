# src/cboed/priors/base.py
from abc import ABC, abstractmethod

from beartype import beartype
from jax import Array
from jaxtyping import Float, PRNGKeyArray, jaxtyped


class KernelBase(ABC):
    """Abstract base class for covariance kernels.

    Used for defining prior covariance structures in Gaussian processes.

    Parameters
    ----------
    **hyperparams : dict
        Hyperparameters specific to the kernel (e.g., length_scale, sigma)

    Attributes
    ----------
    hyperparams : dict
        Dictionary of hyperparameters

    Examples
    --------
    >>> from boed.priors.kernels import Gaussian, Matern32
    >>> kernel_se = Gaussian(length_scale=0.5, sigma=1.0)
    >>> kernel_m32 = Matern32(length_scale=0.5, sigma=1.0)
    """

    def __init__(self, **hyperparameters):
        self._hyperparameters = hyperparameters

    @abstractmethod
    def __call__(
        self, x1: Float[Array, " n"], x2: Float[Array, " m"]
    ) -> Float[Array, "n m"]:
        """Evaluate the kernel at point pairs.

        Parameters
        ----------
        x1 : np.ndarray
            First set of points, shape (n,)
        x2 : np.ndarray
            Second set of points, shape (m,)

        Returns
        -------
        np.ndarray
            Kernel matrix K(x1, x2), shape (n, m)
        """
        ...


class Prior(ABC):
    """p(theta). Owns the prior model."""

    def __init__(self, **hyperparameters):
        self._hyperparameters = hyperparameters

    @abstractmethod
    @jaxtyped(typechecker=beartype)
    def log_prior(
        self, y: Float[Array, " n_obs"], theta: Float[Array, " n_param"]
    ) -> Float[Array, ""]:
        """log p(y | theta, design)."""
        ...

    @abstractmethod
    def sample(
        self,
        key: PRNGKeyArray,
        theta: Float[Array, " n_param"],
        n_samples: int = 1,
    ) -> Float[Array, "n_samples n_obs"]:
        """Tire y ~ p(· | theta, design)."""
        ...
