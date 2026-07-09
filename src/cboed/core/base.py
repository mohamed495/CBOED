# src/cboed/core/base.py
from abc import ABC, abstractmethod


class ForwardModel(ABC):
    """
    Abstract base class for forward models G : θ → y.

    Subclasses implement specific PDEs (Burgers, advection-diffusion,
    shallow water...) in 1D, 2D or 3D.
    """

    def __init__(self, **hyperparameters):
        self._hyperparameters = hyperparameters

    @property
    @abstractmethod
    def dim(self) -> int:
        """Spatial dimension — 1, 2 or 3."""
        ...

    @property
    @abstractmethod
    def n_parameters(self) -> int:
        """Dimension of the parameter space theta"""
        ...

    @property
    @abstractmethod
    def n_obs(self) -> int:
        """Dimension of the observation space y"""
        ...

    @abstractmethod
    def __call__(self, theta, xi):
        """
        Evaluate G(θ, ξ).

        Parameters
        ----------
        theta : array (n_params,)
            Model parameter.
        xi : array (n_sensors, dim)
            Sensor positions.

        Returns
        -------
        y : array (n_obs,)
        """
        ...

    @abstractmethod
    def jacobian(self, theta, xi):
        """
        Jacobian ∂G/∂θ at (θ, ξ).

        Returns
        -------
        J : array (n_obs, n_params)
        """
        ...

    def matvec(self, v, theta, xi):
        """J(θ, ξ) · v — default: calls jacobian(). Override for matrix-free."""
        return self.jacobian(theta, xi) @ v

    def rmatvec(self, v, theta, xi):
        """J(θ, ξ)ᵀ · v — default: calls jacobian(). Override for matrix-free."""
        return self.jacobian(theta, xi).T @ v
