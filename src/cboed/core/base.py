r"""Abstract base class for the forward models of :mod:`cboed.core`."""

from abc import ABC, abstractmethod

from jax import Array
from jaxtyping import Float


class ForwardModel(ABC):
    r"""Abstract forward model ``G : theta -> y``.

    Subclasses implement specific PDEs (Burgers, advection-diffusion,
    shallow water...) in 1D, 2D or 3D, exposing a common interface for
    evaluation, differentiation, and hyperparameter storage that the rest of
    the library (bounds, estimators, priors) builds on.

    Parameters
    ----------
    **hyperparameters
        Named model constants (e.g. diffusivity, velocity, domain), stored
        verbatim and exposed by subclasses through read-only properties.

    Attributes
    ----------
    dim : int
        Spatial dimension of the underlying PDE (1, 2, or 3).
    n_parameters : int
        Dimension of the parameter space ``theta``.
    n_obs : int
        Dimension of the full observation space ``y``.
    """

    def __init__(self, **hyperparameters):
        self._hyperparameters = hyperparameters

    @property
    @abstractmethod
    def dim(self) -> int:
        """Spatial dimension -- 1, 2 or 3."""
        ...

    @property
    @abstractmethod
    def n_parameters(self) -> int:
        """Dimension of the parameter space ``theta``."""
        ...

    @property
    @abstractmethod
    def n_obs(self) -> int:
        """Dimension of the full observation space ``y``."""
        ...

    @abstractmethod
    def __call__(
        self,
        theta: Float[Array, " n_parameters"],
        design: Float[Array, " n_sensors"] | None = None,
    ) -> Float[Array, " n_obs"]:
        r"""Evaluate ``G(theta, design)``.

        Parameters
        ----------
        theta : Float[Array, " n_parameters"]
            Model parameter at which to evaluate the forward map.
        design : Float[Array, " n_sensors"] | None
            Sensor positions/indices restricting the output. ``None``
            returns the full observation ``y = G(theta)``.

        Returns
        -------
        Float[Array, " n_obs"]
            Observation ``y`` (full state, or restricted to ``design`` when
            a subclass implements the restriction).
        """
        ...

    @abstractmethod
    def jacobian(self, theta, design):
        r"""Jacobian ``dG/dtheta`` at ``(theta, design)``, materialized.

        Parameters
        ----------
        theta : array
            Model parameter at which to evaluate the Jacobian.
        design : array or None
            Sensor positions/indices restricting the rows of the Jacobian.
            ``None`` returns the full Jacobian.

        Returns
        -------
        J : array, shape (n_obs, n_parameters)
            Dense Jacobian matrix.
        """
        ...

    def matvec(self, v, theta, design):
        """Apply ``J(theta, design) @ v``.

        Default implementation: materializes ``jacobian`` then multiplies.
        Override in a subclass for a matrix-free tangent (e.g. via
        :class:`cboed.core.linear_operator.LinearizedOperator`).

        Returns
        -------
        array, shape (n_obs,)
            The product ``J(theta, design) @ v``.
        """
        return self.jacobian(theta, design) @ v

    def rmatvec(self, v, theta, design):
        """Apply ``J(theta, design).T @ v``.

        Default implementation: materializes ``jacobian`` then multiplies.
        Override in a subclass for a matrix-free adjoint.

        Returns
        -------
        array, shape (n_parameters,)
            The product ``J(theta, design).T @ v``.
        """
        return self.jacobian(theta, design).T @ v
