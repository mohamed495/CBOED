from abc import ABC, abstractmethod

from beartype import beartype
from jax import Array
from jaxtyping import Float, PRNGKeyArray, jaxtyped

from cboed.core.linear_operator import LinearizedOperator


class Likelihood(ABC):
    """p(y | theta, design). Owns the observation operator and the noise model.

    The design design enters here and nowhere else: it selects what is observed,
    leaving the prior and the forward dynamics untouched.
    """

    def __init__(self, **hyperparameters):
        self._hyperparameters = hyperparameters

    @abstractmethod
    def jacobian(
        self,
        theta: Float[Array, " n_param"],
        design: Float[Array, " n_sensors"] | None = None,
    ) -> LinearizedOperator:
        """d(mean)/dtheta at (theta, design), as a matrix-free operator."""
        ...

    @abstractmethod
    @jaxtyped(typechecker=beartype)
    def log_likelihood(
        self,
        y: Float[Array, " n_obs"],
        theta: Float[Array, " n_param"],
        design: Float[Array, " n_sensors"] | None = None,
    ) -> Float[Array, ""]:
        """log p(y | theta, design)."""
        ...

    @abstractmethod
    def sample(
        self,
        key: PRNGKeyArray,
        theta: Float[Array, " n_param"],
        design: Float[Array, " n_sensors"] | None = None,
        n_samples: int = 1,
    ) -> Float[Array, "n_samples n_obs"]:
        """Tire y ~ p(· | theta, design)."""
        ...
