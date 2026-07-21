# cboed/criteria/base.py
from abc import ABC, abstractmethod

from jaxtyping import Array, Float, Int


class Criterion(ABC):
    """Optimality criterion = scalar function of the posterior precision."""

    def __init__(self, **hyperparameters):
        self._hyperparameters = hyperparameters

    @property
    def inference(self):
        return self._hyperparameters["inference"]

    @abstractmethod
    def evaluate(
        self,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, ""]: ...
