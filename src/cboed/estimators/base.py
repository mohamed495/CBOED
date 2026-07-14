# estimators/base.py
from abc import ABC, abstractmethod

from jaxtyping import Array, Float, Int


class EIGEstimator(ABC):
    """Estimateur d'EIG. Les sous-classes décident *comment* l'approximer."""

    def __init__(self, **hyperparameters):
        self._hyperparameters = hyperparameters

    @property
    def inference(self):
        return self._hyperparameters["inference"]

    @abstractmethod
    def estimate(
        self,
        design: Int[Array, " n_obs"] | None = None,
    ) -> Float[Array, ""]: ...
