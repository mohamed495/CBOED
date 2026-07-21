from abc import ABC, abstractmethod
from typing import NamedTuple

from jaxtyping import Array, Int


class Result(NamedTuple):
    """Result of a design optimization."""

    design: Int[Array, " n_selected"]  # selected indices, in the order added
    scores: list[float]  # criterion score after each addition


class Optimizer(ABC):
    def __init__(self, criterion):
        self.criterion = criterion

    @abstractmethod
    def run(self, theta, n_sensors, n_candidates) -> Result: ...
