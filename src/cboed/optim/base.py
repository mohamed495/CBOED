from abc import ABC, abstractmethod
from typing import NamedTuple

from jaxtyping import Array, Int


class Result(NamedTuple):
    """Résultat d'une optimisation de design."""

    design: Int[Array, " n_selected"]  # indices retenus, dans l'ordre d'ajout
    scores: list[float]  # score du critère après chaque ajout


class Optimizer(ABC):
    def __init__(self, criterion):
        self.criterion = criterion

    @abstractmethod
    def run(self, theta, n_sensors, n_candidates) -> Result: ...
