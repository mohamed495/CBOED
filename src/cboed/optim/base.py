from abc import ABC, abstractmethod
from typing import NamedTuple

from jaxtyping import Array, Int


class Result(NamedTuple):
    """Hold the outcome of a design optimization.

    Attributes
    ----------
    design : Int[Array, " n_selected"]
        Selected sensor indices, in the order they were added.
    scores : list of float
        Criterion score after each addition, same length as `design`.
    """

    design: Int[Array, " n_selected"]  # selected indices, in the order added
    scores: list[float]  # criterion score after each addition


class Optimizer(ABC):
    """Abstract base class for greedy design optimization strategies.

    Parameters
    ----------
    criterion
        Criterion object exposing ``evaluate(theta, design)``, scoring a
        candidate design at a given `theta`.
    """

    def __init__(self, criterion):
        self.criterion = criterion

    @abstractmethod
    def run(self, theta, n_sensors, n_candidates) -> Result:
        """Select a design of `n_sensors` candidates out of `n_candidates`.

        Parameters
        ----------
        theta
            Parameter value(s) at which the criterion is evaluated.
        n_sensors : int
            Number of sensors to select (budget).
        n_candidates : int
            Total number of candidate sensor locations to choose from.

        Returns
        -------
        Result
            Selected design and the score trace.
        """
        ...
