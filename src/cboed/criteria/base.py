# cboed/criteria/base.py
from abc import ABC, abstractmethod

from jaxtyping import Array, Float, Int


class Criterion(ABC):
    """Base contract for optimality criteria: scalar functions of the posterior precision.

    Subclasses fix *which* functional of the posterior precision
    ``Gamma_post^{-1}`` is optimized (EIG, D-optimal log-det, A-optimal
    trace, ...). This base class only stores the hyperparameters and
    exposes the ``inference`` object the criterion is evaluated against.

    Parameters
    ----------
    **hyperparameters : dict
        Keyword hyperparameters stored verbatim. Must include ``inference``,
        an :class:`~cboed.inference.base.InferenceModel` instance exposing
        the posterior/prior precision actions the criterion consumes.
    """

    def __init__(self, **hyperparameters):
        self._hyperparameters = hyperparameters

    @property
    def inference(self):
        """The :class:`~cboed.inference.base.InferenceModel` the criterion evaluates."""
        return self._hyperparameters["inference"]

    @abstractmethod
    def evaluate(
        self,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, ""]:
        """Evaluate the criterion at a linearization point.

        Parameters
        ----------
        theta : Float[Array, " n_param"]
            Point at which the posterior precision is evaluated (the
            linearization point in the nonlinear case; irrelevant, but
            still required by the signature, in the linear-Gaussian case).
        design : Int[Array, " n_sensors"] or None, optional
            Indices of the observed sensors. ``None`` means the full field
            is observed.

        Returns
        -------
        Float[Array, ""]
            Scalar value of the criterion at ``(theta, design)``.
        """
        ...
