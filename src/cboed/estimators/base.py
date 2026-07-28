# estimators/base.py
from abc import ABC, abstractmethod

from jaxtyping import Array, Float, Int

from cboed.bounds.base import chunked_vmap  # re-exported: existing call sites unchanged

__all__ = ["EIGEstimator", "chunked_vmap"]


class EIGEstimator(ABC):
    """Base contract for EIG estimators.

    Subclasses decide *how* the Expected Information Gain is approximated
    (nested Monte Carlo, variational NMC, Laplace linearization, prior
    contrastive estimation, ...).

    Parameters
    ----------
    **hyperparameters : dict
        Keyword hyperparameters stored verbatim (e.g. ``inference``,
        ``prior``, ``likelihood`` depending on the subclass).
    """

    def __init__(self, **hyperparameters):
        self._hyperparameters = hyperparameters

    @property
    def inference(self):
        """The :class:`~cboed.inference.base.InferenceModel`, if used by the subclass."""
        return self._hyperparameters["inference"]

    @abstractmethod
    def estimate(
        self,
        design: Int[Array, " n_obs"] | None = None,
    ) -> Float[Array, ""]:
        """Estimate the EIG for a given design.

        Parameters
        ----------
        design : Int[Array, " n_obs"] or None, optional
            Indices of the observed sensors. ``None`` means the full field
            is observed.

        Returns
        -------
        Float[Array, ""]
            Estimate of the Expected Information Gain for ``design``.
        """
        ...
