# estimators/laplace.py
"""EIG via linearization of the model at a reference point."""

from jaxtyping import Array, Float, Int

from cboed.criteria.optimality import EIG
from cboed.estimators.base import EIGEstimator


class LaplaceEIG(EIGEstimator):
    """Estimate EIG by linearizing the forward model at a reference point.

    Exact in linear-Gaussian. A first-order approximation otherwise: the
    model is replaced by its tangent at the point, and the corresponding
    Gaussian EIG is computed. The error grows with the nonlinearity.

    Notes
    -----
    The linearization point is ``mu_prior`` by default (see :meth:`estimate`).
    The MAP (a data-dependent, more accurate point) is an optimization
    concern and is supplied externally via :meth:`estimate_at`.

    Examples
    --------
    >>> laplace = LaplaceEIG(inference=linear_model)  # doctest: +SKIP
    >>> laplace.estimate(design)  # doctest: +SKIP
    """

    @property
    def eig(self) -> EIG:
        """The underlying :class:`~cboed.criteria.optimality.EIG` criterion."""
        return EIG(inference=self.inference)

    def estimate(
        self,
        design: Int[Array, " n_obs"] | None = None,
    ) -> Float[Array, ""]:
        """Estimate the EIG linearized at ``mu_prior``.

        Parameters
        ----------
        design : Int[Array, " n_obs"] or None, optional
            Indices of the observed sensors. ``None`` means the full field
            is observed.

        Returns
        -------
        Float[Array, ""]
            EIG evaluated at ``self.inference.prior.mu``.
        """
        return self.eig.evaluate(self.inference.prior.mu, design)

    def estimate_at(
        self,
        point: Float[Array, " n_param"],
        design: Int[Array, " n_obs"] | None = None,
    ) -> Float[Array, ""]:
        """Estimate the EIG linearized at an explicit point.

        Parameters
        ----------
        point : Float[Array, " n_param"]
            Linearization point, e.g. a MAP estimate supplied by an
            optimizer.
        design : Int[Array, " n_obs"] or None, optional
            Indices of the observed sensors. ``None`` means the full field
            is observed.

        Returns
        -------
        Float[Array, ""]
            EIG evaluated at ``point``.
        """
        return self.eig.evaluate(point, design)
