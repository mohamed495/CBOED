# estimators/laplace.py
from jaxtyping import Array, Float, Int

from cboed.criteria.optimality import EIG
from cboed.estimators.base import EIGEstimator


class LaplaceEIG(EIGEstimator):
    """EIG via linearization of the model at a reference point.

    Exact in linear-gaussian. A first-order approximation otherwise:
    the model is replaced by its tangent at the point, and the corresponding
    Gaussian EIG is computed. The error grows with the nonlinearity.

    The linearization point is mu_prior by default. The MAP (a data-dependent,
    more accurate point) is an optimization concern and will be supplied
    externally via estimate_at.
    """

    @property
    def eig(self) -> EIG:
        return EIG(inference=self.inference)

    def estimate(
        self,
        design: Int[Array, " n_obs"] | None = None,
    ) -> Float[Array, ""]:
        """EIG linearized at μ_prior."""
        return self.eig.evaluate(self.inference.prior.mu, design)

    def estimate_at(
        self,
        point: Float[Array, " n_param"],
        design: Int[Array, " n_obs"] | None = None,
    ) -> Float[Array, ""]:
        """EIG linearized at an explicit point (e.g. a MAP supplied by an optimizer)."""
        return self.eig.evaluate(point, design)
