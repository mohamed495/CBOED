# estimators/laplace.py
from jaxtyping import Array, Float, Int

from cboed.criteria.optimality import EIG
from cboed.estimators.base import EIGEstimator


class LaplaceEIG(EIGEstimator):
    """EIG par linéarisation du modèle au point de référence.

    Exact in lineare-gaussian. Approximation au premier ordre sinon :
    le modèle est remplacé par sa tangente au point, et l'EIG gaussienne
    correspondante est calculée. L'erreur croît avec la non-linéarité.

    Le point de linéarisation est μ_prior par défaut. Le MAP (point
    data-dépendant, plus précis) relève de l'optimisation et sera fourni
    de l'extérieur via estimate_at.
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
        """EIG linéarisée à un point explicite (ex : MAP fourni par un optimiseur)."""
        return self.eig.evaluate(point, design)
