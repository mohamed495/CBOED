# inference/goal_oriented.py
import jax
import jax.numpy as jnp
from beartype import beartype
from jax import Array
from jaxtyping import Float, Int, jaxtyped

from cboed.inference.base import InferenceModel


class GoalOrientedModel:
    r"""Inférence goal-oriented : θ = h(η) + ξ, Y = u(η) + ε.

    Enveloppe un modèle d'inférence sur η et propage vers la QoI θ.
    """

    def __init__(self, **hyperparameters):
        self._hyperparameters = hyperparameters

    @property
    def inner(self) -> InferenceModel:
        """Inférence sur la variable latente η."""
        return self._hyperparameters["inner"]

    @property
    def h(self):
        """Fonction d'extraction de la QoI : η → θ."""
        return self._hyperparameters["h"]

    @property
    def Sigma_theta(self) -> Float[Array, "n_qoi n_qoi"]:
        """Covariance du bruit sur θ = h(η) + ξ."""
        return self._hyperparameters["Sigma_theta"]

    def _h_jacobian(self, eta: Float[Array, " n_eta"]) -> Float[Array, "n_qoi n_eta"]:
        """Jacobienne H = ∂h/∂η au point eta. Constante si h linéaire."""
        return jax.jacobian(self.h)(eta)

    @jaxtyped(typechecker=beartype)
    def posterior_covariance_qoi(
        self,
        eta: Float[Array, " n_eta"],
        design: Int[Array, " n_obs"] | None = None,
    ) -> Float[Array, "n_qoi n_qoi"]:
        r"""Σ_{θ|Y} = H Σ_{η|Y} Hᵀ + Σ_θ."""
        H = self._h_jacobian(eta)
        cov_eta = self.inner._cov(eta, design)  # ton LinearModel
        return H @ cov_eta @ H.T + self.Sigma_theta

    def prior_covariance_qoi(
        self, eta: Float[Array, " n_eta"]
    ) -> Float[Array, "n_qoi n_qoi"]:
        r"""Σ_θ (prior) = H Σ_η Hᵀ + Σ_θ."""
        H = self._h_jacobian(eta)
        Sigma_eta = self.inner.prior.Sigma
        return H @ Sigma_eta @ H.T + self.Sigma_theta

    def log_det_posterior_precision(self, eta, design=None):
        cov = self.posterior_covariance_qoi(eta, design)
        _, ld = jnp.linalg.slogdet(cov)
        return -ld

    def log_det_prior_precision(self, eta=None):
        if eta is None:
            eta = self.inner.prior.mu
        cov = self.prior_covariance_qoi(eta)
        _, ld = jnp.linalg.slogdet(cov)
        return -ld
