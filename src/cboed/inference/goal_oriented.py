"""Inférence goal-oriented : propagation de la postérieure vers une QoI."""

from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from beartype import beartype
from jax import Array
from jaxtyping import Float, Int, jaxtyped

from cboed.inference.base import InferenceModel


class GoalOrientedModel(InferenceModel):
    r"""``theta = h(eta) + xi``, ``Y = u(eta) + eps``.

    Enveloppe une inférence sur la variable latente ``eta`` et propage vers la
    quantité d'intérêt ``theta`` :

    .. math::
        \Sigma_{\theta|Y} = H \Sigma_{\eta|Y} H^T + \Sigma_\theta
        \qquad
        \Sigma_\theta^{prior} = H \Sigma_\eta H^T + \Sigma_\theta

    Implémente le contrat :class:`InferenceModel`, donc le critère EIG existant
    fonctionne sans modification : le goal-oriented change l'inférence, **pas le
    critère**.

    Parameters
    ----------
    inner : InferenceModel
        Inférence sur ``eta``.
    h : Callable
        Extraction ``eta -> theta``. Linéaire dans le cas nominal.
    Sigma_theta : Float[Array, "n_qoi n_qoi"]
        Covariance du bruit ``xi``.
    """

    def __init__(self, **hyperparameters) -> None:
        self._hyperparameters = hyperparameters

    @property
    def inner(self) -> InferenceModel:
        return self._hyperparameters["inner"]

    @property
    def h(self):
        return self._hyperparameters["h"]

    @property
    def Sigma_theta(self) -> Float[Array, "n_qoi n_qoi"]:
        return self._hyperparameters["Sigma_theta"]

    def _h_jacobian(self, eta: Float[Array, " n_param"]) -> Float[Array, "n_qoi n_param"]:
        """``H = dh/deta``. Constante si ``h`` est linéaire."""
        return jax.jacobian(self.h)(eta)

    @staticmethod
    @jax.jit
    def _log_det_precision_from_cov(
        cov: Float[Array, "n_qoi n_qoi"],
    ) -> Float[Array, ""]:
        """``log det Sigma^{-1} = -2 sum log diag L``.

        Cholesky et non ``slogdet`` : ``cov`` est SDP par construction (somme
        d'une forme quadratique PSD et de ``Sigma_theta`` SDP).
        """
        chol = jsp.linalg.cho_factor(cov, lower=True)
        return -2.0 * jnp.sum(jnp.log(jnp.diag(chol[0])))

    # -- covariances QoI --------------------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    @jaxtyped(typechecker=beartype)
    def posterior_covariance_qoi(
        self,
        eta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, "n_qoi n_qoi"]:
        r"""``H Gamma_{eta|Y} H^T + Sigma_theta``.

        ``n_qoi`` solves via l'action, au lieu de matérialiser ``Gamma_{eta|Y}``.
        """
        H = self._h_jacobian(eta)
        return H @ self.inner.posterior_cov_matmul(H.T, eta, design) + self.Sigma_theta

    @partial(jax.jit, static_argnums=(0,))
    @jaxtyped(typechecker=beartype)
    def prior_covariance_qoi(self, eta: Float[Array, " n_param"]) -> Float[Array, "n_qoi n_qoi"]:
        r"""``H Gamma_eta H^T + Sigma_theta``."""
        H = self._h_jacobian(eta)
        return H @ self.inner.prior.prior_cov_matmul(H.T) + self.Sigma_theta

    # -- contrat ----------------------------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    @jaxtyped(typechecker=beartype)
    def log_det_posterior_precision(
        self,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, ""]:
        return self._log_det_precision_from_cov(self.posterior_covariance_qoi(theta, design))

    @partial(jax.jit, static_argnums=(0,))
    @jaxtyped(typechecker=beartype)
    def log_det_prior_precision(self) -> Float[Array, ""]:
        r"""``log det Sigma_theta^{prior,-1}``.

        Évalué au point ``mu_prior``. Valide **uniquement si ``h`` est
        linéaire** : ``H`` est alors constante et le prior QoI ne dépend pas du
        point. Le jour où ``h`` devient non linéaire, ce contrat doit exposer le
        point de linéarisation -- cf.
        ``test_prior_qoi_independent_of_eta_when_linear``.
        """
        return self._log_det_precision_from_cov(self.prior_covariance_qoi(self.inner.prior.mu))

    @partial(jax.jit, static_argnums=(0,))
    @jaxtyped(typechecker=beartype)
    def posterior_cov_matmul(
        self,
        B: Float[Array, "n_qoi k"],
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, "n_qoi k"]:
        """``Sigma_{theta|Y} @ B``. Dense en ``n_qoi`` : la QoI est petite."""
        return self.posterior_covariance_qoi(theta, design) @ B
