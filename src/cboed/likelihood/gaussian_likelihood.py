"""Vraisemblance gaussienne à bruit additif."""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from beartype import beartype
from jax import Array
from jaxtyping import Float, Int, PRNGKeyArray, jaxtyped

from cboed.core.base import ForwardModel
from cboed.core.linear_operator import LinearizedOperator
from cboed.likelihood.base import Likelihood


class GaussianLikelihood(Likelihood):
    r"""``y = M(theta) + eps``, ``eps ~ N(0, Sigma_obs)``.

    Parameters
    ----------
    model : ForwardModel
        Modèle direct.
    Sigma_obs : Float[Array, "n_obs n_obs"]
        Covariance du bruit sur l'observable **complet** (``p x p``).
    """

    def __init__(self, **hyperparameters) -> None:
        self._hyperparameters = hyperparameters

    @property
    def Sigma_obs(self) -> Float[Array, "n_obs n_obs"]:
        return self._hyperparameters["Sigma_obs"]

    @property
    def model(self) -> ForwardModel:
        return self._hyperparameters["model"]

    def _obs_chol(
        self, design: Int[Array, " n_sensors"] | None = None
    ) -> tuple[Float[Array, "n_sensors n_sensors"], bool]:
        r"""Cholesky de ``Sigma_obs`` restreint au design (``W_m^T Sigma_obs W_m``).

        **Le seul endroit qui sait restreindre.** Toute méthode qui touche au
        bruit passe par ici -- y compris :meth:`sample`. Le jour où ``Sigma_obs``
        devient isotrope (``sigma^2 I_m``), un seul point change.
        """
        Sigma = self.Sigma_obs if design is None else self.Sigma_obs[jnp.ix_(design, design)]
        return jsp.linalg.cho_factor(Sigma, lower=True)

    @jaxtyped(typechecker=beartype)
    def log_likelihood(
        self,
        y: Float[Array, " n_sensors"],
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, ""]:
        chol = self._obs_chol(design)
        r = y - self.model(theta, design)
        n = y.shape[0]
        quad = r @ jsp.linalg.cho_solve(chol, r)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(chol[0])))
        return -0.5 * (n * jnp.log(2 * jnp.pi) + logdet + quad)

    def jacobian_operator(
        self,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> LinearizedOperator:
        """Déjà composé avec ``H(design)`` par le modèle direct."""
        return self.model.jacobian_operator(theta, design)

    @jaxtyped(typechecker=beartype)
    def precision_weighted_residual(
        self,
        y: Float[Array, " n_sensors"],
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, " n_sensors"]:
        r"""``Sigma_obs^{-1} (y - M(theta))``, en espace **observation**.

        Rend bien ``Sigma^{-1} r`` et non ``L^{-1} r`` : le résidu blanchi au
        sens strict est ``L^{-1} r``, mais c'est ``Sigma^{-1} r`` dont le
        gradient a besoin (``J^T Sigma^{-1} r``).
        """
        r = y - self.model(theta=theta, design=design)
        return jsp.linalg.cho_solve(self._obs_chol(design), r)

    @jaxtyped(typechecker=beartype)
    def grad_log_likelihood(
        self,
        y: Float[Array, " n_sensors"],
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, " n_param"]:
        op = self.jacobian_operator(theta, design)
        return op.rmatvec(self.precision_weighted_residual(y, theta, design))

    def hessian_operator(
        self,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> LinearizedOperator:
        """``-J^T Sigma_obs^{-1} J``, matrix-free. Rien n'est matérialisé."""
        A = self.model.jacobian_operator(theta=theta, design=design)
        chol = self._obs_chol(design)

        def matvec(v: Float[Array, " n_param"]) -> Float[Array, " n_param"]:
            return -A.rmatvec(jsp.linalg.cho_solve(chol, A.matvec(v)))

        n = A.shape[1]
        # matvec passé deux fois **à dessein** : (A^T S^-1 A)^T = A^T S^-1 A,
        # l'opérateur est symétrique. Ce n'est pas le bug historique du
        # rmatvec dupliqué -- ne pas « corriger ».
        return LinearizedOperator(matvec, matvec, (n, n))

    def sample(
        self,
        key: PRNGKeyArray,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
        n_samples: int = 1,
    ) -> Float[Array, "n_samples n_sensors"]:
        """``y ~ p(. | theta, design)``, via la factorisation partagée."""
        mean = self.model(theta, design)
        L = jnp.tril(self._obs_chol(design)[0])
        z = jax.random.normal(key, (n_samples, mean.shape[0]))
        return mean + z @ L.T
