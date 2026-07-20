"""Postérieure linéaire-gaussienne."""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from beartype import beartype
from jax import Array
from jaxtyping import Float, Int, PRNGKeyArray, jaxtyped

from cboed.inference.base import InferenceModel
from cboed.likelihood.base import Likelihood
from cboed.priors.base import Prior


class LinearModel(InferenceModel):
    r"""Postérieure gaussienne par linéarisation.

    .. math::
        \Gamma_{post}^{-1} = \Gamma_{prior}^{-1} + J^T \Sigma_{obs}^{-1} J

    Exacte si le modèle direct est linéaire ; approximation de Laplace sinon.
    """

    def __init__(self, **hyperparameters) -> None:
        self._hyperparameters = hyperparameters

    @property
    def prior(self) -> Prior:
        return self._hyperparameters["prior"]

    @property
    def likelihood(self) -> Likelihood:
        return self._hyperparameters["likelihood"]

    # -- précision et factorisation ---------------------------------------

    @jaxtyped(typechecker=beartype)
    def posterior_precision(
        self,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, "n_param n_param"]:
        r"""``Gamma_post^{-1}``, dense.

        Le signe ``-`` global transforme la somme de deux Hessiennes
        **négatives** en précision **positive** : c'est lui qui garantit que
        ``cho_factor`` reçoit une SDP.

        Hors contrat : dense assumé. En haute dimension, cette somme devient un
        opérateur.
        """
        return -(self.prior.hessian() + self.likelihood.hessian(theta=theta, design=design))

    @jaxtyped(typechecker=beartype)
    def _posterior_chol(
        self,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> tuple[Float[Array, "n_param n_param"], bool]:
        """Factorisation partagée. `y`-indépendante : hissable hors des boucles."""
        return jsp.linalg.cho_factor(self.posterior_precision(theta, design), lower=True)

    # -- contrat ----------------------------------------------------------

    @jaxtyped(typechecker=beartype)
    def log_det_posterior_precision(
        self,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, ""]:
        chol = self._posterior_chol(theta, design)
        return 2.0 * jnp.sum(jnp.log(jnp.diag(chol[0])))

    @jaxtyped(typechecker=beartype)
    def log_det_prior_precision(self) -> Float[Array, ""]:
        return self.prior.log_det_precision()

    @jaxtyped(typechecker=beartype)
    def posterior_cov_matmul(
        self,
        B: Float[Array, "n_param k"],
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, "n_param k"]:
        r"""``Gamma_post @ B`` : un ``cho_solve`` sur les k colonnes.

        ``O(d^2 k)`` -- au lieu de matérialiser ``Gamma_post`` en ``O(d^3)``
        pour ensuite n'en projeter que k directions.
        """
        return jsp.linalg.cho_solve(self._posterior_chol(theta, design), B)

    # -- moyenne postérieure (dépend de y) --------------------------------

    @jaxtyped(typechecker=beartype)
    def _mu(
        self,
        y,
        theta,
        design=None,
    ):
        grad_like = self.likelihood.grad_log_likelihood(
            y=y,
            theta=theta,
            design=design,
        )

        grad_prior = self.prior.grad_log_prior(theta)

        grad_post = grad_like + grad_prior

        correction = self.posterior_cov_matmul(
            grad_post[:, None],
            theta,
            design,
        )[:, 0]

        return theta + correction

    @jaxtyped(typechecker=beartype)
    def _cov(
        self,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, "n_param n_param"]:
        """``Gamma_post`` dense -- oracle de test, cas ``B = I``.

        Interdite en haute dimension. Ne rien en faire dépendre : passer par
        :meth:`posterior_cov_matmul`.
        """
        n = theta.shape[0]
        return self.posterior_cov_matmul(jnp.eye(n, dtype=theta.dtype), theta, design)

    @jaxtyped(typechecker=beartype)
    def sample(
        self,
        key: PRNGKeyArray,
        y: Float[Array, " n_sensors"],
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
        n_samples: int = 1,
    ) -> Float[Array, "n_samples n_param"]:
        """Échantillons de la postérieure gaussienne."""

        mean = self._mu(y, theta, design)

        cov = self._cov(theta, design)
        L = jsp.linalg.cholesky(cov, lower=True)

        z = jax.random.normal(
            key,
            (n_samples, mean.shape[0]),
            dtype=mean.dtype,
        )

        return mean + z @ L.T
