"""Critères de design : quelle quantité on optimise."""

from functools import partial

import jax
import jax.numpy as jnp
from beartype import beartype
from jax import Array
from jaxtyping import Float, Int, jaxtyped

from cboed.criteria.base import Criterion


class EIG(Criterion):
    r"""Expected Information Gain.

    .. math::
        \mathrm{EIG} = \tfrac12 \left(
        \log\det \Gamma_{post}^{-1} - \log\det \Gamma_{prior}^{-1}\right)

    Postérieur **moins** prior : observer augmente l'information, donc
    ``log det Gamma_post^{-1} >= log det Gamma_prior^{-1}``. Une EIG négative
    trahit l'inversion.
    """

    @partial(jax.jit, static_argnums=(0,))
    @jaxtyped(typechecker=beartype)
    def evaluate(
        self,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, ""]:
        return 0.5 * (
            self.inference.log_det_posterior_precision(theta, design)
            - self.inference.log_det_prior_precision()
        )


class DOptimal(Criterion):
    r"""``log det Gamma_post^{-1}``.

    Via le log-det Cholesky et non ``eigvalsh`` : ``log det = sum log lambda``
    mathématiquement, mais le Cholesky ne diagonalise pas -- plus rapide, plus
    stable. Réserver ``eigvalsh`` aux critères qui touchent les valeurs propres
    une à une (E-optimal).
    """

    @partial(jax.jit, static_argnums=(0,))
    @jaxtyped(typechecker=beartype)
    def evaluate(
        self,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, ""]:
        return self.inference.log_det_posterior_precision(theta, design)


class AOptimal(Criterion):
    r"""``-tr Gamma_post``.

    Passe par ``posterior_cov_matmul(I)`` : ``tr Gamma_post = tr(Gamma_post I)``.
    Un ``cho_solve``, pas d'eigendécomposition -- plus stable et plus rapide que
    ``-sum(1/eigvals)``, et surtout le critère ne fait plus d'algèbre linéaire
    pour son compte.

    En haute dimension, remplacer ``I`` par une matrice de Rademacher ``Z`` et
    renvoyer ``-mean(sum(Z * cov_matmul(Z)))`` : estimateur de Hutchinson,
    **sans changer ni le contrat ni ce critère**.
    """

    @partial(jax.jit, static_argnums=(0,))
    @jaxtyped(typechecker=beartype)
    def evaluate(
        self,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, ""]:
        d = theta.shape[0]
        cov = self.inference.posterior_cov_matmul(jnp.eye(d, dtype=theta.dtype), theta, design)
        return -jnp.trace(cov)
