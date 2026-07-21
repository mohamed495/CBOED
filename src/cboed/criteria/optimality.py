"""Design criteria: which quantity is being optimized."""

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

    Posterior **minus** prior: observing increases information, so
    ``log det Gamma_post^{-1} >= log det Gamma_prior^{-1}``. A negative EIG
    signals the terms have been swapped.
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

    Via the Cholesky log-det rather than ``eigvalsh``: ``log det = sum log lambda``
    mathematically, but Cholesky does not diagonalize -- faster and more
    stable. Reserve ``eigvalsh`` for criteria that touch eigenvalues
    individually (E-optimal).
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

    Goes through ``posterior_cov_matmul(I)``: ``tr Gamma_post = tr(Gamma_post I)``.
    A ``cho_solve``, no eigendecomposition -- more stable and faster than
    ``-sum(1/eigvals)``, and above all the criterion no longer does its own
    linear algebra.

    In high dimension, replace ``I`` with a Rademacher matrix ``Z`` and
    return ``-mean(sum(Z * cov_matmul(Z)))``: the Hutchinson estimator,
    **without changing either the contract or this criterion**.
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
