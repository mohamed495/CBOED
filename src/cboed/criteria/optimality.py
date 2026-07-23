"""Define design criteria: scalar functionals of the posterior precision to optimize."""

from functools import partial

import jax
import jax.numpy as jnp
from beartype import beartype
from jax import Array
from jaxtyping import Float, Int, jaxtyped

from cboed.criteria.base import Criterion


class EIG(Criterion):
    r"""Compute the Expected Information Gain (EIG).

    .. math::
        \mathrm{EIG} = \tfrac12 \left(
        \log\det \Gamma_{post}^{-1} - \log\det \Gamma_{prior}^{-1}\right)

    Notes
    -----
    Posterior **minus** prior: observing increases information, so
    ``log det Gamma_post^{-1} >= log det Gamma_prior^{-1}``. A negative EIG
    signals the terms have been swapped.

    Examples
    --------
    >>> criterion = EIG(inference=linear_model)  # doctest: +SKIP
    >>> criterion.evaluate(theta, design)  # doctest: +SKIP
    """

    @partial(jax.jit, static_argnums=(0,))
    @jaxtyped(typechecker=beartype)
    def evaluate(
        self,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, ""]:
        """Evaluate the EIG at ``theta``.

        Parameters
        ----------
        theta : Float[Array, " n_param"]
            Linearization point.
        design : Int[Array, " n_sensors"] or None, optional
            Indices of the observed sensors. ``None`` means the full field
            is observed.

        Returns
        -------
        Float[Array, ""]
            ``0.5 * (log det Gamma_post^{-1}(theta, design) - log det Gamma_prior^{-1})``.
        """
        return 0.5 * (
            self.inference.log_det_posterior_precision(theta, design)
            - self.inference.log_det_prior_precision()
        )


class DOptimal(Criterion):
    r"""Compute the D-optimal criterion, ``log det Gamma_post^{-1}``.

    Notes
    -----
    Via the Cholesky log-det rather than ``eigvalsh``: ``log det = sum log lambda``
    mathematically, but Cholesky does not diagonalize -- faster and more
    stable. Reserve ``eigvalsh`` for criteria that touch eigenvalues
    individually (E-optimal).

    Examples
    --------
    >>> criterion = DOptimal(inference=linear_model)  # doctest: +SKIP
    >>> criterion.evaluate(theta, design)  # doctest: +SKIP
    """

    @partial(jax.jit, static_argnums=(0,))
    @jaxtyped(typechecker=beartype)
    def evaluate(
        self,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, ""]:
        """Evaluate ``log det Gamma_post^{-1}`` at ``theta``.

        Parameters
        ----------
        theta : Float[Array, " n_param"]
            Linearization point.
        design : Int[Array, " n_sensors"] or None, optional
            Indices of the observed sensors. ``None`` means the full field
            is observed.

        Returns
        -------
        Float[Array, ""]
            Log-determinant of the posterior precision at ``(theta, design)``.
        """
        return self.inference.log_det_posterior_precision(theta, design)


class AOptimal(Criterion):
    r"""Compute the A-optimal criterion, ``-tr Gamma_post``.

    Notes
    -----
    Goes through ``posterior_cov_matmul(I)``: ``tr Gamma_post = tr(Gamma_post I)``.
    A ``cho_solve``, no eigendecomposition -- more stable and faster than
    ``-sum(1/eigvals)``, and above all the criterion no longer does its own
    linear algebra.

    In high dimension, replace ``I`` with a Rademacher matrix ``Z`` and
    return ``-mean(sum(Z * cov_matmul(Z)))``: the Hutchinson estimator,
    **without changing either the contract or this criterion**.

    Examples
    --------
    >>> criterion = AOptimal(inference=linear_model)  # doctest: +SKIP
    >>> criterion.evaluate(theta, design)  # doctest: +SKIP
    """

    @partial(jax.jit, static_argnums=(0,))
    @jaxtyped(typechecker=beartype)
    def evaluate(
        self,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, ""]:
        """Evaluate ``-tr Gamma_post`` at ``theta``.

        Parameters
        ----------
        theta : Float[Array, " n_param"]
            Linearization point; ``n_param = theta.shape[0]`` also fixes the
            size of the dense identity used to probe the posterior
            covariance action.
        design : Int[Array, " n_sensors"] or None, optional
            Indices of the observed sensors. ``None`` means the full field
            is observed.

        Returns
        -------
        Float[Array, ""]
            Negative trace of the posterior covariance at ``(theta, design)``.
        """
        d = theta.shape[0]
        cov = self.inference.posterior_cov_matmul(jnp.eye(d, dtype=theta.dtype), theta, design)
        return -jnp.trace(cov)
