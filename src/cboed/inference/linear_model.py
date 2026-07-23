"""Linear-Gaussian posterior."""

from functools import partial

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
    r"""Compute the Gaussian posterior via linearization of the forward model.

    .. math::
        \Gamma_{post}^{-1} = \Gamma_{prior}^{-1} + J^T \Sigma_{obs}^{-1} J

    Exact if the forward model is linear; a Laplace approximation otherwise.

    Parameters
    ----------
    prior : Prior
        Prior on ``theta``.
    likelihood : Likelihood
        Likelihood ``p(y | theta, design)``.

    Examples
    --------
    >>> inference = LinearModel(prior=gaussian_prior, likelihood=likelihood)  # doctest: +SKIP
    >>> inference.log_det_posterior_precision(theta, design)  # doctest: +SKIP
    """

    def __init__(self, **hyperparameters) -> None:
        self._hyperparameters = hyperparameters

    @property
    def prior(self) -> Prior:
        """The :class:`~cboed.priors.base.Prior` on ``theta``."""
        return self._hyperparameters["prior"]

    @property
    def likelihood(self) -> Likelihood:
        """The :class:`~cboed.likelihood.base.Likelihood`, ``p(y | theta, design)``."""
        return self._hyperparameters["likelihood"]

    # -- precision and factorization ---------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    @jaxtyped(typechecker=beartype)
    def posterior_precision(
        self,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, "n_param n_param"]:
        r"""Compute the dense posterior precision ``Gamma_post^{-1}``.

        Parameters
        ----------
        theta : Float[Array, " n_param"]
            Linearization point.
        design : Int[Array, " n_sensors"] or None, optional
            Indices of the observed sensors. ``None`` means the full field
            is observed.

        Returns
        -------
        Float[Array, "n_param n_param"]
            ``-(prior.hessian() + likelihood.hessian(theta, design))``.

        Notes
        -----
        The global ``-`` sign turns the sum of two **negative** Hessians into
        a **positive** precision: it is what guarantees that ``cho_factor``
        receives an SPD matrix.

        Outside the contract: dense is assumed here. In high dimension, this
        sum becomes an operator.
        """
        return -(self.prior.hessian() + self.likelihood.hessian(theta=theta, design=design))

    @jaxtyped(typechecker=beartype)
    def _posterior_chol(
        self,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> tuple[Float[Array, "n_param n_param"], bool]:
        """Compute the shared Cholesky factorization of the posterior precision.

        ``y``-independent: hoistable out of loops.

        Notes
        -----
        Warning: no ``jit`` here: the output of ``cho_factor`` contains a
        ``bool`` (``lower``), and jit reboxes scalar outputs as ``jax.Array``
        -- the downstream ``cho_solve`` needs a concrete Python ``bool`` for
        its own ``static_argnames``. Inlines without issue into the jitted
        methods that call it (the same mechanism as ``_obs_chol`` in
        ``GaussianLikelihood``).
        """
        return jsp.linalg.cho_factor(self.posterior_precision(theta, design), lower=True)

    # -- contrat ----------------------------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    @jaxtyped(typechecker=beartype)
    def log_det_posterior_precision(
        self,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, ""]:
        """Compute ``log det Gamma_post^{-1}`` at ``theta``.

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
            ``2 sum log diag L``, from the Cholesky factor ``L`` of
            ``Gamma_post^{-1}``.
        """
        chol = self._posterior_chol(theta, design)
        return 2.0 * jnp.sum(jnp.log(jnp.diag(chol[0])))

    @partial(jax.jit, static_argnums=(0,))
    @jaxtyped(typechecker=beartype)
    def log_det_prior_precision(self) -> Float[Array, ""]:
        """Compute ``log det Gamma_prior^{-1}``.

        Returns
        -------
        Float[Array, ""]
            Delegated to ``self.prior.log_det_precision()``.
        """
        return self.prior.log_det_precision()

    @partial(jax.jit, static_argnums=(0,))
    @jaxtyped(typechecker=beartype)
    def posterior_cov_matmul(
        self,
        B: Float[Array, "n_param k"],
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, "n_param k"]:
        r"""Compute ``Gamma_post @ B`` with a single ``cho_solve`` on the ``k`` columns.

        Parameters
        ----------
        B : Float[Array, "n_param k"]
            Matrix of ``k`` directions to propagate through the posterior
            covariance.
        theta : Float[Array, " n_param"]
            Linearization point.
        design : Int[Array, " n_sensors"] or None, optional
            Indices of the observed sensors. ``None`` means the full field
            is observed.

        Returns
        -------
        Float[Array, "n_param k"]
            ``Gamma_post @ B``, in ``O(d^2 k)`` -- instead of materializing
            ``Gamma_post`` in ``O(d^3)`` only to then project onto ``k``
            directions.
        """
        return jsp.linalg.cho_solve(self._posterior_chol(theta, design), B)

    # -- posterior mean (depends on y) -------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    @jaxtyped(typechecker=beartype)
    def _mu(
        self,
        y,
        theta,
        design=None,
    ):
        """Compute the posterior mean ``mu_post(y)`` by a one-step correction from ``theta``."""
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

    @partial(jax.jit, static_argnums=(0,))
    @jaxtyped(typechecker=beartype)
    def _cov(
        self,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, "n_param n_param"]:
        """Compute the dense posterior covariance ``Gamma_post`` -- test oracle, the ``B = I`` case.

        Forbidden in high dimension. Do not make anything depend on it: go
        through :meth:`posterior_cov_matmul` instead.
        """
        n = theta.shape[0]
        return self.posterior_cov_matmul(jnp.eye(n, dtype=theta.dtype), theta, design)

    @partial(jax.jit, static_argnums=(0, 5))
    @jaxtyped(typechecker=beartype)
    def sample(
        self,
        key: PRNGKeyArray,
        y: Float[Array, " n_sensors"],
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
        n_samples: int = 1,
    ) -> Float[Array, "n_samples n_param"]:
        """Draw samples from the Gaussian posterior.

        Parameters
        ----------
        key : PRNGKeyArray
            Source of randomness.
        y : Float[Array, " n_sensors"]
            Observation.
        theta : Float[Array, " n_param"]
            Linearization point.
        design : Int[Array, " n_sensors"] or None, optional
            Indices of the observed sensors. ``None`` means the full field
            is observed.
        n_samples : int, optional
            Number of samples to draw. Default 1.

        Returns
        -------
        Float[Array, "n_samples n_param"]
            Samples ``mu_post(y) + L z``, ``z ~ N(0, I)``, ``L`` the
            Cholesky factor of the (dense) posterior covariance.
        """

        mean = self._mu(y, theta, design)

        cov = self._cov(theta, design)
        L = jsp.linalg.cholesky(cov, lower=True)

        z = jax.random.normal(
            key,
            (n_samples, mean.shape[0]),
            dtype=mean.dtype,
        )

        return mean + z @ L.T
