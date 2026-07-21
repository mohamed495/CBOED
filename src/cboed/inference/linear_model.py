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
    r"""Gaussian posterior via linearization.

    .. math::
        \Gamma_{post}^{-1} = \Gamma_{prior}^{-1} + J^T \Sigma_{obs}^{-1} J

    Exact if the forward model is linear; a Laplace approximation otherwise.
    """

    def __init__(self, **hyperparameters) -> None:
        self._hyperparameters = hyperparameters

    @property
    def prior(self) -> Prior:
        return self._hyperparameters["prior"]

    @property
    def likelihood(self) -> Likelihood:
        return self._hyperparameters["likelihood"]

    # -- precision and factorization ---------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    @jaxtyped(typechecker=beartype)
    def posterior_precision(
        self,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, "n_param n_param"]:
        r"""``Gamma_post^{-1}``, dense.

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
        """Shared factorization. ``y``-independent: hoistable out of loops.

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
        chol = self._posterior_chol(theta, design)
        return 2.0 * jnp.sum(jnp.log(jnp.diag(chol[0])))

    @partial(jax.jit, static_argnums=(0,))
    @jaxtyped(typechecker=beartype)
    def log_det_prior_precision(self) -> Float[Array, ""]:
        return self.prior.log_det_precision()

    @partial(jax.jit, static_argnums=(0,))
    @jaxtyped(typechecker=beartype)
    def posterior_cov_matmul(
        self,
        B: Float[Array, "n_param k"],
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, "n_param k"]:
        r"""``Gamma_post @ B``: a single ``cho_solve`` on the k columns.

        ``O(d^2 k)`` -- instead of materializing ``Gamma_post`` in ``O(d^3)``
        only to then project onto k directions.
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
        """``Gamma_post`` dense -- test oracle, the ``B = I`` case.

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
        """Samples from the Gaussian posterior."""

        mean = self._mu(y, theta, design)

        cov = self._cov(theta, design)
        L = jsp.linalg.cholesky(cov, lower=True)

        z = jax.random.normal(
            key,
            (n_samples, mean.shape[0]),
            dtype=mean.dtype,
        )

        return mean + z @ L.T
