"""Gaussian likelihood with additive noise."""

from functools import partial

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
        Forward model.
    Sigma_obs : Float[Array, "n_obs n_obs"]
        Noise covariance on the **full** observable (``p x p``).
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
        r"""Cholesky of ``Sigma_obs`` restricted to the design (``W_m^T Sigma_obs W_m``).

        **The only place that knows how to restrict.** Every method that
        touches the noise goes through here -- including :meth:`sample`. The
        day ``Sigma_obs`` becomes isotropic (``sigma^2 I_m``), only one place
        changes.
        """
        Sigma = self.Sigma_obs if design is None else self.Sigma_obs[jnp.ix_(design, design)]
        return jsp.linalg.cho_factor(Sigma, lower=True)

    @partial(jax.jit, static_argnums=(0,))
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
        """Already composed with ``H(design)`` by the forward model."""
        return self.model.jacobian_operator(theta, design)

    @partial(jax.jit, static_argnums=(0,))
    @jaxtyped(typechecker=beartype)
    def precision_weighted_residual(
        self,
        y: Float[Array, " n_sensors"],
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, " n_sensors"]:
        r"""``Sigma_obs^{-1} (y - M(theta))``, in **observation** space.

        Returns ``Sigma^{-1} r`` and not ``L^{-1} r``: the residual whitened
        in the strict sense is ``L^{-1} r``, but it is ``Sigma^{-1} r`` that
        the gradient needs (``J^T Sigma^{-1} r``).
        """
        r = y - self.model(theta=theta, design=design)
        return jsp.linalg.cho_solve(self._obs_chol(design), r)

    @partial(jax.jit, static_argnums=(0,))
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
        """``-J^T Sigma_obs^{-1} J``, matrix-free. Nothing is materialized."""
        A = self.model.jacobian_operator(theta=theta, design=design)
        chol = self._obs_chol(design)

        def matvec(v: Float[Array, " n_param"]) -> Float[Array, " n_param"]:
            return -A.rmatvec(jsp.linalg.cho_solve(chol, A.matvec(v)))

        n = A.shape[1]
        # matvec passed twice **on purpose**: (A^T S^-1 A)^T = A^T S^-1 A,
        # the operator is symmetric. This is not the historical duplicated-rmatvec
        # bug -- do not "fix" it.
        return LinearizedOperator(matvec, matvec, (n, n))

    @partial(jax.jit, static_argnums=(0, 4))
    def sample(
        self,
        key: PRNGKeyArray,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
        n_samples: int = 1,
    ) -> Float[Array, "n_samples n_sensors"]:
        """``y ~ p(. | theta, design)``, via the shared factorization."""
        mean = self.model(theta, design)
        L = jnp.tril(self._obs_chol(design)[0])
        z = jax.random.normal(key, (n_samples, mean.shape[0]))
        return mean + z @ L.T
