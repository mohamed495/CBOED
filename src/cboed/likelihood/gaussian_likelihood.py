"""Implement the Gaussian likelihood with additive observation noise."""

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
    r"""Implement the additive Gaussian noise likelihood ``y = M(theta) + eps``,
      ``eps ~ N(0, Sigma_obs)``.

    Instantiated with keyword hyperparameters,
    ``GaussianLikelihood(model=..., Sigma_obs=...)``.

    Parameters
    ----------
    model : ForwardModel
        Forward model ``M`` mapping ``theta`` (and, through
        :meth:`jacobian_operator`, a design) to the noiseless full
        observable.
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
        r"""Factor ``Sigma_obs`` restricted to the design (``W_m^T Sigma_obs W_m``).

        Parameters
        ----------
        design : Int[Array, " n_sensors"] or None, default=None
            Indices of the observed sensors; None uses the full
            ``Sigma_obs``.

        Returns
        -------
        tuple[Float[Array, "n_sensors n_sensors"], bool]
            Cholesky factor and lower-flag, as returned by
            ``jax.scipy.linalg.cho_factor``.

        Notes
        -----
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
        """Evaluate ``log p(y | theta, design)`` via a Cholesky solve.

        See :meth:`Likelihood.log_likelihood` for the general contract.
        """
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
        """Build the matrix-free Jacobian operator, delegated to the forward model.

        Parameters
        ----------
        theta : Float[Array, " n_param"]
            Parameter at which the mean map is linearized.
        design : Int[Array, " n_sensors"] or None, default=None
            Indices of the observed sensors; None observes everything.

        Returns
        -------
        LinearizedOperator
            Matrix-free Jacobian of ``model``, already composed with the
            restriction operator ``H(design)`` by the forward model.
        """
        return self.model.jacobian_operator(theta, design)

    @partial(jax.jit, static_argnums=(0,))
    @jaxtyped(typechecker=beartype)
    def precision_weighted_residual(
        self,
        y: Float[Array, " n_sensors"],
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, " n_sensors"]:
        r"""Compute ``Sigma_obs^{-1} (y - M(theta))``, in **observation** space.

        Parameters
        ----------
        y : Float[Array, " n_sensors"]
            Observed data, restricted to `design` if it is not None.
        theta : Float[Array, " n_param"]
            Parameter at which the residual is evaluated.
        design : Int[Array, " n_sensors"] or None, default=None
            Indices of the observed sensors; None observes everything.

        Returns
        -------
        Float[Array, " n_sensors"]
            Precision-weighted residual ``Sigma_obs^{-1} (y - M(theta))``.

        Notes
        -----
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
        """Compute the gradient of the Gaussian log-likelihood.

        See :meth:`Likelihood.grad_log_likelihood` for the general contract.
        Computed as ``J^T`` applied to :meth:`precision_weighted_residual`,
        via the matrix-free Jacobian operator.
        """
        op = self.jacobian_operator(theta, design)
        return op.rmatvec(self.precision_weighted_residual(y, theta, design))

    def hessian_operator(
        self,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> LinearizedOperator:
        """Build the matrix-free Gauss-Newton Hessian operator ``-J^T Sigma_obs^{-1} J``.

        Parameters
        ----------
        theta : Float[Array, " n_param"]
            Parameter at which the mean map is linearized.
        design : Int[Array, " n_sensors"] or None, default=None
            Indices of the observed sensors; None observes everything.

        Returns
        -------
        LinearizedOperator
            Matrix-free, symmetric operator of shape ``(n_param, n_param)``.
            Nothing is materialized.

        Notes
        -----
        ``matvec`` is passed twice for both the forward and adjoint action
        **on purpose**: ``(A^T S^-1 A)^T = A^T S^-1 A``, the operator is
        symmetric. This is not the historical duplicated-``rmatvec`` bug --
        do not "fix" it.
        """
        A = self.model.jacobian_operator(theta=theta, design=design)
        chol = self._obs_chol(design)

        def matvec(v: Float[Array, " n_param"]) -> Float[Array, " n_param"]:
            return -A.rmatvec(jsp.linalg.cho_solve(chol, A.matvec(v)))

        n = A.shape[1]
        return LinearizedOperator(matvec, matvec, (n, n))

    @partial(jax.jit, static_argnums=(0, 4))
    def sample(
        self,
        key: PRNGKeyArray,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
        n_samples: int = 1,
    ) -> Float[Array, "n_samples n_sensors"]:
        """Draw samples ``y ~ p(. | theta, design)``, via the shared factorization.

        Parameters
        ----------
        key : PRNGKeyArray
            JAX random key.
        theta : Float[Array, " n_param"]
            Parameter conditioning the distribution.
        design : Int[Array, " n_sensors"] or None, default=None
            Indices of the observed sensors; None observes everything.
        n_samples : int, default=1
            Number of samples to draw.

        Returns
        -------
        Float[Array, "n_samples n_sensors"]
            Samples, one per row: ``mean + z @ L.T`` with ``z`` standard
            normal and ``L`` the Cholesky factor from :meth:`_obs_chol`.
        """
        mean = self.model(theta, design)
        L = jnp.tril(self._obs_chol(design)[0])
        z = jax.random.normal(key, (n_samples, mean.shape[0]))
        return mean + z @ L.T
