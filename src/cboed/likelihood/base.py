"""Abstract contract for the observation likelihood ``p(y | theta, design)``."""

from abc import ABC, abstractmethod
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, Int, PRNGKeyArray

from cboed.core.linear_operator import LinearizedOperator


class Likelihood(ABC):
    r"""Define the abstract contract for ``p(y | theta, design)``.

    Carries the observation operator and the noise model. The ``design``
    enters **here and nowhere else**: it selects what is observed, without
    touching the prior or the forward dynamics.

    Notes
    -----
    **Observation space.** When a ``design`` is given, ``y`` has
    ``n_sensors`` components (``m``); when it is ``None``, everything is
    observed and ``m = p``. In both cases ``y`` lives in ``n_sensors`` --
    no ``y`` in this module has the ``n_obs`` dimension. Only ``Sigma_obs``
    (``p x p``), unrestricted, legitimately carries it.
    """

    @abstractmethod
    def log_likelihood(
        self,
        y: Float[Array, " n_sensors"],
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, ""]:
        """Evaluate the log-likelihood ``log p(y | theta, design)``.

        Parameters
        ----------
        y : Float[Array, " n_sensors"]
            Observed data, restricted to `design` if it is not None.
        theta : Float[Array, " n_param"]
            Parameter at which the likelihood is evaluated.
        design : Int[Array, " n_sensors"] or None, default=None
            Indices of the observed sensors. If None, the full observable
            (``m = p`` components) is used.

        Returns
        -------
        Float[Array, ""]
            Scalar log-density ``log p(y | theta, design)``.
        """
        ...

    @abstractmethod
    def jacobian_operator(
        self,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> LinearizedOperator:
        """Build the matrix-free Jacobian operator ``d(mean)/dtheta`` at ``(theta, design)``.

        Independent of ``y``.

        Parameters
        ----------
        theta : Float[Array, " n_param"]
            Parameter at which the mean map is linearized.
        design : Int[Array, " n_sensors"] or None, default=None
            Indices of the observed sensors; None observes everything.

        Returns
        -------
        LinearizedOperator
            Matrix-free Jacobian, of shape ``(n_sensors, n_param)``.
        """
        ...

    @abstractmethod
    def grad_log_likelihood(
        self,
        y: Float[Array, " n_sensors"],
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, " n_param"]:
        """Compute the gradient of the log-likelihood with respect to ``theta``.

        Parameters
        ----------
        y, theta, design
            As in :meth:`log_likelihood`.

        Returns
        -------
        Float[Array, " n_param"]
            ``J^T Sigma_obs^{-1} (y - M(theta))``, in parameter space.
        """
        ...

    @abstractmethod
    def hessian_operator(
        self,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> LinearizedOperator:
        """Build the matrix-free Gauss-Newton Hessian operator, symmetric.

        Parameters
        ----------
        theta, design
            As in :meth:`jacobian_operator`.

        Returns
        -------
        LinearizedOperator
            Matrix-free ``-J^T Sigma_obs^{-1} J``, of shape
            ``(n_param, n_param)``.
        """
        ...

    @abstractmethod
    def sample(
        self,
        key: PRNGKeyArray,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
        n_samples: int = 1,
    ) -> Float[Array, "n_samples n_sensors"]:
        """Draw samples ``y ~ p(. | theta, design)``.

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
            Samples, one per row.
        """
        ...

    # -- dense oracles: materialized from the operators ---------------------

    @partial(jax.jit, static_argnums=(0,))
    def jacobian(
        self,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, "n_sensors n_param"]:
        """Materialize the dense Jacobian, shape ``(n_sensors, n_param)``.

        Parameters
        ----------
        theta, design
            As in :meth:`jacobian_operator`.

        Returns
        -------
        Float[Array, "n_sensors n_param"]
            Dense Jacobian matrix ``J``, ``(m, d)``.

        Notes
        -----
        Built by applying :meth:`jacobian_operator` to the identity via
        ``vmap`` and transposing. The transpose matters: ``vmap`` stacks the
        images as **rows**, but the Jacobian needs them as columns.
        """
        op = self.jacobian_operator(theta, design)
        return jax.vmap(op.matvec)(jnp.eye(op.shape[1])).T

    @partial(jax.jit, static_argnums=(0,))
    def hessian(
        self,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, "n_param n_param"]:
        r"""Materialize the dense, symmetrized Gauss-Newton Hessian, shape ``(n_param, n_param)``.

        Parameters
        ----------
        theta, design
            As in :meth:`hessian_operator`.

        Returns
        -------
        Float[Array, "n_param n_param"]
            Dense, symmetrized Gauss-Newton approximation ``(d, d)``.
            Oracle -- forbidden in high dimension.

        Notes
        -----
        Concrete: materializes :meth:`hessian_operator` through a single
        numerical path, so no subclass reimplements it differently.

        **This is not the true Hessian**:

        .. math::
            \nabla^2 \log p = -J^T \Sigma^{-1} J
            + [\text{term in } \partial^2 u/\partial\theta^2 -- \text{IGNORED}]

        At ``lambda_=0`` the omitted term is zero and Gauss-Newton is exact. At
        ``lambda_>0`` it differs from autodiff: this is **not a bug**, it is
        the Laplace approximation, and the gap *is* the nonlinearity that the
        bounds quantify. Do not write a test against
        ``jax.hessian(log_likelihood)`` at ``lambda_>0`` -- it will fail, rightly so.
        """
        op = self.hessian_operator(theta, design)
        H = jax.vmap(op.matvec)(jnp.eye(op.shape[1])).T
        return 0.5 * (H + H.T)
