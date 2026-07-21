"""Likelihood contract."""

from abc import ABC, abstractmethod
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, Int, PRNGKeyArray

from cboed.core.linear_operator import LinearizedOperator


class Likelihood(ABC):
    r"""``p(y | theta, design)``. Carries the observation operator and the noise.

    The ``design`` enters **here and nowhere else**: it selects what is
    observed, without touching the prior or the forward dynamics.

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
        """``log p(y | theta, design)``."""
        ...

    @abstractmethod
    def jacobian_operator(
        self,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> LinearizedOperator:
        """``d(mean)/dtheta`` at ``(theta, design)``, matrix-free. Independent of y."""
        ...

    @abstractmethod
    def grad_log_likelihood(
        self,
        y: Float[Array, " n_sensors"],
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, " n_param"]:
        """``J^T Sigma_obs^{-1} (y - M(theta))``, in parameter space."""
        ...

    @abstractmethod
    def hessian_operator(
        self,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> LinearizedOperator:
        """Gauss-Newton ``-J^T Sigma_obs^{-1} J``, matrix-free and symmetric."""
        ...

    @abstractmethod
    def sample(
        self,
        key: PRNGKeyArray,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
        n_samples: int = 1,
    ) -> Float[Array, "n_samples n_sensors"]:
        """``y ~ p(. | theta, design)``."""
        ...

    # -- dense oracles: materialized from the operators ---------------------

    @partial(jax.jit, static_argnums=(0,))
    def jacobian(
        self,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, "n_sensors n_param"]:
        """Dense ``J``, ``(m, d)``.

        The ``.T`` matters: ``vmap`` stacks the images as **rows**, but we
        want the columns.
        """
        op = self.jacobian_operator(theta, design)
        return jax.vmap(op.matvec)(jnp.eye(op.shape[1])).T

    @partial(jax.jit, static_argnums=(0,))
    def hessian(
        self,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, "n_param n_param"]:
        r"""Dense Gauss-Newton, ``(d, d)``. Oracle -- forbidden in high dimension.

        Concrete: materializes :meth:`hessian_operator`, a single numerical
        path, no subclass reimplements it in a different way.

        **This is not the true Hessian**:

        .. math::
            \nabla^2 \log p = -J^T \Sigma^{-1} J
            + [\text{term in } \partial^2 u/\partial\theta^2 -- \text{IGNORED}]

        At ``lambda=0`` the omitted term is zero and Gauss-Newton is exact. At
        ``lambda>0`` it differs from autodiff: this is **not a bug**, it is
        the Laplace approximation, and the gap *is* the nonlinearity that the
        bounds quantify. Do not write a test against
        ``jax.hessian(log_likelihood)`` at ``lambda>0`` -- it will fail, rightly so.
        """
        op = self.hessian_operator(theta, design)
        H = jax.vmap(op.matvec)(jnp.eye(op.shape[1])).T
        return 0.5 * (H + H.T)
