"""Goal-oriented inference: propagation of the posterior toward a QoI."""

from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from beartype import beartype
from jax import Array
from jaxtyping import Float, Int, jaxtyped

from cboed.inference.base import InferenceModel


class GoalOrientedModel(InferenceModel):
    r"""Propagate an inference on a latent field ``eta`` toward a QoI ``theta = h(eta) + xi``.

    Wraps an inference on the latent variable ``eta`` and propagates it toward
    the quantity of interest ``theta``, with ``Y = u(eta) + eps``:

    .. math::
        \Sigma_{\theta|Y} = H \Sigma_{\eta|Y} H^T + \Sigma_\theta
        \qquad
        \Sigma_\theta^{prior} = H \Sigma_\eta H^T + \Sigma_\theta

    Implements the :class:`~cboed.inference.base.InferenceModel` contract, so
    the existing EIG criterion works unmodified: the goal-oriented layer
    changes the inference, **not the criterion**.

    Parameters
    ----------
    inner : InferenceModel
        Inference on ``eta``.
    h : Callable
        Extraction map ``eta -> theta``. Linear in the nominal case.
    Sigma_theta : Float[Array, "n_qoi n_qoi"]
        Covariance of the noise ``xi``.

    Examples
    --------
    >>> go = GoalOrientedModel(
    ...     inner=inference, h=lambda eta: eta[:n_qoi], Sigma_theta=Sigma_xi
    ... )  # doctest: +SKIP
    >>> go.log_det_posterior_precision(eta0, design)  # doctest: +SKIP
    """

    def __init__(self, **hyperparameters) -> None:
        self._hyperparameters = hyperparameters

    @property
    def inner(self) -> InferenceModel:
        """The wrapped :class:`~cboed.inference.base.InferenceModel` on ``eta``."""
        return self._hyperparameters["inner"]

    @property
    def h(self):
        """The extraction map ``eta -> theta``."""
        return self._hyperparameters["h"]

    @property
    def Sigma_theta(self) -> Float[Array, "n_qoi n_qoi"]:
        """Covariance of the QoI noise ``xi``."""
        return self._hyperparameters["Sigma_theta"]

    def _h_jacobian(self, eta: Float[Array, " n_param"]) -> Float[Array, "n_qoi n_param"]:
        """Compute ``H = dh/deta`` at ``eta``. Constant if ``h`` is linear."""
        return jax.jacobian(self.h)(eta)

    @staticmethod
    @jax.jit
    def _log_det_precision_from_cov(
        cov: Float[Array, "n_qoi n_qoi"],
    ) -> Float[Array, ""]:
        """Compute ``log det Sigma^{-1}`` from a dense SPD covariance, via Cholesky.

        Parameters
        ----------
        cov : Float[Array, "n_qoi n_qoi"]
            Dense SPD covariance matrix.

        Returns
        -------
        Float[Array, ""]
            ``log det Sigma^{-1} = -2 sum log diag L``, computed via Cholesky
            rather than ``slogdet``: ``cov`` is SPD by construction (the sum
            of a PSD quadratic form and the SPD ``Sigma_theta``).
        """
        chol = jsp.linalg.cho_factor(cov, lower=True)
        return -2.0 * jnp.sum(jnp.log(jnp.diag(chol[0])))

    # -- QoI covariances ----------------------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    @jaxtyped(typechecker=beartype)
    def posterior_covariance_qoi(
        self,
        eta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, "n_qoi n_qoi"]:
        r"""Compute the posterior QoI covariance ``H Gamma_{eta|Y} H^T + Sigma_theta``.

        Parameters
        ----------
        eta : Float[Array, " n_param"]
            Linearization point of the inner inference (and of ``h``).
        design : Int[Array, " n_sensors"] or None, optional
            Indices of the observed sensors. ``None`` means the full field
            is observed.

        Returns
        -------
        Float[Array, "n_qoi n_qoi"]
            ``H @ Gamma_{eta|Y} @ H^T + Sigma_theta``, dense in ``n_qoi``.

        Notes
        -----
        ``n_qoi`` solves via the posterior covariance action of ``inner``,
        instead of materializing ``Gamma_{eta|Y}``.
        """
        H = self._h_jacobian(eta)
        return H @ self.inner.posterior_cov_matmul(H.T, eta, design) + self.Sigma_theta

    @partial(jax.jit, static_argnums=(0,))
    @jaxtyped(typechecker=beartype)
    def prior_covariance_qoi(self, eta: Float[Array, " n_param"]) -> Float[Array, "n_qoi n_qoi"]:
        r"""Compute the prior QoI covariance ``H Gamma_eta H^T + Sigma_theta``.

        Parameters
        ----------
        eta : Float[Array, " n_param"]
            Point at which the (constant, if ``h`` linear) Jacobian ``H`` is
            evaluated.

        Returns
        -------
        Float[Array, "n_qoi n_qoi"]
            ``H @ Gamma_eta @ H^T + Sigma_theta``.
        """
        H = self._h_jacobian(eta)
        return H @ self.inner.prior.prior_cov_matmul(H.T) + self.Sigma_theta

    # -- contract -----------------------------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    @jaxtyped(typechecker=beartype)
    def log_det_posterior_precision(
        self,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, ""]:
        """Compute ``log det Sigma_{theta|Y}^{-1}`` at ``theta``.

        Parameters
        ----------
        theta : Float[Array, " n_param"]
            Linearization point (of the inner inference and of ``h``).
        design : Int[Array, " n_sensors"] or None, optional
            Indices of the observed sensors. ``None`` means the full field
            is observed.

        Returns
        -------
        Float[Array, ""]
            Log-determinant of the posterior QoI precision.
        """
        return self._log_det_precision_from_cov(self.posterior_covariance_qoi(theta, design))

    @partial(jax.jit, static_argnums=(0,))
    @jaxtyped(typechecker=beartype)
    def log_det_prior_precision(self) -> Float[Array, ""]:
        """Compute ``log det Sigma_theta^{prior,-1}``.

        Returns
        -------
        Float[Array, ""]
            Log-determinant of the prior QoI precision, evaluated at the
            point ``mu_prior``.

        Notes
        -----
        Valid **only if ``h`` is linear**: ``H`` is then constant and the QoI
        prior does not depend on the point. The day ``h`` becomes nonlinear,
        this contract must expose the linearization point -- cf.
        ``test_prior_qoi_independent_of_eta_when_linear``.
        """
        return self._log_det_precision_from_cov(self.prior_covariance_qoi(self.inner.prior.mu))

    @partial(jax.jit, static_argnums=(0,))
    @jaxtyped(typechecker=beartype)
    def posterior_cov_matmul(
        self,
        B: Float[Array, "n_qoi k"],
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, "n_qoi k"]:
        """Compute ``Sigma_{theta|Y} @ B``.

        Parameters
        ----------
        B : Float[Array, "n_qoi k"]
            Matrix of ``k`` directions in QoI space.
        theta : Float[Array, " n_param"]
            Linearization point.
        design : Int[Array, " n_sensors"] or None, optional
            Indices of the observed sensors. ``None`` means the full field
            is observed.

        Returns
        -------
        Float[Array, "n_qoi k"]
            ``Sigma_{theta|Y} @ B``. Dense in ``n_qoi``: the QoI is assumed
            small enough to materialize its covariance.
        """
        return self.posterior_covariance_qoi(theta, design) @ B
