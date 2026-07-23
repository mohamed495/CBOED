"""Inference contract: prior + likelihood -> posterior on theta."""

from abc import ABC, abstractmethod

from jax import Array
from jaxtyping import Float, Int


class InferenceModel(ABC):
    r"""Base contract mapping a prior and a likelihood to a posterior on ``theta``.

    Decides **how the posterior is obtained** (closed form, linearization,
    propagation to a QoI...). *How the EIG is estimated* is the concern of
    ``estimators/`` -- not this contract.

    The contract is expressed in **actions**: ``Gamma_post`` (d x d) cannot
    be materialized in high dimension. Everything here is **``y``-independent**
    -- exactly what the criteria consume, and why the EIG can be computed
    before any observation.

    Notes
    -----
    ``theta`` is the **linearization point**. Ignored in the linear-Gaussian
    case (constant Jacobian), it stays in the signature: this is the
    interface that survives into the nonlinear case.
    """

    @abstractmethod
    def log_det_posterior_precision(
        self,
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, ""]:
        """Compute ``log det Gamma_post^{-1}`` at point ``theta``.

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
            Log-determinant of the posterior precision, computed via
            Cholesky (``2 sum log diag L``) -- without inverting, without
            diagonalizing.
        """
        ...

    @abstractmethod
    def log_det_prior_precision(self) -> Float[Array, ""]:
        """Compute ``log det Gamma_prior^{-1}``.

        Returns
        -------
        Float[Array, ""]
            Log-determinant of the prior precision. Takes no argument:
            depends neither on ``theta`` nor on ``design``.
        """
        ...

    @abstractmethod
    def posterior_cov_matmul(
        self,
        B: Float[Array, "n_param k"],
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, "n_param k"]:
        """Compute ``Gamma_post @ B`` without materializing ``Gamma_post``.

        Parameters
        ----------
        B : Float[Array, "n_param k"]
            Matrix of ``k`` directions to propagate through the posterior
            covariance action.
        theta : Float[Array, " n_param"]
            Linearization point.
        design : Int[Array, " n_sensors"] or None, optional
            Indices of the observed sensors. ``None`` means the full field
            is observed.

        Returns
        -------
        Float[Array, "n_param k"]
            ``Gamma_post @ B``.

        Notes
        -----
        A single primitive: the posterior mean (``B = grad``), QoI
        propagation (``B = H^T``), the A-optimal trace, and the dense oracle
        (``B = I``) all derive from it.
        """
        ...
