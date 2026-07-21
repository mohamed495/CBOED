"""Inference contract: prior + likelihood -> posterior on theta."""

from abc import ABC, abstractmethod

from jax import Array
from jaxtyping import Float, Int


class InferenceModel(ABC):
    r"""Prior + likelihood -> posterior on ``theta``.

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
        """``log det Gamma_post^{-1}`` at point ``theta``.

        Via Cholesky (``2 sum log diag L``) -- without inverting, without diagonalizing.
        """
        ...

    @abstractmethod
    def log_det_prior_precision(self) -> Float[Array, ""]:
        """``log det Gamma_prior^{-1}``.

        No argument: depends neither on ``theta`` nor on ``design``.
        """
        ...

    @abstractmethod
    def posterior_cov_matmul(
        self,
        B: Float[Array, "n_param k"],
        theta: Float[Array, " n_param"],
        design: Int[Array, " n_sensors"] | None = None,
    ) -> Float[Array, "n_param k"]:
        """``Gamma_post @ B``, without materializing ``Gamma_post``.

        A single primitive: the posterior mean (``B = grad``), QoI
        propagation (``B = H^T``), the A-optimal trace, and the dense oracle
        (``B = I``) all derive from it.
        """
        ...
