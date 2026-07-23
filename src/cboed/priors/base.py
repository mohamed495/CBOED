"""Base contracts: covariance kernels and priors."""

from abc import ABC, abstractmethod

import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, PRNGKeyArray


class KernelBase(ABC):
    r"""Define the base class for stationary covariance kernels.

    Carries ``length_scale`` (:math:`\ell`), ``sigma`` (:math:`\sigma`),
    their validation, and the pairwise distance computation. Subclasses only
    implement :meth:`__call__`; those with an extra hyperparameter declare it
    in ``_extra_params``.

    Parameters
    ----------
    length_scale : float
        Correlation length. Strictly positive.
    sigma : float
        Signal standard deviation. Strictly positive.
    **extra : float
        Hyperparameters specific to the subclass. Rejected if absent from
        ``_extra_params``.

    Raises
    ------
    TypeError
        If an unexpected hyperparameter is passed.
    ValueError
        If ``length_scale`` or ``sigma`` is not strictly positive.
    """

    #: Extra hyperparameters accepted by the subclass.
    _extra_params: frozenset[str] = frozenset()

    def __init__(self, length_scale: float, sigma: float, **extra: float) -> None:
        unknown = set(extra) - self._extra_params
        if unknown:
            raise TypeError(
                f"{type(self).__name__} got unexpected hyperparameters: {sorted(unknown)}"
            )
        if length_scale <= 0:
            raise ValueError(f"length_scale must be > 0, got {length_scale}")
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")

        self._hyperparameters: dict[str, float] = {
            "length_scale": length_scale,
            "sigma": sigma,
            **extra,
        }

    @property
    def length_scale(self) -> float:
        return self._hyperparameters["length_scale"]

    @property
    def sigma(self) -> float:
        return self._hyperparameters["sigma"]

    @staticmethod
    def _pairwise_distance(x1: Float[Array, " n"], x2: Float[Array, " m"]) -> Float[Array, "n m"]:
        r"""Compute pairwise distances :math:`|x_1 - x_2|`, shape ``(n, m)``.

        Parameters
        ----------
        x1 : Float[Array, " n"]
            First set of 1D locations.
        x2 : Float[Array, " m"]
            Second set of 1D locations.

        Returns
        -------
        Float[Array, "n m"]
            Pairwise absolute distances.

        Notes
        -----
        Explicit broadcast: ``jnp.subtract.outer`` belongs to the NumPy API
        and is not guaranteed in JAX.
        """
        return jnp.abs(x1[:, None] - x2[None, :])

    @abstractmethod
    def __call__(self, x1: Float[Array, " n"], x2: Float[Array, " m"]) -> Float[Array, "n m"]:
        """Evaluate the Gram matrix ``K(x1, x2)``, shape ``(n, m)``.

        Parameters
        ----------
        x1 : Float[Array, " n"]
            First set of 1D locations.
        x2 : Float[Array, " m"]
            Second set of 1D locations.

        Returns
        -------
        Float[Array, "n m"]
            Gram matrix. Rectangular by construction: no assumption that
            ``x1 is x2``.
        """
        ...


class Prior(ABC):
    r"""Define the abstract contract for the prior ``p(theta)`` on the parameter.

    **Never** takes ``design`` or ``y``: the prior does not depend on the
    observations (cf. the rule "design touches everything that touches the
    data, never what touches only theta").

    The contract is in **actions** (``*_matmul``, ``log_det_*``), not
    matrices: in high dimension ``Gamma_prior`` (d x d) cannot be
    materialized. :meth:`Sigma` and :meth:`hessian` are provided here as
    **dense oracles**, implemented once from the actions -- useful for
    testing and in low dimension, forbidden in high dimension.
    """

    @property
    @abstractmethod
    def mu(self) -> Float[Array, " n_param"]:
        """Return the prior mean, shape ``(n_param,)``.

        Survives low-rank representations: it is a vector, unlike
        ``Gamma_prior``.
        """
        ...

    @abstractmethod
    def log_prior(self, theta: Float[Array, " n_param"]) -> Float[Array, ""]:
        """Evaluate the prior log-density ``log p(theta)``.

        Parameters
        ----------
        theta : Float[Array, " n_param"]
            Parameter at which the prior is evaluated.

        Returns
        -------
        Float[Array, ""]
            ``log p(theta)``, normalization constant included.
        """
        ...

    @abstractmethod
    def grad_log_prior(self, theta: Float[Array, " n_param"]) -> Float[Array, " n_param"]:
        """Compute the gradient of the prior log-density.

        Parameters
        ----------
        theta : Float[Array, " n_param"]
            Parameter at which the gradient is evaluated.

        Returns
        -------
        Float[Array, " n_param"]
            ``-Gamma_prior^{-1} (theta - mu)``.
        """
        ...

    @abstractmethod
    def log_det_precision(self) -> Float[Array, ""]:
        """Compute the log-determinant of the prior precision.

        Returns
        -------
        Float[Array, ""]
            ``log det Gamma_prior^{-1}``, without materializing the inverse.
        """
        ...

    @abstractmethod
    def prior_cov_matmul(self, B: Float[Array, "n_param k"]) -> Float[Array, "n_param k"]:
        """Apply the prior covariance to a batch of vectors.

        Parameters
        ----------
        B : Float[Array, "n_param k"]
            Batch of ``k`` vectors, as columns.

        Returns
        -------
        Float[Array, "n_param k"]
            ``Gamma_prior @ B``, without materializing ``Gamma_prior``.
        """
        ...

    @abstractmethod
    def prior_precision_matmul(self, B: Float[Array, "n_param k"]) -> Float[Array, "n_param k"]:
        """Apply the prior precision to a batch of vectors.

        Parameters
        ----------
        B : Float[Array, "n_param k"]
            Batch of ``k`` vectors, as columns.

        Returns
        -------
        Float[Array, "n_param k"]
            ``Gamma_prior^{-1} @ B``, via solve -- never via inversion.
        """
        ...

    @abstractmethod
    def sample(self, key: PRNGKeyArray, n_samples: int = 1) -> Float[Array, "n_samples n_param"]:
        """Draw samples ``theta ~ p(.)``.

        Parameters
        ----------
        key : PRNGKeyArray
            JAX random key.
        n_samples : int, default=1
            Number of samples to draw.

        Returns
        -------
        Float[Array, "n_samples n_param"]
            Samples, one per row.
        """
        ...

    # -- dense oracles: derived from the actions, never reimplemented ------

    def Sigma(self) -> Float[Array, "n_param n_param"]:
        """Materialize the dense prior covariance ``Gamma_prior``.

        Returns
        -------
        Float[Array, "n_param n_param"]
            Dense ``Gamma_prior``, obtained by applying
            :meth:`prior_cov_matmul` to the identity.

        Notes
        -----
        Oracle -- ``O(d^2)`` memory, forbidden in high dimension.
        """
        n = self.mu.shape[0]
        return self.prior_cov_matmul(jnp.eye(n, dtype=self.mu.dtype))

    def hessian(self) -> Float[Array, "n_param n_param"]:
        """Materialize the dense Hessian of the prior log-density.

        Returns
        -------
        Float[Array, "n_param n_param"]
            ``-Gamma_prior^{-1}``, negative semi-definite.

        Notes
        -----
        Oracle, obtained by applying :meth:`prior_precision_matmul` to the
        identity. Materialized **on demand**, not at construction.
        """
        n = self.mu.shape[0]
        return -self.prior_precision_matmul(jnp.eye(n, dtype=self.mu.dtype))
