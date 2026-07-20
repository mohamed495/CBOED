"""Contrats de base : noyaux de covariance et priors."""

from abc import ABC, abstractmethod

import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, PRNGKeyArray


class KernelBase(ABC):
    r"""Base des noyaux de covariance stationnaires.

    Porte ``length_scale`` (:math:`\ell`), ``sigma`` (:math:`\sigma`), leur
    validation et le calcul des distances appariées. Les sous-classes
    n'implémentent que :meth:`__call__` ; celles qui ont un hyperparamètre
    supplémentaire le déclarent dans ``_extra_params``.

    Parameters
    ----------
    length_scale : float
        Longueur de corrélation. Strictement positive.
    sigma : float
        Écart-type du signal. Strictement positif.
    **extra : float
        Hyperparamètres propres à la sous-classe. Refusés si absents de
        ``_extra_params``.

    Raises
    ------
    TypeError
        Si un hyperparamètre inattendu est passé.
    ValueError
        Si ``length_scale`` ou ``sigma`` n'est pas strictement positif.
    """

    #: Hyperparamètres supplémentaires acceptés par la sous-classe.
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
        r"""Distances appariées :math:`|x_1 - x_2|`, shape ``(n, m)``.

        Broadcast explicite : ``jnp.subtract.outer`` relève de l'API NumPy et
        n'est pas garanti en JAX.
        """
        return jnp.abs(x1[:, None] - x2[None, :])

    @abstractmethod
    def __call__(self, x1: Float[Array, " n"], x2: Float[Array, " m"]) -> Float[Array, "n m"]:
        """Matrice de Gram ``K(x1, x2)``, shape ``(n, m)``.

        Rectangulaire par construction : aucune hypothèse ``x1 is x2``.
        """
        ...


class Prior(ABC):
    r"""``p(theta)`` -- le prior sur le paramètre.

    Ne prend **jamais** ``design`` ni ``y`` : le prior ne dépend pas des
    observations (cf. règle « le design touche tout ce qui touche les données,
    jamais ce qui touche seulement theta »).

    Le contrat est en **actions** (``*_matmul``, ``log_det_*``), pas en
    matrices : en haute dimension ``Gamma_prior`` (d x d) n'est pas
    matérialisable. :meth:`Sigma` et :meth:`hessian` sont fournies ici comme
    **oracles denses**, implémentées une fois à partir des actions -- utiles en
    test et en basse dimension, interdites en haute.
    """

    @property
    @abstractmethod
    def mu(self) -> Float[Array, " n_param"]:
        """Moyenne du prior. Survit au low-rank : c'est un vecteur."""
        ...

    @abstractmethod
    def log_prior(self, theta: Float[Array, " n_param"]) -> Float[Array, ""]:
        """``log p(theta)``, constante de normalisation incluse."""
        ...

    @abstractmethod
    def grad_log_prior(self, theta: Float[Array, " n_param"]) -> Float[Array, " n_param"]:
        """``-Gamma_prior^{-1} (theta - mu)``."""
        ...

    @abstractmethod
    def log_det_precision(self) -> Float[Array, ""]:
        """``log det Gamma_prior^{-1}``, sans matérialiser l'inverse."""
        ...

    @abstractmethod
    def prior_cov_matmul(self, B: Float[Array, "n_param k"]) -> Float[Array, "n_param k"]:
        """``Gamma_prior @ B``, sans matérialiser ``Gamma_prior``."""
        ...

    @abstractmethod
    def prior_precision_matmul(self, B: Float[Array, "n_param k"]) -> Float[Array, "n_param k"]:
        """``Gamma_prior^{-1} @ B``, par solve -- jamais par inversion."""
        ...

    @abstractmethod
    def sample(self, key: PRNGKeyArray, n_samples: int = 1) -> Float[Array, "n_samples n_param"]:
        """``theta ~ p(.)``."""
        ...

    # -- oracles denses : dérivés des actions, jamais réimplémentés --------

    def Sigma(self) -> Float[Array, "n_param n_param"]:
        """``Gamma_prior`` dense. Oracle -- O(d^2) mémoire, interdit en haute dim."""
        n = self.mu.shape[0]
        return self.prior_cov_matmul(jnp.eye(n, dtype=self.mu.dtype))

    def hessian(self) -> Float[Array, "n_param n_param"]:
        """``-Gamma_prior^{-1}`` dense -- Hessienne de la log-densité, négative.

        Oracle. Matérialisée **à la demande**, plus à la construction.
        """
        n = self.mu.shape[0]
        return -self.prior_precision_matmul(jnp.eye(n, dtype=self.mu.dtype))
