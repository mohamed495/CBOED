# src/cboed/priors/kernel.py
"""Noyaux de covariance stationnaires pour les priors gaussiens.

Tous les noyaux héritent de :class:`KernelBase`, qui porte ``length_scale``,
``sigma``, leur validation et ``_pairwise_distance``. Un noyau sans
hyperparamètre supplémentaire se réduit à son ``__call__``.
"""

import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from cboed.priors.base import KernelBase


class Gaussian(KernelBase):
    r"""Noyau gaussien (RBF, exponentielle quadratique).

    .. math::
        k(x, x') = \sigma^2 \exp\left(-\frac{|x - x'|^2}{2\ell^2}\right)

    Infiniment différentiable. Décroissance spectrale super-exponentielle : la
    Gram devient numériquement de rang déficient sur grille fine, ce qui en
    fait un noyau de cas-jouet plutôt qu'un prior réaliste en haute dimension
    (préférer les Matérn).

    Parameters
    ----------
    length_scale : float
        Longueur de corrélation :math:`\ell`.
    sigma : float
        Écart-type du signal :math:`\sigma`.

    Examples
    --------
    >>> kernel = Gaussian(length_scale=0.1, sigma=1.5)
    >>> K = kernel(x1, x2)
    """

    def __call__(self, x1: Float[Array, " n"], x2: Float[Array, " m"]) -> Float[Array, "n m"]:
        d = self._pairwise_distance(x1, x2)
        return (self.sigma**2) * jnp.exp(-(d**2) / (2 * self.length_scale**2))


class Matern12(KernelBase):
    r"""Noyau de Matérn :math:`\nu = 1/2` (exponentiel, Ornstein-Uhlenbeck).

    .. math::
        k(x, x') = \sigma^2 \exp\left(-\frac{|x - x'|}{\ell}\right)

    Continu mais nulle part différentiable en moyenne quadratique : les
    trajectoires sont les plus rugueuses de la famille.

    Parameters
    ----------
    length_scale : float
        Longueur de corrélation :math:`\ell`.
    sigma : float
        Écart-type du signal :math:`\sigma`.

    Examples
    --------
    >>> kernel = Matern12(length_scale=0.1, sigma=1.5)
    >>> K = kernel(x1, x2)
    """

    def __call__(self, x1: Float[Array, " n"], x2: Float[Array, " m"]) -> Float[Array, "n m"]:
        d = self._pairwise_distance(x1, x2)
        return (self.sigma**2) * jnp.exp(-d / self.length_scale)


class Matern32(KernelBase):
    r"""Noyau de Matérn :math:`\nu = 3/2`.

    .. math::
        k(x, x') = \sigma^2 \left(1 + \frac{\sqrt{3}\,r}{\ell}\right)
                   \exp\left(-\frac{\sqrt{3}\,r}{\ell}\right),
        \qquad r = |x - x'|

    Une fois différentiable en moyenne quadratique.

    Parameters
    ----------
    length_scale : float
        Longueur de corrélation :math:`\ell`.
    sigma : float
        Écart-type du signal :math:`\sigma`.

    Examples
    --------
    >>> kernel = Matern32(length_scale=0.1, sigma=1.5)
    >>> K = kernel(x1, x2)
    """

    def __call__(self, x1: Float[Array, " n"], x2: Float[Array, " m"]) -> Float[Array, "n m"]:
        d = self._pairwise_distance(x1, x2)
        r = jnp.sqrt(3.0) * d / self.length_scale
        return (self.sigma**2) * (1.0 + r) * jnp.exp(-r)


class Matern52(KernelBase):
    r"""Noyau de Matérn :math:`\nu = 5/2`.

    .. math::
        k(x, x') = \sigma^2 \left(1 + \frac{\sqrt{5}\,r}{\ell}
                   + \frac{5r^2}{3\ell^2}\right)
                   \exp\left(-\frac{\sqrt{5}\,r}{\ell}\right),
        \qquad r = |x - x'|

    Deux fois différentiable en moyenne quadratique.

    Parameters
    ----------
    length_scale : float
        Longueur de corrélation :math:`\ell`.
    sigma : float
        Écart-type du signal :math:`\sigma`.

    Examples
    --------
    >>> kernel = Matern52(length_scale=0.1, sigma=1.5)
    >>> K = kernel(x1, x2)
    """

    def __call__(self, x1: Float[Array, " n"], x2: Float[Array, " m"]) -> Float[Array, "n m"]:
        d = self._pairwise_distance(x1, x2)
        r = jnp.sqrt(5.0) * d / self.length_scale
        return (self.sigma**2) * (1.0 + r + r**2 / 3.0) * jnp.exp(-r)


class RationalQuadratic(KernelBase):
    r"""Noyau rationnel quadratique — mélange continu de noyaux RBF.

    .. math::
        k(x, x') = \sigma^2 \left(1 + \frac{|x - x'|^2}
                   {2\alpha\ell^2}\right)^{-\alpha}

    Mélange d'échelles de corrélation. Tend vers :class:`Gaussian` quand
    :math:`\alpha \to \infty`.

    Parameters
    ----------
    length_scale : float
        Longueur de corrélation :math:`\ell`.
    sigma : float
        Écart-type du signal :math:`\sigma`.
    alpha : float
        Paramètre de mélange :math:`\alpha`. Doit être strictement positif.

    Examples
    --------
    >>> kernel = RationalQuadratic(length_scale=0.1, sigma=1.5, alpha=1.0)
    >>> K = kernel(x1, x2)
    """

    _extra_params = frozenset({"alpha"})

    def __init__(self, length_scale: float, sigma: float, alpha: float) -> None:
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        super().__init__(length_scale, sigma, alpha=alpha)

    @property
    def alpha(self) -> float:
        return self._hyperparameters["alpha"]

    def __call__(self, x1: Float[Array, " n"], x2: Float[Array, " m"]) -> Float[Array, "n m"]:
        d = self._pairwise_distance(x1, x2)
        base = 1.0 + d**2 / (2.0 * self.alpha * self.length_scale**2)
        return (self.sigma**2) * base ** (-self.alpha)


class Periodic(KernelBase):
    r"""Noyau périodique (exp-sine-squared).

    .. math::
        k(x, x') = \sigma^2 \exp\left(-\frac{2}{\ell^2}
                   \sin^2\left(\frac{\pi |x - x'|}{p}\right)\right)

    Non stationnaire au sens de la distance euclidienne seule : la corrélation
    dépend de la distance *modulo* la période :math:`p`.

    Parameters
    ----------
    length_scale : float
        Longueur de corrélation :math:`\ell`.
    sigma : float
        Écart-type du signal :math:`\sigma`.
    period : float
        Période :math:`p`. Doit être strictement positive.

    Examples
    --------
    >>> kernel = Periodic(length_scale=0.1, sigma=1.5, period=1.0)
    >>> K = kernel(x1, x2)
    """

    _extra_params = frozenset({"period"})

    def __init__(self, length_scale: float, sigma: float, period: float) -> None:
        if period <= 0:
            raise ValueError(f"period must be > 0, got {period}")
        super().__init__(length_scale, sigma, period=period)

    @property
    def period(self) -> float:
        return self._hyperparameters["period"]

    def __call__(self, x1: Float[Array, " n"], x2: Float[Array, " m"]) -> Float[Array, "n m"]:
        d = self._pairwise_distance(x1, x2)
        arg = jnp.pi * d / self.period
        return (self.sigma**2) * jnp.exp(-2.0 * jnp.sin(arg) ** 2 / (self.length_scale**2))
