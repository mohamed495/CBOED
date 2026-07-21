# src/cboed/priors/kernel.py
"""Stationary covariance kernels for Gaussian priors.

All kernels inherit from :class:`KernelBase`, which carries ``length_scale``,
``sigma``, their validation, and ``_pairwise_distance``. A kernel with no
extra hyperparameter reduces to its ``__call__``.
"""

import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from cboed.priors.base import KernelBase


class Gaussian(KernelBase):
    r"""Gaussian kernel (RBF, squared exponential).

    .. math::
        k(x, x') = \sigma^2 \exp\left(-\frac{|x - x'|^2}{2\ell^2}\right)

    Infinitely differentiable. Super-exponential spectral decay: the Gram
    matrix becomes numerically rank-deficient on a fine grid, which makes it
    a toy-case kernel rather than a realistic prior in high dimension
    (prefer the Matern kernels).

    Parameters
    ----------
    length_scale : float
        Correlation length :math:`\ell`.
    sigma : float
        Signal standard deviation :math:`\sigma`.

    Examples
    --------
    >>> kernel = Gaussian(length_scale=0.1, sigma=1.5)
    >>> K = kernel(x1, x2)
    """

    def __call__(self, x1: Float[Array, " n"], x2: Float[Array, " m"]) -> Float[Array, "n m"]:
        d = self._pairwise_distance(x1, x2)
        return (self.sigma**2) * jnp.exp(-(d**2) / (2 * self.length_scale**2))


class Matern12(KernelBase):
    r"""Matern kernel :math:`\nu = 1/2` (exponential, Ornstein-Uhlenbeck).

    .. math::
        k(x, x') = \sigma^2 \exp\left(-\frac{|x - x'|}{\ell}\right)

    Continuous but nowhere differentiable in mean square: the sample paths
    are the roughest in the family.

    Parameters
    ----------
    length_scale : float
        Correlation length :math:`\ell`.
    sigma : float
        Signal standard deviation :math:`\sigma`.

    Examples
    --------
    >>> kernel = Matern12(length_scale=0.1, sigma=1.5)
    >>> K = kernel(x1, x2)
    """

    def __call__(self, x1: Float[Array, " n"], x2: Float[Array, " m"]) -> Float[Array, "n m"]:
        d = self._pairwise_distance(x1, x2)
        return (self.sigma**2) * jnp.exp(-d / self.length_scale)


class Matern32(KernelBase):
    r"""Matern kernel :math:`\nu = 3/2`.

    .. math::
        k(x, x') = \sigma^2 \left(1 + \frac{\sqrt{3}\,r}{\ell}\right)
                   \exp\left(-\frac{\sqrt{3}\,r}{\ell}\right),
        \qquad r = |x - x'|

    Once differentiable in mean square.

    Parameters
    ----------
    length_scale : float
        Correlation length :math:`\ell`.
    sigma : float
        Signal standard deviation :math:`\sigma`.

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
    r"""Matern kernel :math:`\nu = 5/2`.

    .. math::
        k(x, x') = \sigma^2 \left(1 + \frac{\sqrt{5}\,r}{\ell}
                   + \frac{5r^2}{3\ell^2}\right)
                   \exp\left(-\frac{\sqrt{5}\,r}{\ell}\right),
        \qquad r = |x - x'|

    Twice differentiable in mean square.

    Parameters
    ----------
    length_scale : float
        Correlation length :math:`\ell`.
    sigma : float
        Signal standard deviation :math:`\sigma`.

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
    r"""Rational quadratic kernel -- continuous mixture of RBF kernels.

    .. math::
        k(x, x') = \sigma^2 \left(1 + \frac{|x - x'|^2}
                   {2\alpha\ell^2}\right)^{-\alpha}

    Mixture of correlation scales. Tends to :class:`Gaussian` as
    :math:`\alpha \to \infty`.

    Parameters
    ----------
    length_scale : float
        Correlation length :math:`\ell`.
    sigma : float
        Signal standard deviation :math:`\sigma`.
    alpha : float
        Mixing parameter :math:`\alpha`. Must be strictly positive.

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
    r"""Periodic kernel (exp-sine-squared).

    .. math::
        k(x, x') = \sigma^2 \exp\left(-\frac{2}{\ell^2}
                   \sin^2\left(\frac{\pi |x - x'|}{p}\right)\right)

    Not stationary in the sense of Euclidean distance alone: the correlation
    depends on the distance *modulo* the period :math:`p`.

    Parameters
    ----------
    length_scale : float
        Correlation length :math:`\ell`.
    sigma : float
        Signal standard deviation :math:`\sigma`.
    period : float
        Period :math:`p`. Must be strictly positive.

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
