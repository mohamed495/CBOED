import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from cboed.priors.base import KernelBase


class Gaussian(KernelBase):
    """
    RBF/Gaussian kernel.

    .. math::
        k(x, x') = \\sigma^2 \\exp\\left(-\\frac{\\|x - x'\\|^2}{2\\ell^2}\\right)

    Infinitely differentiable, very smooth.

    Parameters
    ----------
    length_scale : float
        Correlation length
    sigma : float
        Signal variance

    Examples
    --------
    >>> kernel = Gaussian(length_scale=0.1, sigma=1.5)
    >>> K = kernel(x1, x2)
    """

    def __init__(self, length_scale: float, sigma: float) -> None:
        super().__init__(length_scale=length_scale, sigma=sigma)

    @property
    def length_scale(self) -> float:
        return self._hyperparameters["length_scale"]

    @property
    def sigma(self) -> float:
        return self._hyperparameters["sigma"]

    def __call__(
        self, x1: Float[Array, " n"], x2: Float[Array, " m"]
    ) -> Float[Array, "n m"]:
        d = jnp.abs(jnp.subtract.outer(x1, x2))
        return (self.sigma**2) * jnp.exp(-(d**2) / (2 * self.length_scale**2))


class Matern12(KernelBase):
    """
    Matérn kernel with v=1/2 (Exponential kernel).

    k(x, x') = sigma^2 exp(-|x - x'| / l)

    Once differentiable in mean square.

    Parameters
    ----------
    length_scale : float
        Correlation length l
    sigma : float
        Signal variance

    Examples
    --------
    >>> kernel = Matern12(length_scale=0.1, sigma=1.5)
    >>> K = kernel(x, x')
    """

    def __init__(self, length_scale: float, sigma: float) -> None:
        super().__init__(length_scale=length_scale, sigma=sigma)

    def __call__(
        self, x1: Float[Array, " n"], x2: Float[Array, " m"]
    ) -> Float[Array, "n m"]:
        d = jnp.abs(jnp.subtract.outer(x1, x2))
        return (self.sigma**2) * jnp.exp(-d / self.length_scale)

    @property
    def length_scale(self) -> float:
        return self._hyperparameters["length_scale"]

    @property
    def sigma(self) -> float:
        return self._hyperparameters["sigma"]


class Matern32(KernelBase):
    """
    Matérn kernel with v=3/2.

    k(x, x') = sigma^2 (1 + √3r/l) exp(-√3r/l)

    Once differentiable.

    Parameters
    ----------
    length_scale : float
        Correlation length l
    sigma : float
        Signal variance

    Examples
    --------
    >>> kernel = Matern32(length_scale=0.1, sigma=1.5)
    >>> K = kernel(x, x')
    """

    def __init__(self, length_scale: float, sigma: float) -> None:
        super().__init__(length_scale=length_scale, sigma=sigma)

    def __call__(
        self, x1: Float[Array, " n"], x2: Float[Array, " m"]
    ) -> Float[Array, "n m"]:
        d = jnp.abs(jnp.subtract.outer(x1, x2))
        r = jnp.sqrt(3) * d / self.length_scale
        return (self.sigma**2) * (1 + r) * jnp.exp(-r)

    @property
    def length_scale(self) -> float:
        return self._hyperparameters["length_scale"]

    @property
    def sigma(self) -> float:
        return self._hyperparameters["sigma"]


class Matern52(KernelBase):
    """
    Matérn kernel with v=5/2.

    k(x, x') = sigma^2 (1 + √5r/l + 5r²/(3l^2)) exp(-√5r/l)

    Twice differentiable.

    Parameters
    ----------
    length_scale : float
        Correlation length l
    sigma : float
        Signal variance

    Examples
    --------
    >>> kernel = Matern52(length_scale=0.1, sigma=1.5)
    >>> K = kernel(x, x')
    """

    def __init__(self, length_scale: float, sigma: float) -> None:
        super().__init__(length_scale=length_scale, sigma=sigma)

    def __call__(
        self, x1: Float[Array, " n"], x2: Float[Array, " m"]
    ) -> Float[Array, "n m"]:
        d = jnp.abs(jnp.subtract.outer(x1, x2))
        r = jnp.sqrt(5) * d / self.length_scale
        return (self.sigma**2) * (1 + r + r**2 / 3) * jnp.exp(-r)

    @property
    def length_scale(self) -> float:
        return self._hyperparameters["length_scale"]

    @property
    def sigma(self) -> float:
        return self._hyperparameters["sigma"]


class RationalQuadratic(KernelBase):
    """
    Rational quadratic kernel (mixture of SE kernels).

    k(x, x') = sigma^2 (1 + ||x-x'||²/(2alpha l^2))^(-alpha)

    Parameters
    ----------
    length_scale : float
        Correlation length l
    sigma : float
        Signal variance
    alpha : float

    Examples
    --------
    >>> kernel = RationalQuadratic(length_scale=0.1, sigma=1.5, alpha=1.0)
    >>> K = kernel(x, x')
    """

    def __init__(self, length_scale: float, sigma: float, alpha: float) -> None:
        super().__init__(length_scale=length_scale, sigma=sigma, alpha=alpha)

    def __call__(
        self, x1: Float[Array, " n"], x2: Float[Array, " m"]
    ) -> Float[Array, "n m"]:
        d = jnp.abs(jnp.subtract.outer(x1, x2))
        return (self.sigma**2) * (
            1 + d**2 / (2 * self.alpha * self.length_scale**2)
        ) ** (-self.alpha)

    @property
    def length_scale(self) -> float:
        return self._hyperparameters["length_scale"]

    @property
    def sigma(self) -> float:
        return self._hyperparameters["sigma"]

    @property
    def alpha(self) -> float:
        return self._hyperparameters["alpha"]


class Periodic(KernelBase):
    """
    Periodic kernel for periodic phenomena.

    k(x, x') = sigma^2 exp(-2 sin^2(pi|x-x'|/p) / l^2)

    Parameters
    ----------
    length_scale : float
        Correlation length l
    sigma : float
        Signal variance
    period : float

    Examples
    --------
    >>> kernel = Gaussian(length_scale=0.1, sigma=1.5, period=1.0)
    >>> K = kernel(x, x')
    """

    def __init__(self, length_scale: float, sigma: float, period: float) -> None:
        super().__init__(length_scale=length_scale, sigma=sigma, period=period)

    def __call__(
        self, x1: Float[Array, " n"], x2: Float[Array, " m"]
    ) -> Float[Array, "n m"]:
        d = jnp.abs(jnp.subtract.outer(x1, x2))
        arg = jnp.pi * d / self.period
        return (self.sigma**2) * jnp.exp(
            -2 * jnp.sin(arg) ** 2 / (self.length_scale**2)
        )

    @property
    def length_scale(self) -> float:
        return self._hyperparameters["length_scale"]

    @property
    def sigma(self) -> float:
        return self._hyperparameters["sigma"]

    @property
    def period(self) -> float:
        return self._hyperparameters["period"]
