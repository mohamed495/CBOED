from cboed.priors.base import KernelBase
import jax.numpy as jnp


class Gaussian(KernelBase):
    """
    RBF/Gaussian kernel.

    k(x, x') = sigma^2 exp(-||x - x'||^2 / (2l^2))

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

    def __init__(self, length_scale: float, sigma: float):
        super().__init__(length_scale=length_scale, sigma=sigma)

    @property
    def length_scale(self):
        return self._hyperparameters["length_scale"]

    @length_scale.setter
    def length_scale(self, length_scale):
        self._hyperparameters["length_scale"] = length_scale

    @property
    def sigma(self):
        return self._hyperparameters["sigma"]

    @sigma.setter
    def sigma(self, sigma):
        self._hyperparameters["sigma"] = sigma

    def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
        x1 = jnp.atleast_1d(x1)
        x2 = jnp.atleast_1d(x2)
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

    def __init__(self, length_scale: float, sigma: float):
        super().__init__(length_scale=length_scale, sigma=sigma)

    def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
        d = jnp.abs(jnp.subtract.outer(x1, x2))
        return (self.sigma**2) * jnp.exp(-d / self.length_scale)

    @property
    def length_scale(self):
        return self._hyperparameters["length_scale"]

    @property
    def sigma(self):
        return self._hyperparameters["sigma"]

    @length_scale.setter
    def length_scale(self, length_scale):
        self.length_scale = length_scale

    @sigma.setter
    def sigma(self, sigma):
        self.sigma = sigma


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

    def __init__(self, length_scale: float, sigma: float):
        super().__init__(length_scale=length_scale, sigma=sigma)

    def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
        d = jnp.abs(jnp.subtract.outer(x1, x2))
        r = jnp.sqrt(3) * d / self.length_scale
        return (self.sigma**2) * (1 + r) * jnp.exp(-r)

    @property
    def length_scale(self):
        return self._hyperparameters["length_scale"]

    @property
    def sigma(self):
        return self._hyperparameters["sigma"]

    @length_scale.setter
    def length_scale(self, length_scale):
        self.length_scale = length_scale

    @sigma.setter
    def sigma(self, sigma):
        self.sigma = sigma


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

    def __init__(self, length_scale: float, sigma: float):
        super().__init__(length_scale=length_scale, sigma=sigma)

    def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
        d = jnp.abs(jnp.subtract.outer(x1, x2))
        r = jnp.sqrt(5) * d / self.length_scale
        return (self.sigma**2) * (1 + r + r**2 / 3) * jnp.exp(-r)

    @property
    def length_scale(self):
        return self._hyperparameters["length_scale"]

    @property
    def sigma(self):
        return self._hyperparameters["sigma"]

    @length_scale.setter
    def length_scale(self, length_scale):
        self.length_scale = length_scale

    @sigma.setter
    def sigma(self, sigma):
        self.sigma = sigma


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

    def __init__(self, length_scale: float, sigma: float, alpha: float):
        super().__init__(length_scale=length_scale, sigma=sigma, alpha=alpha)

    def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
        d = jnp.abs(jnp.subtract.outer(x1, x2))
        return (self.sigma**2) * (
            1 + d**2 / (2 * self.alpha * self.length_scale**2)
        ) ** (-self.alpha)

    @property
    def length_scale(self):
        return self._hyperparameters["length_scale"]

    @property
    def sigma(self):
        return self._hyperparameters["sigma"]

    @property
    def alpha(self):
        return self._hyperparameters["alpha"]

    @length_scale.setter
    def length_scale(self, length_scale):
        self.length_scale = length_scale

    @sigma.setter
    def sigma(self, sigma):
        self.sigma = sigma

    @alpha.setter
    def alpha(self, alpha):
        self.alpha = alpha


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

    def __init__(self, length_scale: float, sigma: float, period: float):
        super().__init__(length_scale=length_scale, sigma=sigma, period=period)

    def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
        d = jnp.abs(jnp.subtract.outer(x1, x2))
        arg = jnp.pi * d / self.period
        return (self.sigma**2) * jnp.exp(
            -2 * jnp.sin(arg) ** 2 / (self.length_scale**2)
        )

    @property
    def length_scale(self):
        return self._hyperparameters["length_scale"]

    @property
    def sigma(self):
        return self._hyperparameters["sigma"]

    @property
    def period(self):
        return self._hyperparameters["period"]

    @length_scale.setter
    def length_scale(self, length_scale):
        self.length_scale = length_scale

    @sigma.setter
    def sigma(self, sigma):
        self.sigma = sigma

    @period.setter
    def alpha(self, period):
        self.period = period
