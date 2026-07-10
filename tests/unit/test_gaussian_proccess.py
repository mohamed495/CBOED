import jax.numpy as jnp

from cboed.priors import kernel
from cboed.priors.gaussian_priors import GaussianProcessPrior


def test_gaussian_process():
    mu = jnp.zeros(10)

    x = jnp.linspace(0.0, 1.0, 10)
    d = jnp.abs(jnp.subtract.outer(x, x))

    # GP with Gaussian Kernel
    assert jnp.allclose(
        GaussianProcessPrior(
            kernel=kernel.Gaussian(length_scale=1.0, sigma=1.0), mu=mu
        ).Sigma,
        jnp.exp(-0.5 * (d) ** 2),
    )

    # GP with matern12 kernel
    assert jnp.allclose(
        GaussianProcessPrior(
            kernel=kernel.Matern12(length_scale=1.0, sigma=1.0), mu=mu
        ).Sigma,
        jnp.exp(-d),
    )
