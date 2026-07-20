import jax.numpy as jnp
import jax.scipy as jsp
import pytest  # type: ignore

import cboed.priors.kernel as kernel
from cboed.priors.gaussian_process import GaussianProcess


def test_gaussian_process():
    mu = jnp.zeros(10)

    x = jnp.linspace(0.0, 1.0, 10)
    d = jnp.abs(jnp.subtract.outer(x, x))

    # GP with Gaussian Kernel
    assert jnp.allclose(
        GaussianProcess(kernel=kernel.Gaussian(length_scale=1.0, sigma=1.0), mu=mu).Sigma,
        jnp.exp(-0.5 * (d) ** 2),
    )

    # GP with matern12 kernel
    assert jnp.allclose(
        GaussianProcess(kernel=kernel.Matern12(length_scale=1.0, sigma=1.0), mu=mu).Sigma,
        jnp.exp(-d),
    )


@pytest.mark.parametrize("nx", [10, 50, 200])
def test_prior_covariance_is_cholesky_factorizable(nx):
    prior = GaussianProcess(kernel.Gaussian(length_scale=1.0, sigma=1.0), mu=jnp.zeros(nx))
    L, _ = jsp.linalg.cho_factor(prior.Sigma, lower=True)
    assert jnp.all(jnp.isfinite(L))
