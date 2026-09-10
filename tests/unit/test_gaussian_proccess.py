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


def test_conditioned_prior_uses_schur_complement():
    mu = jnp.zeros(4)
    values = jnp.array([0.0, 0.0])
    gp = GaussianProcess(
        kernel.Matern12(length_scale=0.4, sigma=1.0),
        mu=mu,
        boundary_values=values,
        jitter=0.0,
    )

    x_full = jnp.linspace(0.0, 1.0, 6)
    Sigma_full = gp.kernel(x_full, x_full)
    Sigma_ii = Sigma_full[1:-1, 1:-1]
    Sigma_ie = Sigma_full[1:-1, jnp.array([0, 5])]
    Sigma_ee = Sigma_full[jnp.ix_(jnp.array([0, 5]), jnp.array([0, 5]))]
    expected = Sigma_ii - Sigma_ie @ jnp.linalg.solve(Sigma_ee, Sigma_ie.T)

    assert gp.mu.shape == (4,)
    assert gp.Sigma.shape == (4, 4)
    assert jnp.allclose(gp.Sigma, expected)
    assert jnp.allclose(gp.mu, 0.0)


def test_conditioned_prior_has_inhomogeneous_conditional_mean():
    gp = GaussianProcess(
        kernel.Gaussian(length_scale=0.5, sigma=1.0),
        mu=jnp.zeros(3),
        boundary_values=jnp.array([1.0, -2.0]),
        jitter=0.0,
    )

    x_full = jnp.linspace(0.0, 1.0, 5)
    Sigma_full = gp.kernel(x_full, x_full)
    Sigma_ie = Sigma_full[1:-1, jnp.array([0, 4])]
    Sigma_ee = Sigma_full[jnp.ix_(jnp.array([0, 4]), jnp.array([0, 4]))]
    expected = Sigma_ie @ jnp.linalg.solve(Sigma_ee, jnp.array([1.0, -2.0]))

    assert jnp.allclose(gp.mu, expected)
