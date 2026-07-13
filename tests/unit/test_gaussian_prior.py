from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import pytest  # type: ignore

import cboed.priors.kernel as kernel
from cboed.core.advection_diffusion import AdvectionDiffusion
from cboed.priors.gaussian_process import GaussianPrior, GaussianProcess


class Setup(NamedTuple):
    gauss_prior: GaussianPrior


@pytest.fixture
def setup() -> Setup:
    model = model = AdvectionDiffusion(
        diffusivity=0.0,
        velocity=2.0,
        T=1,
        domain=[0, 1],
        nt=5,
        n=4,
    )
    prior = GaussianProcess(
        kernel=kernel.Gaussian(length_scale=1.0, sigma=1.0), mu=jnp.zeros(model.n)
    )
    gauss_prior = GaussianPrior(prior=prior)

    return Setup(gauss_prior)


def test_isinstance(setup: Setup) -> None:
    assert isinstance(setup.gauss_prior, GaussianPrior)
    assert isinstance(setup.gauss_prior.prior, GaussianProcess)


def test_log_prior_value(setup: Setup) -> None:

    n = setup.gauss_prior.prior.mu.shape[0]
    theta = jnp.arange(1.0, n + 1)
    mu = setup.gauss_prior.prior.mu
    Sigma = setup.gauss_prior.prior.Sigma

    diff = theta - mu

    _, logdet = jnp.linalg.slogdet(Sigma)

    cf = jsp.linalg.cho_factor(Sigma, lower=True)
    quad = diff @ jsp.linalg.cho_solve(cf, diff)

    expected = -0.5 * (n * jnp.log(2 * jnp.pi) + logdet + quad)

    computed = setup.gauss_prior.log_prior(theta=theta)
    assert jnp.allclose(computed, expected)
    assert isinstance(computed, jax.Array)


def test_grad_log_prior(setup: Setup) -> None:
    theta = jnp.arange(1.0, setup.gauss_prior.prior.mu.shape[0] + 1)

    mu = setup.gauss_prior.prior.mu
    Sigma = setup.gauss_prior.prior.Sigma

    diff = theta - mu

    cf = jsp.linalg.cho_factor(Sigma, lower=True)
    expected = -jsp.linalg.cho_solve(cf, diff)

    computed = setup.gauss_prior.grad_log_prior(theta=theta)

    assert jnp.allclose(expected, computed)
    assert isinstance(computed, jax.Array)


def test_hessian(setup: Setup) -> None:
    computed = setup.gauss_prior.hessian()
    expected = -jnp.linalg.inv(setup.gauss_prior.prior.Sigma)
    assert jnp.allclose(expected, computed)
    assert isinstance(computed, jax.Array)


def test_sample_is_deterministic_given_key(setup):
    key = jax.random.key(0)
    a = setup.gauss_prior.sample(key, n_samples=5)
    b = setup.gauss_prior.sample(key, n_samples=5)
    assert jnp.array_equal(a, b)


def test_different_keys_give_different_samples(setup):
    a = setup.gauss_prior.sample(jax.random.key(0))
    b = setup.gauss_prior.sample(jax.random.key(1))
    assert not jnp.allclose(a, b)
