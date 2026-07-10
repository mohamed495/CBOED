from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import pytest

import cboed.priors.kernel as kernel
from cboed.core.advection_diffusion import AdvectionDiffusion
from cboed.core.linear_operator import LinearizedOperator
from cboed.likelihood.gaussian_likelihood import gaussianLikelihood
from cboed.priors.gaussian_priors import GaussianProcessPrior


class Setup(NamedTuple):
    model: AdvectionDiffusion
    prior: GaussianProcessPrior
    likelihood: gaussianLikelihood


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
    prior = GaussianProcessPrior(
        kernel=kernel.Gaussian(length_scale=1.0, sigma=1.0), mu=jnp.ones(model.n)
    )
    likelihood = gaussianLikelihood(
        model=model, prior=prior, Sigma_obs=jnp.eye(model.n)
    )
    return Setup(model, prior, likelihood)


def test_isinstance(setup: Setup) -> None:
    assert isinstance(setup.likelihood.prior, GaussianProcessPrior)
    assert isinstance(setup.likelihood.prior.kernel, kernel.Gaussian)


def test_properties(setup: Setup) -> None:
    assert jnp.allclose(setup.likelihood.Sigma_obs, jnp.eye(setup.model.n))
    assert jnp.allclose(setup.likelihood.prior.Sigma, setup.prior.Sigma)
    # model checking
    assert jnp.allclose(setup.model.diffusivity, setup.likelihood.model.diffusivity)


def test_log_likelihood_value(setup: Setup) -> None:
    theta = jnp.arange(1.0, setup.model.n + 1)
    y = setup.model(theta=theta)

    computed = setup.likelihood.log_likelihood(y=y, theta=theta)
    # r = 0, Sigma_obs = I  =>  log p = -n/2 * log(2 pi)
    expected = -0.5 * setup.model.n * jnp.log(2 * jnp.pi)
    assert jnp.allclose(computed, expected)
    assert isinstance(computed, jax.Array)


def test_log_likelihood_quadratic(setup: Setup) -> None:
    theta = jnp.arange(1.0, setup.model.n + 1)
    key = jax.random.key(42)
    r = jax.random.multivariate_normal(
        key=key, mean=jnp.zeros(setup.model.n), cov=setup.likelihood.Sigma_obs
    )  # residu impose

    y = setup.model(theta=theta) + r

    computed = setup.likelihood.log_likelihood(y=y, theta=theta)
    expected = -0.5 * (setup.model.n * jnp.log(2 * jnp.pi) + r @ r)
    assert jnp.allclose(computed, expected)


def test_jacobian(setup: Setup) -> None:
    theta = jnp.arange(1.0, setup.model.n + 1)
    jacobian = setup.likelihood.jacobian(theta=theta)
    assert isinstance(jacobian, LinearizedOperator)
    assert jnp.allclose(jacobian.matvec(theta), setup.model(theta))


def test_whitened_residual(setup: Setup) -> None:
    theta = jnp.arange(1.0, setup.model.n + 1)
    key = jax.random.key(42)
    r = jax.random.multivariate_normal(
        key=key, mean=jnp.zeros(setup.model.n), cov=setup.likelihood.Sigma_obs
    )  # residu impose

    y = setup.model(theta=theta) + r
    expected = jsp.linalg.cho_solve(
        jsp.linalg.cho_factor(setup.likelihood.Sigma_obs, lower=True), r
    )
    computed = setup.likelihood.whitened_residual(y=y, theta=theta)
    assert jnp.allclose(expected, computed)
    assert isinstance(computed, jax.Array)


def test_grad_log_likelihood(setup: Setup) -> None:

    theta = jnp.arange(1.0, setup.model.n + 1)
    key = jax.random.key(42)
    r = jax.random.multivariate_normal(
        key=key, mean=jnp.zeros(setup.model.n), cov=setup.likelihood.Sigma_obs
    )  # residu impose

    y = setup.model(theta=theta) + r
    op = setup.likelihood.jacobian(theta, None)
    expected = op.rmatvec(setup.likelihood.whitened_residual(y, theta, None))
    computed = setup.likelihood.grad_log_likelihood(y=y, theta=theta)
    assert jnp.allclose(expected, computed)
    assert isinstance(computed, jax.Array)


def test_hessian(setup: Setup) -> None:
    theta = jnp.arange(1.0, setup.model.n + 1)
    A = setup.model.jacobian(theta)
    SI = jsp.linalg.cho_factor(setup.likelihood.Sigma_obs, lower=True)
    assert jnp.allclose(-A.T @ SI[0] @ A, setup.likelihood.hessian(theta))
    assert isinstance(A, jax.Array)
    assert isinstance(SI[0], jax.Array)
