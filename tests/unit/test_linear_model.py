from typing import NamedTuple

import jax.numpy as jnp
import pytest  # type: ignore

import cboed.priors.kernel as kernel
from cboed.core.advection_diffusion import AdvectionDiffusion
from cboed.inference.linear_model import LinearModel
from cboed.likelihood.gaussian_likelihood import gaussianLikelihood
from cboed.priors.gaussian_process import GaussianPrior, GaussianProcess


class Setup(NamedTuple):
    model: AdvectionDiffusion
    prior: GaussianProcess
    likelihood: gaussianLikelihood
    inference: LinearModel


@pytest.fixture
def setup() -> Setup:
    model = AdvectionDiffusion(
        diffusivity=0.0,
        velocity=2.0,
        T=1.0,
        domain=[0, 1],
        nt=5,
        n=4,
    )
    prior = GaussianProcess(
        kernel=kernel.Gaussian(length_scale=1.0, sigma=1.0), mu=jnp.ones(model.n)
    )
    likelihood = gaussianLikelihood(
        model=model, prior=prior, Sigma_obs=jnp.eye(model.n)
    )
    gaussian_prior = GaussianPrior(prior=prior)
    inference = LinearModel(prior=gaussian_prior, likelihood=likelihood)
    return Setup(model, prior, likelihood, inference)


def test_cov_is_symmetric_pd(setup):
    cov = setup.inference._cov(theta=setup.prior.mu)
    assert jnp.allclose(cov, cov.T)  # symétrique
    assert jnp.all(jnp.linalg.eigvalsh(cov) > 0)  # définie positive


def test_posterior_less_than_prior(setup):
    """Observed reduce incertainty : Γ_post ⪯ Γ_prior."""
    cov_post = setup.inference._cov(theta=setup.prior.mu)
    cov_prior = setup.prior.Sigma
    # Γ_prior - Γ_post doit être semi-définie positive
    diff = cov_prior - cov_post
    assert jnp.all(jnp.linalg.eigvalsh(diff) > -1e-8)
