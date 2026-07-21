from typing import NamedTuple

import jax.numpy as jnp
import pytest  # type: ignore

import cboed.priors.kernel as kernel
from cboed.core.advection_diffusion import AdvectionDiffusion
from cboed.criteria.optimality import EIG
from cboed.inference.linear_model import LinearModel
from cboed.likelihood.gaussian_likelihood import GaussianLikelihood
from cboed.optim.greedy import GreedyOptimizer
from cboed.optim.greedy_batch import GreedyBatchReopt
from cboed.priors.gaussian_process import GaussianPrior, GaussianProcess


class Setup(NamedTuple):
    model: AdvectionDiffusion
    prior: GaussianProcess
    likelihood: GaussianLikelihood
    inference: LinearModel


@pytest.fixture
def setup() -> Setup:
    model = AdvectionDiffusion(
        diffusivity=0.0,
        velocity=2.0,
        T=1.0,
        domain=[0, 1],
        nt=5,
        n=50,
    )
    prior = GaussianProcess(
        kernel=kernel.Gaussian(length_scale=1.0, sigma=1.0), mu=jnp.ones(model.n)
    )
    likelihood = GaussianLikelihood(model=model, prior=prior, Sigma_obs=jnp.eye(model.n))
    gaussian_prior = GaussianPrior(prior=prior)
    inference = LinearModel(prior=gaussian_prior, likelihood=likelihood)
    return Setup(model, prior, likelihood, inference)


def test_batch_reopt_at_least_as_good(setup):
    """Greedy with reoptimization >= plain greedy."""
    eig = EIG(inference=setup.inference)
    theta = setup.prior.mu

    simple = GreedyOptimizer(criterion=eig).run(theta, 2, 10)
    batch = GreedyBatchReopt(criterion=eig).run(theta, 2, 10)

    score_simple = eig.evaluate(theta, simple.design)
    score_batch = eig.evaluate(theta, batch.design)
    assert score_batch >= score_simple - 1e-8


def test_batch_reopt_valid_design(setup):
    """Valid design: correct count, no duplicates."""
    eig = EIG(inference=setup.inference)
    result = GreedyBatchReopt(criterion=eig).run(setup.prior.mu, 3, 20)
    assert len(result.design) == 3
    assert len(set(result.design.tolist())) == 3
