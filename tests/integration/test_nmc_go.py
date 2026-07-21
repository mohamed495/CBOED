"""Goal-oriented EIG via nested MC -- oracle against LaplaceEIG in the linear-Gaussian case."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest

import cboed.priors.kernel as kernel
from cboed.core.advection_diffusion import AdvectionDiffusion
from cboed.estimators.laplace import LaplaceEIG
from cboed.estimators.nmc_go import GoalOrientedNestedMonteCarloEIG
from cboed.inference.goal_oriented import GoalOrientedModel
from cboed.inference.linear_model import LinearModel
from cboed.likelihood.gaussian_likelihood import GaussianLikelihood
from cboed.priors.gaussian_process import GaussianPrior, GaussianProcess


class Setup(NamedTuple):
    model: AdvectionDiffusion
    prior: GaussianProcess
    gaussian_prior: GaussianPrior
    likelihood: GaussianLikelihood
    inference: LinearModel
    go: GoalOrientedModel
    n_qoi: int
    B: jax.Array
    Sigma_xi: jax.Array


def _make_setup() -> Setup:
    """Linear (advection-diffusion), small dimensions -- fast."""
    model = AdvectionDiffusion(diffusivity=0.0, velocity=2.0, T=1.0, domain=[0, 1], nt=5, n=6)
    prior = GaussianProcess(kernel=kernel.Gaussian(length_scale=1.0, sigma=1.0), mu=jnp.zeros(6))
    gaussian_prior = GaussianPrior(prior=prior)
    likelihood = GaussianLikelihood(model=model, Sigma_obs=0.1 * jnp.eye(6))
    inference = LinearModel(prior=gaussian_prior, likelihood=likelihood)

    n_qoi = 3
    B = jnp.eye(6)[:n_qoi]
    Sigma_xi = 0.01 * jnp.eye(n_qoi)
    go = GoalOrientedModel(inner=inference, h=lambda eta: eta[:n_qoi], Sigma_theta=Sigma_xi)
    return Setup(model, prior, gaussian_prior, likelihood, inference, go, n_qoi, B, Sigma_xi)


@pytest.fixture
def setup() -> Setup:
    return _make_setup()


def test_go_nmc_returns_finite_scalar(setup):
    est = GoalOrientedNestedMonteCarloEIG(
        likelihood=setup.likelihood,
        prior_eta=setup.gaussian_prior,
        B=setup.B,
        Sigma_xi=setup.Sigma_xi,
    )
    val = est.estimate(jax.random.key(0), n_outer=100, n_inner_theta=100, n_inner_marginal=100)
    assert val.shape == ()
    assert jnp.isfinite(val)


def test_go_nmc_is_deterministic_given_key(setup):
    est = GoalOrientedNestedMonteCarloEIG(
        likelihood=setup.likelihood,
        prior_eta=setup.gaussian_prior,
        B=setup.B,
        Sigma_xi=setup.Sigma_xi,
    )
    a = est.estimate(jax.random.key(1), n_outer=100, n_inner_theta=100, n_inner_marginal=100)
    b = est.estimate(jax.random.key(1), n_outer=100, n_inner_theta=100, n_inner_marginal=100)
    assert jnp.array_equal(a, b)


@pytest.mark.slow
def test_go_nmc_converges_to_laplace_when_linear(setup):
    """Linear model: LaplaceEIG(go) is exact, the goal-oriented MC must converge to it."""
    exact = LaplaceEIG(inference=setup.go).estimate_at(setup.gaussian_prior.mu)
    est = GoalOrientedNestedMonteCarloEIG(
        likelihood=setup.likelihood,
        prior_eta=setup.gaussian_prior,
        B=setup.B,
        Sigma_xi=setup.Sigma_xi,
    )
    val = est.estimate(jax.random.key(2), n_outer=4000, n_inner_theta=2000, n_inner_marginal=2000)
    print(f"\nexact(Laplace,GO)={float(exact):.4f}  MC(GO)={float(val):.4f}")
    assert jnp.abs(val - exact) < 0.3
