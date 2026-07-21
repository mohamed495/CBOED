from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest  # type: ignore

import cboed.priors.kernel as kernel
from cboed.core.burgers import Burgers
from cboed.criteria.optimality import EIG
from cboed.estimators.nmc import NestedMonteCarloEIG
from cboed.inference.linear_model import LinearModel
from cboed.likelihood.gaussian_likelihood import GaussianLikelihood
from cboed.priors.gaussian_process import GaussianPrior, GaussianProcess


class Setup(NamedTuple):
    model: Burgers
    prior: GaussianProcess
    gaussian_prior: GaussianPrior
    likelihood: GaussianLikelihood
    inference: LinearModel


def _make_setup(lambda_: float) -> Setup:
    model = Burgers(diffusivity=1.0, lambda_=lambda_, T=1.0, domain=[0, 1], nt=5, n=4)
    prior = GaussianProcess(
        kernel=kernel.Gaussian(length_scale=1.0, sigma=1.0), mu=jnp.ones(model.n)
    )
    gaussian_prior = GaussianPrior(prior=prior)
    likelihood = GaussianLikelihood(model=model, prior=prior, Sigma_obs=jnp.eye(model.n))
    inference = LinearModel(prior=gaussian_prior, likelihood=likelihood)
    return Setup(model, prior, gaussian_prior, likelihood, inference)


@pytest.fixture
def setup_linear() -> Setup:
    """λ=0: linear-Gaussian, NMC must recover the exact EIG."""
    return _make_setup(lambda_=0.0)


@pytest.fixture
def setup_nonlinear() -> Setup:
    """λ=1: nonlinear, no closed-form oracle."""
    return _make_setup(lambda_=1.0)


# ─────────────────────────────────────────────────────────
# Sanity: NMC returns a finite scalar
# ─────────────────────────────────────────────────────────


def test_nmc_returns_finite_scalar(setup_linear):
    nmc = NestedMonteCarloEIG(likelihood=setup_linear.likelihood, prior=setup_linear.gaussian_prior)
    val = nmc.estimate(jax.random.key(0), n_outer=200, n_inner=200)
    assert val.shape == ()
    assert jnp.isfinite(val)


def test_nmc_is_deterministic_given_key(setup_linear):
    """Same key -> same estimate."""
    nmc = NestedMonteCarloEIG(likelihood=setup_linear.likelihood, prior=setup_linear.gaussian_prior)
    a = nmc.estimate(jax.random.key(0), n_outer=200, n_inner=200)
    b = nmc.estimate(jax.random.key(0), n_outer=200, n_inner=200)
    assert jnp.array_equal(a, b)


# ─────────────────────────────────────────────────────────
# Convergence: in LG, NMC -> exact EIG (slow)
# ─────────────────────────────────────────────────────────


@pytest.mark.slow
def test_nmc_converges_to_exact_lg(setup_linear):
    """In the linear-Gaussian case, NMC recovers the closed-form formula."""
    exact = EIG(inference=setup_linear.inference).evaluate(setup_linear.prior.mu)
    nmc = NestedMonteCarloEIG(likelihood=setup_linear.likelihood, prior=setup_linear.gaussian_prior)
    est = nmc.estimate(jax.random.key(0), n_outer=5000, n_inner=5000)
    assert jnp.abs(est - exact) < 0.15


@pytest.mark.slow
def test_nmc_bias_decreases_with_inner(setup_linear):
    """The NMC bias (lower bound) decreases as n_inner increases."""
    exact = EIG(inference=setup_linear.inference).evaluate(setup_linear.prior.mu)
    nmc = NestedMonteCarloEIG(likelihood=setup_linear.likelihood, prior=setup_linear.gaussian_prior)
    est_small = nmc.estimate(jax.random.key(0), n_outer=3000, n_inner=100)
    est_large = nmc.estimate(jax.random.key(0), n_outer=3000, n_inner=3000)
    # NMC underestimates; the larger M, the closer it gets to the exact value
    assert est_large >= est_small - 0.05
    assert jnp.abs(est_large - exact) <= jnp.abs(est_small - exact) + 0.05


# ─────────────────────────────────────────────────────────
# The core test: the Laplace/NMC gap grows with λ
# ─────────────────────────────────────────────────────────


@pytest.mark.slow
def test_laplace_error_grows_with_lambda():
    """The gap between Laplace (approximate) and NMC (ground truth) measures nonlinearity."""
    from cboed.estimators.laplace import LaplaceEIG

    key = jax.random.key(0)
    gaps = []
    for lam in [0.0, 0.5, 1.0]:
        s = _make_setup(lambda_=lam)
        eig_laplace = LaplaceEIG(inference=s.inference).estimate()
        eig_nmc = NestedMonteCarloEIG(likelihood=s.likelihood, prior=s.gaussian_prior).estimate(
            key, n_outer=4000, n_inner=4000
        )
        gaps.append(float(jnp.abs(eig_laplace - eig_nmc)))

    assert gaps[0] < 0.2  # λ=0: Laplace ≈ NMC
    assert gaps[-1] > gaps[0]  # λ=1: Laplace diverges
