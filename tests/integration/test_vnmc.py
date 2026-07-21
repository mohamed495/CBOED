from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest  # type: ignore

import cboed.priors.kernel as kernel
from cboed.core.burgers import Burgers
from cboed.criteria.optimality import EIG
from cboed.estimators.laplace import LaplaceEIG
from cboed.estimators.vnmc import VariationalNMCEIG
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
    model = Burgers(diffusivity=0.01, lambda_=lambda_, T=1.0, domain=[0, 1], nt=5, n=4)
    prior = GaussianProcess(
        kernel=kernel.Gaussian(length_scale=1.0, sigma=1.0), mu=jnp.ones(model.n)
    )
    gaussian_prior = GaussianPrior(prior=prior)
    likelihood = GaussianLikelihood(model=model, prior=prior, Sigma_obs=0.001 * jnp.eye(model.n))
    inference = LinearModel(prior=gaussian_prior, likelihood=likelihood)
    return Setup(model, prior, gaussian_prior, likelihood, inference)


@pytest.fixture
def setup_linear() -> Setup:
    """λ=0: linear-Gaussian, the exact EIG serves as the oracle."""
    return _make_setup(lambda_=0.0)


@pytest.fixture
def setup_nonlinear() -> Setup:
    """λ=1: nonlinear, no closed-form oracle."""
    return _make_setup(lambda_=1.0)


# ─────────────────────────────────────────────────────────
# Sanity -- fast
# ─────────────────────────────────────────────────────────


def test_vnmc_returns_finite_scalar(setup_linear):
    vnmc = VariationalNMCEIG(
        likelihood=setup_linear.likelihood,
        prior=setup_linear.gaussian_prior,
        inference=setup_linear.inference,
    )
    val = vnmc.estimate(jax.random.key(0), n_outer=200, n_inner=200)
    assert val.shape == ()
    assert jnp.isfinite(val)


def test_vnmc_is_deterministic_given_key(setup_linear):
    """Same key -> same estimate."""
    vnmc = VariationalNMCEIG(
        likelihood=setup_linear.likelihood,
        prior=setup_linear.gaussian_prior,
        inference=setup_linear.inference,
    )
    a = vnmc.estimate(jax.random.key(0), n_outer=200, n_inner=200)
    b = vnmc.estimate(jax.random.key(0), n_outer=200, n_inner=200)
    assert jnp.array_equal(a, b)


def test_vnmc_with_design(setup_linear):
    """VNMC works with a restricted design."""
    vnmc = VariationalNMCEIG(
        likelihood=setup_linear.likelihood,
        prior=setup_linear.gaussian_prior,
        inference=setup_linear.inference,
    )
    val = vnmc.estimate(jax.random.key(0), design=jnp.array([0, 2]), n_outer=200, n_inner=200)
    assert val.shape == ()
    assert jnp.isfinite(val)


# ─────────────────────────────────────────────────────────
# The problem must be informative -- otherwise nothing is measurable
# ─────────────────────────────────────────────────────────


def test_problem_is_informative(setup_linear):
    """Safety net: the EIG must be of order >= 1 nat.

    With Sigma_obs = I, the EIG drops to ~5e-3 and all the Monte Carlo tests
    run in numerical noise.
    """
    eig = EIG(inference=setup_linear.inference).evaluate(setup_linear.prior.mu)
    assert eig > 1.0


# ─────────────────────────────────────────────────────────
# LG convergence -- slow
# ─────────────────────────────────────────────────────────


@pytest.mark.slow
def test_vnmc_converges_to_exact_lg(setup_linear):
    """In the LG case, the Laplace proposal is the exact posterior:
    constant importance weights, zero variance, VNMC ≈ EIG."""
    exact = EIG(inference=setup_linear.inference).evaluate(setup_linear.prior.mu)
    vnmc = VariationalNMCEIG(
        likelihood=setup_linear.likelihood,
        prior=setup_linear.gaussian_prior,
        inference=setup_linear.inference,
    ).estimate(jax.random.key(0), n_outer=3000, n_inner=3000)
    assert jnp.abs(vnmc - exact) < 0.1


@pytest.mark.slow
def test_vnmc_is_upper_bound_lg(setup_linear):
    """VNMC is an upper bound (Jensen on an internal unbiased estimator
    of p(y)). Tolerance: Monte Carlo noise."""
    exact = EIG(inference=setup_linear.inference).evaluate(setup_linear.prior.mu)
    vnmc = VariationalNMCEIG(
        likelihood=setup_linear.likelihood,
        prior=setup_linear.gaussian_prior,
        inference=setup_linear.inference,
    ).estimate(jax.random.key(0), n_outer=3000, n_inner=3000)
    assert vnmc >= exact - 0.1


# ─────────────────────────────────────────────────────────
# Nonlinearity: the Laplace / Monte Carlo gap grows with λ
# ─────────────────────────────────────────────────────────


@pytest.mark.slow
def test_laplace_error_grows_with_lambda():
    """Laplace (linearized approximation) diverges from the Monte Carlo
    ground truth as nonlinearity increases."""
    key = jax.random.key(0)
    gaps = []
    for lam in [0.0, 0.5, 1.0]:
        s = _make_setup(lambda_=lam)
        eig_laplace = LaplaceEIG(inference=s.inference).estimate()
        eig_vnmc = VariationalNMCEIG(
            likelihood=s.likelihood,
            prior=s.gaussian_prior,
            inference=s.inference,
        ).estimate(key, n_outer=3000, n_inner=3000)
        gaps.append(float(jnp.abs(eig_laplace - eig_vnmc)))

    assert gaps[0] < 0.2  # λ=0: Laplace exact, VNMC exact
    assert gaps[-1] > gaps[0]  # λ=1: Laplace diverges
