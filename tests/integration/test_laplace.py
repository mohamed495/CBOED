from typing import NamedTuple

import jax.numpy as jnp
import pytest  # type: ignore

import cboed.priors.kernel as kernel
from cboed.core.burgers import Burgers
from cboed.criteria.optimality import EIG
from cboed.estimators.laplace import LaplaceEIG
from cboed.inference.linear_model import LinearModel
from cboed.likelihood.gaussian_likelihood import GaussianLikelihood
from cboed.priors.gaussian_process import GaussianPrior, GaussianProcess


class Setup(NamedTuple):
    model: Burgers
    prior: GaussianProcess
    likelihood: GaussianLikelihood
    inference: LinearModel


def _make_setup(lambda_: float) -> Setup:
    """Full chain on Burgers, at a given λ."""
    model = Burgers(diffusivity=1.0, lambda_=lambda_, T=1.0, domain=[0, 1], nt=5, n=4)
    prior = GaussianProcess(
        kernel=kernel.Gaussian(length_scale=1.0, sigma=1.0), mu=jnp.ones(model.n)
    )
    likelihood = GaussianLikelihood(model=model, prior=prior, Sigma_obs=jnp.eye(model.n))
    inference = LinearModel(prior=GaussianPrior(prior=prior), likelihood=likelihood)
    return Setup(model, prior, likelihood, inference)


@pytest.fixture
def setup_linear() -> Setup:
    """λ=0: linear model, Laplace exact."""
    return _make_setup(lambda_=0.0)


@pytest.fixture
def setup_nonlinear() -> Setup:
    """λ=1: nonlinear model, Laplace approximate."""
    return _make_setup(lambda_=1.0)


# ─────────────────────────────────────────────────────────
# At λ=0: Laplace reduces to the exact EIG
# ─────────────────────────────────────────────────────────


def test_laplace_equals_exact_at_lambda_zero(setup_linear):
    """Linear model -> Laplace = exact EIG."""
    laplace = LaplaceEIG(inference=setup_linear.inference)
    exact = EIG(inference=setup_linear.inference).evaluate(setup_linear.prior.mu)
    assert jnp.allclose(laplace.estimate(), exact, atol=1e-10)


def test_laplace_independent_of_point_when_linear(setup_linear):
    """At λ=0, the linearization is the same everywhere: point has no effect."""
    laplace = LaplaceEIG(inference=setup_linear.inference)
    at_zero = laplace.estimate_at(jnp.zeros(setup_linear.model.n))
    at_five = laplace.estimate_at(jnp.ones(setup_linear.model.n) * 5.0)
    assert jnp.allclose(at_zero, at_five, atol=1e-10)


# ─────────────────────────────────────────────────────────
# At λ>0: Laplace depends on the point (nonlinearity)
# ─────────────────────────────────────────────────────────


def test_laplace_depends_on_point_when_nonlinear(setup_nonlinear):
    """At λ=1, the linearization point matters."""
    laplace = LaplaceEIG(inference=setup_nonlinear.inference)
    at_zero = laplace.estimate_at(jnp.zeros(setup_nonlinear.model.n))
    at_five = laplace.estimate_at(jnp.ones(setup_nonlinear.model.n) * 5.0)
    assert not jnp.allclose(at_zero, at_five)


# ─────────────────────────────────────────────────────────
# General properties, regardless of λ
# ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0])
def test_laplace_returns_positive_scalar(lam):
    """Laplace returns a positive scalar in all regimes."""
    setup = _make_setup(lambda_=lam)
    laplace = LaplaceEIG(inference=setup.inference)
    val = laplace.estimate()
    assert val.shape == ()
    assert val > 0


def test_laplace_estimate_matches_estimate_at_prior_mean(setup_nonlinear):
    """estimate() = estimate_at(μ_prior), by definition of the default."""
    laplace = LaplaceEIG(inference=setup_nonlinear.inference)
    default = laplace.estimate()
    explicit = laplace.estimate_at(setup_nonlinear.prior.mu)
    assert jnp.allclose(default, explicit)


def test_laplace_with_design(setup_nonlinear):
    """Laplace works with a restricted design."""
    laplace = LaplaceEIG(inference=setup_nonlinear.inference)
    design = jnp.array([0, 2])
    val = laplace.estimate(design=design)
    assert val.shape == ()
    assert val > 0
