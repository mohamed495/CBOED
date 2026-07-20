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
    """Chaîne complète sur Burgers, à un λ donné."""
    model = Burgers(diffusivity=1.0, lambda_=lambda_, T=1.0, domain=[0, 1], nt=5, n=4)
    prior = GaussianProcess(
        kernel=kernel.Gaussian(length_scale=1.0, sigma=1.0), mu=jnp.ones(model.n)
    )
    likelihood = GaussianLikelihood(model=model, prior=prior, Sigma_obs=jnp.eye(model.n))
    inference = LinearModel(prior=GaussianPrior(prior=prior), likelihood=likelihood)
    return Setup(model, prior, likelihood, inference)


@pytest.fixture
def setup_linear() -> Setup:
    """λ=0 : modèle linéaire, Laplace exact."""
    return _make_setup(lambda_=0.0)


@pytest.fixture
def setup_nonlinear() -> Setup:
    """λ=1 : modèle non-linéaire, Laplace approximatif."""
    return _make_setup(lambda_=1.0)


# ─────────────────────────────────────────────────────────
# À λ=0 : Laplace se réduit à l'EIG exacte
# ─────────────────────────────────────────────────────────


def test_laplace_equals_exact_at_lambda_zero(setup_linear):
    """Modèle linéaire → Laplace = EIG exact."""
    laplace = LaplaceEIG(inference=setup_linear.inference)
    exact = EIG(inference=setup_linear.inference).evaluate(setup_linear.prior.mu)
    assert jnp.allclose(laplace.estimate(), exact, atol=1e-10)


def test_laplace_independent_of_point_when_linear(setup_linear):
    """À λ=0, la linéarisation est la même partout : point sans effet."""
    laplace = LaplaceEIG(inference=setup_linear.inference)
    at_zero = laplace.estimate_at(jnp.zeros(setup_linear.model.n))
    at_five = laplace.estimate_at(jnp.ones(setup_linear.model.n) * 5.0)
    assert jnp.allclose(at_zero, at_five, atol=1e-10)


# ─────────────────────────────────────────────────────────
# À λ>0 : Laplace dépend du point (non-linéarité)
# ─────────────────────────────────────────────────────────


def test_laplace_depends_on_point_when_nonlinear(setup_nonlinear):
    """À λ=1, le point de linéarisation compte."""
    laplace = LaplaceEIG(inference=setup_nonlinear.inference)
    at_zero = laplace.estimate_at(jnp.zeros(setup_nonlinear.model.n))
    at_five = laplace.estimate_at(jnp.ones(setup_nonlinear.model.n) * 5.0)
    assert not jnp.allclose(at_zero, at_five)


# ─────────────────────────────────────────────────────────
# Propriétés générales, quel que soit λ
# ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0])
def test_laplace_returns_positive_scalar(lam):
    """Laplace renvoie un scalaire positif dans tous les régimes."""
    setup = _make_setup(lambda_=lam)
    laplace = LaplaceEIG(inference=setup.inference)
    val = laplace.estimate()
    assert val.shape == ()
    assert val > 0


def test_laplace_estimate_matches_estimate_at_prior_mean(setup_nonlinear):
    """estimate() = estimate_at(μ_prior), par définition du défaut."""
    laplace = LaplaceEIG(inference=setup_nonlinear.inference)
    default = laplace.estimate()
    explicit = laplace.estimate_at(setup_nonlinear.prior.mu)
    assert jnp.allclose(default, explicit)


def test_laplace_with_design(setup_nonlinear):
    """Laplace fonctionne avec un design restreint."""
    laplace = LaplaceEIG(inference=setup_nonlinear.inference)
    design = jnp.array([0, 2])
    val = laplace.estimate(design=design)
    assert val.shape == ()
    assert val > 0
