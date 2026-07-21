from typing import NamedTuple

import jax.numpy as jnp
import pytest  # type: ignore

import cboed.priors.kernel as kernel
from cboed.core.advection_diffusion import AdvectionDiffusion
from cboed.criteria.optimality import EIG, AOptimal, DOptimal
from cboed.inference.linear_model import LinearModel
from cboed.likelihood.gaussian_likelihood import GaussianLikelihood
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
        n=4,
    )
    prior = GaussianProcess(
        kernel=kernel.Gaussian(length_scale=1.0, sigma=1.0), mu=jnp.ones(model.n)
    )
    likelihood = GaussianLikelihood(model=model, prior=prior, Sigma_obs=jnp.eye(model.n))
    gaussian_prior = GaussianPrior(prior=prior)
    inference = LinearModel(prior=gaussian_prior, likelihood=likelihood)
    return Setup(model, prior, likelihood, inference)


# ─────────────────────────────────────────────────────────
# Types and shapes
# ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("Crit", [EIG, DOptimal, AOptimal])
def test_evaluate_returns_scalar(setup, Crit):
    crit = Crit(inference=setup.inference)
    val = crit.evaluate(theta=setup.prior.mu)
    assert val.shape == ()  # JAX scalar
    assert jnp.isdtype(val.dtype, "real floating")


@pytest.mark.parametrize("Crit", [EIG, DOptimal, AOptimal])
def test_evaluate_is_finite(setup, Crit):
    crit = Crit(inference=setup.inference)
    val = crit.evaluate(theta=setup.prior.mu)
    assert jnp.isfinite(val)


# ─────────────────────────────────────────────────────────
# Mathematical invariants
# ─────────────────────────────────────────────────────────


def test_eig_is_positive(setup):
    """Observing can only add information."""
    eig = EIG(inference=setup.inference)
    assert eig.evaluate(theta=setup.prior.mu) > 0


def test_doptimal_equals_log_det_posterior(setup):
    """D-opt = log det Γ_post⁻¹, by definition."""
    dopt = DOptimal(inference=setup.inference)
    val = dopt.evaluate(theta=setup.prior.mu)
    expected = setup.inference.log_det_posterior_precision(setup.prior.mu)
    assert jnp.allclose(val, expected)


def test_aoptimal_is_negative(setup):
    """A-opt = -tr(Γ_post) < 0 (we maximize, hence -trace)."""
    aopt = AOptimal(inference=setup.inference)
    assert aopt.evaluate(theta=setup.prior.mu) < 0


def test_doptimal_matches_sum_log_eigvals(setup):
    """log det = Σ log λ: the two computation paths agree."""
    dopt = DOptimal(inference=setup.inference)
    via_logdet = dopt.evaluate(theta=setup.prior.mu)
    eigs = jnp.linalg.eigvalsh(setup.inference.posterior_precision(setup.prior.mu))
    via_eigvals = jnp.sum(jnp.log(eigs))
    assert jnp.allclose(via_logdet, via_eigvals, atol=1e-8)


# ─────────────────────────────────────────────────────────
# Values against a dense oracle (independent path)
# ─────────────────────────────────────────────────────────


def test_eig_matches_dense(setup):
    eig = EIG(inference=setup.inference)
    val = eig.evaluate(theta=setup.prior.mu)

    ld_post = jnp.linalg.slogdet(setup.inference.posterior_precision(setup.prior.mu))[1]
    ld_prior = -jnp.linalg.slogdet(setup.prior.Sigma)[1]
    expected = 0.5 * (ld_post - ld_prior)

    assert jnp.allclose(val, expected, atol=1e-8)


def test_aoptimal_matches_trace(setup):
    """A-opt = -tr(Γ_post) = -Σ 1/λᵢ, against the direct trace."""
    aopt = AOptimal(inference=setup.inference)
    val = aopt.evaluate(theta=setup.prior.mu)

    cov = setup.inference._cov(setup.prior.mu)
    expected = -jnp.trace(cov)
    assert jnp.allclose(val, expected, atol=1e-8)


# ─────────────────────────────────────────────────────────
# Design behavior -- to activate once H(ξ) exists
# ─────────────────────────────────────────────────────────
def test_eig_submodular(setup):
    """Diminishing returns: the marginal gain of the 2nd sensor <= gain of the 1st."""
    eig = EIG(inference=setup.inference)
    theta = setup.prior.mu
    e0 = eig.evaluate(theta, design=jnp.array([0]))
    e1 = eig.evaluate(theta, design=jnp.array([0, 1]))
    e2 = eig.evaluate(theta, design=jnp.array([0, 1, 2]))
    gain_1 = e1 - e0
    gain_2 = e2 - e1
    assert gain_2 <= gain_1 + 1e-10


def test_different_designs_give_different_scores(setup):
    eig = EIG(inference=setup.inference)
    theta = setup.prior.mu
    a = eig.evaluate(theta, design=jnp.array([0, 1]))
    b = eig.evaluate(theta, design=jnp.array([2, 3]))
    assert not jnp.allclose(a, b)
