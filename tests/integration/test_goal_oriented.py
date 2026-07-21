from typing import NamedTuple

import jax.numpy as jnp
import pytest  # type: ignore

import cboed.priors.kernel as kernel
from cboed.core.advection_diffusion import AdvectionDiffusion
from cboed.criteria.optimality import EIG
from cboed.inference.goal_oriented import GoalOrientedModel
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
# The wrapper breaks nothing: h = identity -> standard inference
# ─────────────────────────────────────────────────────────


def test_go_identity_covariance_equals_standard(setup):
    """h = Id, Σ_θ ≈ 0 -> QoI covariance = standard covariance."""
    n = setup.model.n
    go = GoalOrientedModel(
        inner=setup.inference,
        h=lambda eta: eta,
        Sigma_theta=jnp.zeros((n, n)),
    )
    eta = setup.prior.mu
    cov_go = go.posterior_covariance_qoi(eta)
    cov_std = setup.inference._cov(eta)
    assert jnp.allclose(cov_go, cov_std, atol=1e-8)


def test_go_identity_eig_equals_standard(setup):
    """h = Id -> goal-oriented EIG = standard EIG."""
    n = setup.model.n
    go = GoalOrientedModel(
        inner=setup.inference,
        h=lambda eta: eta,
        Sigma_theta=1e-10 * jnp.eye(n),  # jitter, avoids a singular slogdet
    )
    eig_go = EIG(inference=go).evaluate(setup.prior.mu)
    eig_std = EIG(inference=setup.inference).evaluate(setup.prior.mu)
    assert jnp.allclose(eig_go, eig_std, atol=1e-6)


# ─────────────────────────────────────────────────────────
# The real QoI: first half of the segment
# ─────────────────────────────────────────────────────────


def test_go_half_qoi_dense_oracle(setup):
    """θ = first half: Σ_θ|Y = H Σ_η|Y Hᵀ + Σ_θ, dense oracle."""
    n = setup.model.n
    n_qoi = n // 2
    Sigma_theta = 0.01 * jnp.eye(n_qoi)

    go = GoalOrientedModel(
        inner=setup.inference,
        h=lambda eta: eta[:n_qoi],  # first half
        Sigma_theta=Sigma_theta,
    )
    eta = setup.prior.mu

    # dense oracle: H = [I_{n/2} | 0]
    H = jnp.eye(n)[:n_qoi]
    cov_eta = setup.inference._cov(eta)
    expected = H @ cov_eta @ H.T + Sigma_theta

    computed = go.posterior_covariance_qoi(eta)
    assert computed.shape == (n_qoi, n_qoi)
    assert jnp.allclose(computed, expected, atol=1e-8)


def test_go_qoi_covariance_is_symmetric_pd(setup):
    """The QoI covariance remains symmetric positive definite."""
    n = setup.model.n
    n_qoi = n // 2
    go = GoalOrientedModel(
        inner=setup.inference,
        h=lambda eta: eta[:n_qoi],
        Sigma_theta=0.01 * jnp.eye(n_qoi),
    )
    cov = go.posterior_covariance_qoi(setup.prior.mu)
    assert jnp.allclose(cov, cov.T)
    assert jnp.all(jnp.linalg.eigvalsh(cov) > 0)


def test_go_posterior_less_than_prior_qoi(setup):
    """Observing also reduces uncertainty on the QoI: Σ_θ|Y ⪯ Σ_θ."""
    n = setup.model.n
    n_qoi = n // 2
    go = GoalOrientedModel(
        inner=setup.inference,
        h=lambda eta: eta[:n_qoi],
        Sigma_theta=0.01 * jnp.eye(n_qoi),
    )
    eta = setup.prior.mu
    diff = go.prior_covariance_qoi(eta) - go.posterior_covariance_qoi(eta)
    assert jnp.all(jnp.linalg.eigvalsh(diff) > -1e-8)


# ─────────────────────────────────────────────────────────
# h linear -> QoI prior independent of the point (validates eta=None)
# ─────────────────────────────────────────────────────────


def test_prior_qoi_independent_of_eta_when_linear(setup):
    """h linear: the QoI prior does not depend on the linearization point.

    This is what makes the eta=None default valid in log_det_prior_precision.
    """
    n = setup.model.n
    n_qoi = n // 2
    go = GoalOrientedModel(
        inner=setup.inference,
        h=lambda eta: eta[:n_qoi],
        Sigma_theta=0.01 * jnp.eye(n_qoi),
    )
    c1 = go.prior_covariance_qoi(jnp.zeros(n))
    c2 = go.prior_covariance_qoi(jnp.ones(n) * 5.0)
    assert jnp.allclose(c1, c2)  # constant iff h linear
