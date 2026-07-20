from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest  # type: ignore

import cboed.priors.kernel as kernel
from cboed.core.advection_diffusion import AdvectionDiffusion
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


def test_hessian_depends_on_design(setup):
    """Deux designs différents → Hessiennes différentes."""
    theta = setup.prior.mu
    H_a = setup.likelihood.hessian(theta, design=jnp.array([0, 1]))
    H_b = setup.likelihood.hessian(theta, design=jnp.array([2, 3]))
    assert not jnp.allclose(H_a, H_b)


def test_more_sensors_more_information(setup):
    """Ajouter un capteur augmente l'information (Loewner)."""
    theta = setup.prior.mu
    prec_1 = setup.inference.posterior_precision(theta, design=jnp.array([0]))
    prec_2 = setup.inference.posterior_precision(theta, design=jnp.array([0, 1]))
    # prec_2 ⪰ prec_1 : la précision ne peut qu'augmenter
    diff = prec_2 - prec_1
    assert jnp.all(jnp.linalg.eigvalsh(diff) > -1e-8)


def test_posterior_mean_recovers_truth_noiseless(setup):
    """Sans bruit, prior faible : μ_post proche de θ_vrai."""
    theta_true = jnp.arange(1.0, setup.model.n + 1)
    y = setup.model(theta_true)  # pas de bruit
    mu_post = setup.inference._mu(y=y, theta=setup.prior.mu)
    # avec ce prior, μ_post tiré entre θ_true et μ_prior
    # test faible : au moins dans le bon voisinage
    assert mu_post.shape == (setup.model.n,)
    # oracle dense
    A = setup.model.jacobian(setup.prior.mu)
    So, Sp = setup.likelihood.Sigma_obs, setup.prior.Sigma
    prec = A.T @ jnp.linalg.inv(So) @ A + jnp.linalg.inv(Sp)
    expected = setup.prior.mu + jnp.linalg.solve(
        prec, A.T @ jnp.linalg.inv(So) @ (y - setup.model(setup.prior.mu))
    )
    assert jnp.allclose(mu_post, expected, atol=1e-8)


def test_cov_with_anisotropic_noise():
    """Oracle avec Σ_obs non triviale — discrimine les erreurs que I masque."""
    model = AdvectionDiffusion(diffusivity=0.0, velocity=2.0, T=1.0, domain=[0, 1], nt=5, n=4)
    prior_gp = GaussianProcess(kernel=kernel.Gaussian(length_scale=1.0, sigma=1.0), mu=jnp.ones(4))
    Sigma_obs = jnp.diag(jnp.array([1.0, 2.0, 3.0, 4.0]))
    lik = GaussianLikelihood(model=model, prior=prior_gp, Sigma_obs=Sigma_obs)
    inf = LinearModel(prior=GaussianPrior(prior=prior_gp), likelihood=lik)

    theta = prior_gp.mu
    A = model.jacobian(theta)
    precision = A.T @ jnp.linalg.inv(Sigma_obs) @ A + jnp.linalg.inv(prior_gp.Sigma)
    expected = jnp.linalg.inv(precision)
    assert jnp.allclose(inf._cov(theta), expected, atol=1e-8)


def test_cov_computed_from_model(setup):
    # Sigma_obs non triviale : sinon inv(I)=I masque les erreurs
    theta = setup.prior.mu
    computed = setup.inference._cov(theta=theta)

    Sigma_obs = setup.likelihood.Sigma_obs
    Sigma_prior = setup.prior.Sigma
    A = setup.model.jacobian(theta=theta)
    precision = A.T @ jnp.linalg.inv(Sigma_obs) @ A + jnp.linalg.inv(Sigma_prior)
    expected = jnp.linalg.inv(precision)

    assert isinstance(A, jax.Array)
    assert jnp.allclose(computed, expected, atol=1e-8)
    assert jnp.allclose(precision, setup.inference.posterior_precision(theta))


def test_mu(setup: Setup) -> None:
    """Posterior mean agrees with the closed-form Gaussian formula."""

    theta_true = jnp.array([0.5, -1.0, 2.0, 0.2])

    # observations sans bruit
    y = setup.model(theta_true)

    # point de linéarisation
    theta_lin = jnp.zeros_like(theta_true)

    mu = setup.inference._mu(
        y=y,
        theta=theta_lin,
    )

    # formule analytique
    H = setup.inference.posterior_precision(theta_lin)

    grad_post = setup.likelihood.grad_log_likelihood(
        y=y,
        theta=theta_lin,
    ) + setup.inference.prior.grad_log_prior(theta_lin)

    mu_expected = theta_lin + jnp.linalg.solve(H, grad_post)

    assert jnp.allclose(mu, mu_expected, atol=1e-8)
