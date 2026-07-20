from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import pytest  # type: ignore

import cboed.priors.kernel as kernel
from cboed.core.advection_diffusion import AdvectionDiffusion
from cboed.core.base import ForwardModel
from cboed.core.linear_operator import LinearizedOperator
from cboed.likelihood.gaussian_likelihood import GaussianLikelihood
from cboed.priors.gaussian_process import GaussianProcess


class Setup(NamedTuple):
    model: AdvectionDiffusion
    prior: GaussianProcess
    likelihood: GaussianLikelihood


@pytest.fixture
def setup() -> Setup:
    model = model = AdvectionDiffusion(
        diffusivity=0.0,
        velocity=2.0,
        T=1,
        domain=[0, 1],
        nt=5,
        n=4,
    )
    prior = GaussianProcess(
        kernel=kernel.Gaussian(length_scale=1.0, sigma=1.0), mu=jnp.ones(model.n)
    )
    likelihood = GaussianLikelihood(model=model, prior=prior, Sigma_obs=jnp.eye(model.n))
    return Setup(model, prior, likelihood)


def test_properties(setup: Setup) -> None:
    assert jnp.allclose(setup.likelihood.Sigma_obs, jnp.eye(setup.model.n))
    assert isinstance(setup.likelihood.model, ForwardModel)


def test_log_likelihood_value(setup: Setup) -> None:
    theta = jnp.arange(1.0, setup.model.n + 1)
    y = setup.model(theta=theta)

    computed = setup.likelihood.log_likelihood(y=y, theta=theta)
    # r = 0, Sigma_obs = I  =>  log p = -n/2 * log(2 pi)
    expected = -0.5 * setup.model.n * jnp.log(2 * jnp.pi)
    assert jnp.allclose(computed, expected)
    assert isinstance(computed, jax.Array)


def test_log_likelihood_quadratic(setup: Setup) -> None:
    theta = jnp.arange(1.0, setup.model.n + 1)
    key = jax.random.key(42)
    r = jax.random.multivariate_normal(
        key=key, mean=jnp.zeros(setup.model.n), cov=setup.likelihood.Sigma_obs
    )  # residu impose

    y = setup.model(theta=theta) + r

    computed = setup.likelihood.log_likelihood(y=y, theta=theta)
    expected = -0.5 * (setup.model.n * jnp.log(2 * jnp.pi) + r @ r)
    assert jnp.allclose(computed, expected)


def test_jacobian_operator(setup: Setup) -> None:
    theta = jnp.arange(1.0, setup.model.n + 1)
    op = setup.likelihood.jacobian_operator(theta=theta)
    assert isinstance(op, LinearizedOperator)


def test_jacobian_dense_matches_operator(setup: Setup) -> None:
    """`jacobian` matérialise `jacobian_operator` — colonnes, pas lignes."""
    theta = jnp.arange(1.0, setup.model.n + 1)
    op = setup.likelihood.jacobian_operator(theta=theta)
    J = setup.likelihood.jacobian(theta=theta)
    v = jnp.arange(1.0, setup.model.n + 1)
    assert jnp.allclose(J @ v, op.matvec(v))


def test_whitened_residual(setup: Setup) -> None:
    theta = jnp.arange(1.0, setup.model.n + 1)
    key = jax.random.key(42)
    r = jax.random.multivariate_normal(
        key=key, mean=jnp.zeros(setup.model.n), cov=setup.likelihood.Sigma_obs
    )  # residu impose

    y = setup.model(theta=theta) + r
    expected = jsp.linalg.cho_solve(
        jsp.linalg.cho_factor(setup.likelihood.Sigma_obs, lower=True), r
    )
    computed = setup.likelihood.precision_weighted_residual(y=y, theta=theta)
    assert jnp.allclose(expected, computed)
    assert isinstance(computed, jax.Array)


def test_grad_log_likelihood(setup):
    theta = jnp.arange(1.0, setup.model.n + 1)
    key = jax.random.key(42)
    r = jax.random.multivariate_normal(
        key=key, mean=jnp.zeros(setup.model.n), cov=setup.likelihood.Sigma_obs
    )
    y = setup.model(theta) + r

    computed = setup.likelihood.grad_log_likelihood(y, theta)
    expected = jax.grad(lambda t: setup.likelihood.log_likelihood(y, t))(theta)
    assert jnp.allclose(computed, expected, atol=1e-8)


def test_hessian(setup):
    theta = jnp.arange(1.0, setup.model.n + 1)
    A = setup.model.jacobian(theta)
    Sigma_inv = jnp.linalg.inv(setup.likelihood.Sigma_obs)  # inversion directe
    expected = -A.T @ Sigma_inv @ A
    assert jnp.allclose(expected, setup.likelihood.hessian(theta))


def test_hessian_matches_operator(setup):
    theta = jnp.ones(setup.model.n)
    H_dense = setup.likelihood.hessian(theta)
    H_op = setup.likelihood.hessian_operator(theta)
    v = jax.random.normal(jax.random.key(0), (setup.model.n,))
    assert jnp.allclose(H_dense @ v, H_op.matvec(v))


def test_hessian_matches_finite_difference(setup):
    theta = jnp.arange(1.0, setup.model.n + 1)
    key = jax.random.key(0)
    y = setup.model(theta) + jax.random.normal(key, (setup.model.n_obs,))

    H_analytic = setup.likelihood.hessian(theta)
    H_autodiff = jax.hessian(lambda t: setup.likelihood.log_likelihood(y, t))(theta)

    assert jnp.allclose(H_analytic, H_autodiff, atol=1e-8)


def test_sample_is_deterministic_given_key(setup):
    theta = jnp.ones(setup.model.n)
    key = jax.random.key(0)
    a = setup.likelihood.sample(key, theta, n_samples=5)
    b = setup.likelihood.sample(key, theta, n_samples=5)
    assert jnp.array_equal(a, b)


def test_different_keys_give_different_samples(setup):
    theta = jnp.ones(setup.model.n)
    a = setup.likelihood.sample(jax.random.key(0), theta)
    b = setup.likelihood.sample(jax.random.key(1), theta)
    assert not jnp.allclose(a, b)


def test_sample_shape(setup):
    theta = jnp.ones(setup.model.n)
    y = setup.likelihood.sample(jax.random.key(0), theta, n_samples=7)
    assert y.shape == (7, setup.model.n_obs)


@pytest.mark.slow("n")
def test_sample_moments(setup):
    theta = jnp.arange(1.0, setup.model.n + 1)
    n = 200_000
    y = setup.likelihood.sample(jax.random.key(0), theta, n_samples=n)

    mean_hat = y.mean(axis=0)
    cov_hat = jnp.cov(y.T)

    expected_mean = setup.model(theta)
    expected_cov = setup.likelihood.Sigma_obs

    # erreur-type de la moyenne : sigma / sqrt(n)
    tol_mean = 5.0 * jnp.sqrt(jnp.diag(expected_cov) / n)
    assert jnp.all(jnp.abs(mean_hat - expected_mean) < tol_mean)

    assert jnp.allclose(cov_hat, expected_cov, atol=5e-2)


@pytest.mark.slow("n")
def test_sample_matches_log_likelihood(setup):
    """La log-vraisemblance empirique moyenne doit approcher l'entropie
    différentielle négative de la gaussienne."""
    theta = jnp.ones(setup.model.n)
    n_samples = 50_000
    y = setup.likelihood.sample(jax.random.key(0), theta, n_samples=n_samples)

    ll = jax.vmap(lambda yi: setup.likelihood.log_likelihood(yi, theta))(y)

    d = setup.model.n_obs
    _, logdet = jnp.linalg.slogdet(setup.likelihood.Sigma_obs)
    expected = -0.5 * (d * jnp.log(2 * jnp.pi) + logdet + d)
    assert jnp.abs(ll.mean() - expected) < 0.05


def test_log_likelihood_with_design(setup):
    theta = setup.prior.mu
    design = jnp.array([0, 2])
    y = setup.model(theta, design)  # sans bruit → résidu nul
    ll = setup.likelihood.log_likelihood(y, theta, design)
    # résidu nul → quad = 0 → ll = -½(m log2π + logdet)
    m = 2
    chol = setup.likelihood._obs_chol(design)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(chol[0])))
    expected = -0.5 * (m * jnp.log(2 * jnp.pi) + logdet)
    assert jnp.allclose(ll, expected)


def test_all_methods_consistent_under_design(setup):
    theta = setup.prior.mu
    design = jnp.array([0, 2])
    key = jax.random.key(0)

    # toutes ces quantités doivent avoir la bonne forme
    y = setup.model(theta, design)
    assert y.shape == (2,)

    r = setup.likelihood.precision_weighted_residual(y, theta, design)
    assert r.shape == (2,)  # espace obs

    g = setup.likelihood.grad_log_likelihood(y, theta, design)
    assert g.shape == (4,)  # espace param

    H = setup.likelihood.hessian(theta, design)
    assert H.shape == (4, 4)

    s = setup.likelihood.sample(key, theta, design, n_samples=5)
    assert s.shape == (5, 2)  # 5 tirages, m=2

    ll = setup.likelihood.log_likelihood(y, theta, design)
    assert ll.shape == ()


def test_hessian_design_equals_row_selection(setup):
    """H(design) = -A[design]ᵀ Σ_sub⁻¹ A[design], sélection explicite."""
    theta = setup.prior.mu
    design = jnp.array([0, 2])

    A_full = setup.model.jacobian(theta)  # (p, d)
    A_sub = A_full[design]  # (m, d)
    Sigma_sub = setup.likelihood.Sigma_obs[jnp.ix_(design, design)]
    expected = -A_sub.T @ jnp.linalg.inv(Sigma_sub) @ A_sub

    assert jnp.allclose(expected, setup.likelihood.hessian(theta, design))
